"""Filter-normalized directions for Qwen3 (Li et al. 2018), applied inside vLLM.

Spec:
  1. Canonicalize: fold RMSNorm gammas into the following Linear layers, balance each
     MLP neuron (up/down), each KV head (v/o) and each RoPE pair of q_norm/k_norm.
     Function preserving.
  2. Directions: every 2-D weight gets a Gaussian whose rows are rescaled to the norms of
     the weight's rows. 1-D params get 0, except (optionally) q_norm/k_norm, which get
     noise rescaled per RoPE pair. CPU generator, explicit seed.
  3. Every grid point is rebuilt from the fp32 base: theta = base + a*d1 + b*d2.
"""

import math

import torch

from rlvr_landscape.vllm_utils import apply_to_model

EPS = 1e-12


def load_fp32_params(model_dir):
    """named_parameters() of the HF model in fp32 (a tied lm_head is listed once)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    params = {name: p.detach().clone() for name, p in model.named_parameters()}
    config = model.config
    del model

    biases = [name for name in params if name.endswith(".bias")]
    if biases:
        raise RuntimeError(f"Expected a bias-free Qwen3, found: {biases[:5]}")
    return params, config


@torch.no_grad()
def canonicalize(params, config):
    """Function-preserving rescaling; returns a new dict, input untouched."""
    p = {name: t.clone() for name, t in params.items()}

    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    num_q = config.num_attention_heads
    num_kv = config.num_key_value_heads
    group = num_q // num_kv

    for layer in range(config.num_hidden_layers):
        pre = f"model.layers.{layer}."

        # a. input_layernorm gamma -> input columns of q/k/v.
        gamma = p[pre + "input_layernorm.weight"]
        for proj in ("q_proj", "k_proj", "v_proj"):
            p[pre + f"self_attn.{proj}.weight"].mul_(gamma[None, :])
        gamma.fill_(1.0)

        # b. post_attention_layernorm gamma -> input columns of gate/up.
        gamma = p[pre + "post_attention_layernorm.weight"]
        for proj in ("gate_proj", "up_proj"):
            p[pre + f"mlp.{proj}.weight"].mul_(gamma[None, :])
        gamma.fill_(1.0)

        # c. Balance each MLP neuron between up (row i) and down (column i).
        up = p[pre + "mlp.up_proj.weight"]
        down = p[pre + "mlp.down_proj.weight"]
        a = up.norm(dim=1).clamp_min(EPS)
        b = down.norm(dim=0).clamp_min(EPS)
        s = torch.sqrt(b / a)
        up.mul_(s[:, None])
        down.div_(s[None, :])

        # d. Balance each KV head between v (its rows) and o (its query heads' columns).
        v = p[pre + "self_attn.v_proj.weight"]
        o = p[pre + "self_attn.o_proj.weight"]
        for h in range(num_kv):
            v_h = v[h * head_dim:(h + 1) * head_dim, :]
            o_h = o[:, h * group * head_dim:(h + 1) * group * head_dim]
            s_h = math.sqrt(
                max(o_h.norm().item(), EPS) / max(v_h.norm().item(), EPS)
            )
            v_h.mul_(s_h)
            o_h.div_(s_h)

        # e. Balance q_norm / k_norm gammas per RoPE pair (j, j + head_dim/2). Scores depend on
        #    gamma_q * gamma_k per pair, and a per-pair scale commutes with the rotation.
        q_norm = p.get(pre + "self_attn.q_norm.weight")
        k_norm = p.get(pre + "self_attn.k_norm.weight")
        if q_norm is not None and k_norm is not None:
            half = q_norm.numel() // 2
            q_pair = torch.sqrt(q_norm[:half] ** 2 + q_norm[half:] ** 2).clamp_min(EPS)
            k_pair = torch.sqrt(k_norm[:half] ** 2 + k_norm[half:] ** 2).clamp_min(EPS)
            s = torch.sqrt(k_pair / q_pair).repeat(2)
            q_norm.mul_(s)
            k_norm.div_(s)

    return p


QK_NORMS = ("self_attn.q_norm.weight", "self_attn.k_norm.weight")


def filter_normalized_direction(params, seed, perturb_qk_norm=False):
    """Row-wise filter-normalized Gaussian direction; zero for 1-D params.

    With perturb_qk_norm, q_norm/k_norm gammas also get noise, normalized per RoPE pair
    (j, j + head_dim/2) to the gamma's pair norm. They use their own generator, so the
    matrix noise is identical with or without it.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    norm_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1_000_003)
    direction = {}
    for name, weight in params.items():
        if weight.dim() == 1 and perturb_qk_norm and name.endswith(QK_NORMS):
            half = weight.numel() // 2
            gamma = torch.stack([weight[:half], weight[half:]]).float()  # (2, pairs)
            r = torch.randn(gamma.shape, generator=norm_generator, dtype=torch.float32)
            r = r / r.norm(dim=0, keepdim=True).clamp_min(EPS)
            r = r * gamma.norm(dim=0, keepdim=True)
            direction[name] = torch.cat([r[0], r[1]])
        elif weight.dim() == 1:
            direction[name] = torch.zeros_like(weight, dtype=torch.float32)
        elif weight.dim() == 2:
            r = torch.randn(weight.shape, generator=generator, dtype=torch.float32)
            r = r / r.norm(dim=1, keepdim=True).clamp_min(EPS)
            direction[name] = r * weight.float().norm(dim=1, keepdim=True)
        else:
            raise RuntimeError(f"Unexpected {weight.dim()}-D parameter {name}")
    return direction


@torch.no_grad()
def mean_nll(model, token_lists, device):
    total, count = 0.0, 0
    for ids in token_lists:
        x = torch.tensor([ids], device=device)
        logits = model(x).logits[0, :-1].float()
        nll = torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="sum")
        total += nll.item()
        count += len(ids) - 1
    return total / count


@torch.no_grad()
def check_function_preserved(model_dir, canonical, token_lists, device="cuda"):
    """Mean NLL of the original vs canonicalized weights (fp32, TF32 off)."""
    from transformers import AutoModelForCausalLM

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.to(device).eval()
    before = mean_nll(model, token_lists, device)

    missing, unexpected = model.load_state_dict(canonical, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected[:5]}")
    missing = [k for k in missing if k != "lm_head.weight"]
    if missing:
        raise RuntimeError(f"Canonical state misses: {missing[:5]}")

    after = mean_nll(model, token_lists, device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return before, after


# ------------------------------------------------------------------
# vLLM side
# ------------------------------------------------------------------

def _to_vllm(vllm_params, hf):
    """Map HF-named tensors onto vLLM params (fused qkv_proj / gate_up_proj)."""
    mapped, used = {}, set()
    for name, param in vllm_params.items():
        if ".qkv_proj." in name:
            sources = [name.replace(".qkv_proj.", f".{x}.") for x in ("q_proj", "k_proj", "v_proj")]
        elif ".gate_up_proj." in name:
            sources = [name.replace(".gate_up_proj.", f".{x}.") for x in ("gate_proj", "up_proj")]
        else:
            sources = [name]
        missing = [s for s in sources if s not in hf]
        if missing:
            raise RuntimeError(f"vLLM param {name}: missing {missing}")
        tensor = torch.cat([hf[s].float() for s in sources], dim=0)
        if tensor.shape != param.shape:
            raise RuntimeError(f"{name}: {tuple(tensor.shape)} vs vLLM {tuple(param.shape)}")
        mapped[name] = tensor.contiguous()
        used.update(sources)
    unused = sorted(set(hf) - used)
    if unused:
        raise RuntimeError(f"Unused HF tensors: {unused[:5]}")
    return mapped


def install_plane(llm, base_path, direction_paths):
    """Load base and directions into the vLLM worker (CPU fp32, mapped to vLLM names)."""

    def worker(model):
        params = dict(model.named_parameters())
        model._plane_base = _to_vllm(params, torch.load(base_path, weights_only=True))
        model._plane_dirs = [
            _to_vllm(params, torch.load(path, weights_only=True))
            for path in direction_paths
        ]
        return {"vllm_params": len(params), "directions": len(direction_paths)}

    return apply_to_model(llm, worker)


def set_plane_point(llm, coefficients):
    """theta = base + sum_k c_k * d_k, rebuilt from the base every time."""
    coefficients = [float(c) for c in coefficients]

    def worker(model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                weight = model._plane_base[name].clone()
                for c, direction in zip(coefficients, model._plane_dirs):
                    if c != 0.0:
                        weight.add_(direction[name], alpha=c)
                if not torch.isfinite(weight).all():
                    raise RuntimeError(f"Non-finite weight {name} at {coefficients}")
                param.copy_(weight.to(device=param.device, dtype=param.dtype))
        return coefficients

    return apply_to_model(llm, worker)
