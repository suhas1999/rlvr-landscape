"""Hessian of the expected reward J(theta) = E_{y ~ pi_theta}[R(y)] at a checkpoint.

Surrogate (DiCE / importance ratio, Foerster et al. 2018): with rollouts y_i ~ pi_theta held fixed,

    L(theta') = 1/P sum_prompts 1/G sum_i A_i * exp(s_i(theta') - s_i(theta)),   s_i = log pi(y_i | x)

is an unbiased estimate of J(theta') - baseline for theta' near theta, so grad L = grad J and
Hess L = Hess J at theta' = theta (the ratio supplies the score outer-product term that
sum A log pi misses). A_i = R_i - mean of the other rewards of the same prompt (leave-one-out
baseline: unbiased, and exactly zero for groups that are all right or all wrong, which are skipped).
pi is the plain temperature policy (logits / T, no top-p / top-k), matching hessian_rollouts.py.

Hessian-vector products: central differences of the exact gradient of L,
    H v = [grad L(theta + eps v) - grad L(theta - eps v)] / (2 eps),
in full fp32 (TF32 off, deterministic kernels). The same rollouts are used at both points, so
sampling noise cancels. eps is chosen per product: a forward-only probe measures each rollout's
slope d log pi(y_i) / d eps along v, and eps = min(eps_max, delta / max_i |slope_i|) keeps every
sequence log-ratio within delta nats (a whole-sequence log-likelihood can move by tens of nats
for a 1e-2 step along its own gradient, which would leave the second-order regime).

Noise control: the rollouts of every prompt are split into two halves A and B (same prompts).
Lanczos is driven by H_A; the curvature of its Ritz vectors is then read off H_B (held out),
which removes the upward bias of extreme eigenvalues of a sampled Hessian. H_B v comes from the
same gradient passes, so it costs nothing extra.

Outputs (<out-dir>/<name>_hessian.json):
  J, grad norms, Lanczos Ritz values (in-sample A, held-out B, full), Hutchinson trace / n
  (mean curvature a random unit direction sees), curvature along the gradient and along the
  training displacement theta - theta_base. Unit directions (HF parameter names, fp32) for the
  most negative / most positive Ritz vectors and the gradient go to <work-dir>/<name>/ for
  basin_radius.py --direction files.

    python landscape/reward_hessian.py --model-ckpt <ckpt> --rollouts r.jsonl --name gs8_step140
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-ckpt", required=True)
    p.add_argument("--rollouts", required=True, help="hessian_rollouts.py output")
    p.add_argument("--name", required=True)
    p.add_argument("--base-ckpt", default=None, help="For the curvature along theta - theta_base.")
    p.add_argument("--out-dir", default="results/hessian")
    p.add_argument("--work-dir", default="/scratch/micah/vm2825/hessian_tmp")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--eps", type=float, default=5e-2, help="Largest finite-difference step (L2 norm).")
    p.add_argument("--delta", type=float, default=0.02,
                   help="Largest per-rollout log-ratio change allowed by the step (nats).")
    p.add_argument("--probe-eps", type=float, default=1e-3, help="Step of the slope probe.")
    p.add_argument("--lanczos-iters", type=int, default=16)
    p.add_argument("--hutchinson", type=int, default=6)
    p.add_argument("--save-k", type=int, default=2, help="Most negative Ritz vectors to save.")
    p.add_argument("--max-tokens-per-mb", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--validate", action="store_true",
                   help="Finite-difference / symmetry / eps checks before the main computation.")
    return p.parse_args()


class Surrogate:
    """Gradients of the per-half importance-ratio surrogates, in fp32."""

    def __init__(self, model, rows, temperature, max_tokens, device):
        self.model, self.T, self.device = model, temperature, device
        self.params = [p for _, p in model.named_parameters()]
        self.names = [n for n, _ in model.named_parameters()]
        self.shapes = [p.shape for p in self.params]
        self.numels = [p.numel() for p in self.params]
        self.n = sum(self.numels)

        by_prompt = {}
        for r in rows:
            by_prompt.setdefault(r["prompt_index"], []).append(r)
        P = len(by_prompt)
        seqs = [[], []]
        self.rewards = []
        for group in by_prompt.values():
            half = len(group) // 2
            for h, members in enumerate((group[:half], group[half:])):
                rs = np.array([m["reward"] for m in members])
                for m, r in zip(members, rs):
                    self.rewards.append(r)
                    adv = r - (rs.sum() - r) / (len(rs) - 1)   # leave-one-out baseline
                    if adv != 0:
                        seqs[h].append((m["prompt_token_ids"], m["completion_token_ids"],
                                        adv / (P * len(members))))
        self.n_active = [len(s) for s in seqs]
        self.batches = [self._pack(s, max_tokens) for s in seqs]
        with torch.no_grad():
            self.s0 = [[self._logprob(b) for b in hb] for hb in self.batches]

    def _pack(self, seqs, max_tokens):
        seqs = sorted(seqs, key=lambda s: len(s[0]) + len(s[1]))
        batches, cur = [], []
        for s in seqs:
            length = len(s[0]) + len(s[1])
            if cur and (len(cur) + 1) * max(length, max(len(c[0]) + len(c[1]) for c in cur)) > max_tokens:
                batches.append(cur)
                cur = []
            cur.append(s)
        if cur:
            batches.append(cur)
        out = []
        for b in batches:
            L = max(len(p) + len(c) for p, c, _ in b)
            ids = torch.zeros(len(b), L, dtype=torch.long)
            mask = torch.zeros(len(b), L, dtype=torch.long)
            bi, pos, tgt = [], [], []
            for i, (p, c, _) in enumerate(b):
                seq = list(p) + list(c)
                ids[i, :len(seq)] = torch.tensor(seq)
                mask[i, :len(seq)] = 1
                for t in range(len(c)):
                    bi.append(i)
                    pos.append(len(p) + t - 1)
                    tgt.append(c[t])
            out.append({"ids": ids.to(self.device), "mask": mask.to(self.device),
                        "bi": torch.tensor(bi, device=self.device),
                        "pos": torch.tensor(pos, device=self.device),
                        "tgt": torch.tensor(tgt, device=self.device),
                        "coef": torch.tensor([a for _, _, a in b], dtype=torch.float32, device=self.device),
                        "size": len(b)})
        return out

    def _logprob(self, b):
        """Sum of completion-token log-probs per sequence under pi = softmax(logits / T)."""
        hidden = self.model.model(input_ids=b["ids"], attention_mask=b["mask"]).last_hidden_state
        h = hidden[b["bi"], b["pos"]]
        s = torch.zeros(b["size"], dtype=torch.float32, device=self.device)
        for start in range(0, h.shape[0], 2048):   # lm_head in chunks: [tokens, vocab] is large
            chunk = slice(start, start + 2048)
            logits = self.model.lm_head(h[chunk]).float() / self.T
            lp = torch.log_softmax(logits, dim=-1).gather(1, b["tgt"][chunk, None])[:, 0]
            s = s.index_add(0, b["bi"][chunk], lp)
        return s

    def grads(self):
        """(grad L_A, grad L_B) as flat fp32 GPU tensors, plus (L_A, L_B)."""
        out, values = [], []
        for h in range(2):
            for p in self.params:
                p.grad = None
            value = 0.0
            for b, s0 in zip(self.batches[h], self.s0[h]):
                s = self._logprob(b)
                loss = (b["coef"] * torch.exp(s - s0)).sum()
                loss.backward()
                value += loss.item()
            flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                              for p in self.params])
            out.append(flat)
            values.append(value)
        for p in self.params:
            p.grad = None
        return out, values

    @torch.no_grad()
    def max_slope(self, theta0, v, e0):
        """max_i |d s_i / d eps| along v over the active rollouts (central forward-only probe)."""
        s_pm = []
        for sign in (1, -1):
            self.set_flat(theta0 + sign * e0 * v)
            s_pm.append(torch.cat([self._logprob(b) for hb in self.batches for b in hb]))
        self.set_flat(theta0)
        return ((s_pm[0] - s_pm[1]).abs().max() / (2 * e0)).item()

    def get_flat(self):
        return torch.cat([p.detach().reshape(-1) for p in self.params])

    @torch.no_grad()
    def set_flat(self, flat):
        offset = 0
        for p, k in zip(self.params, self.numels):
            p.copy_(flat[offset:offset + k].view_as(p))
            offset += k

    def to_named(self, flat_cpu):
        out, offset = {}, 0
        for name, shape, k in zip(self.names, self.shapes, self.numels):
            out[name] = flat_cpu[offset:offset + k].view(shape).clone()
            offset += k
        return out


def lanczos(hvp, v, m, log=print):
    """Lanczos driven by the first operator of hvp(v) -> (H_A v, H_B v), full reorthogonalization.

    Returns the projections MA = V^T H_A V, MB = V^T H_B V (m x m), the basis V (CPU) and the
    last beta (for residuals). Ritz pairs of MA approximate H_A's extreme eigenpairs; y^T MB y is
    H_B's curvature along the same Ritz vector V y.
    """
    device = v.device
    V, MA, MB = [], np.zeros((m, m)), np.zeros((m, m))
    beta_prev, v_prev, beta, t0 = 0.0, None, 0.0, time.time()
    for j in range(m):
        V.append(v.cpu())
        HA, HB = hvp(v)
        HA_c, HB_c = HA.cpu(), HB.cpu()
        for k in range(j + 1):
            MA[k, j] = MA[j, k] = float(V[k] @ HA_c)
            MB[k, j] = MB[j, k] = float(V[k] @ HB_c)
        w = HA - MA[j, j] * v
        if v_prev is not None:
            w = w - beta_prev * v_prev
        w_c = w.cpu()
        for _ in range(2):                    # full reorthogonalization
            for k in range(j + 1):
                w_c.add_(V[k], alpha=-float(V[k] @ w_c))
        beta = float(w_c.norm())
        log(f"[lanczos {j + 1}/{m}] alpha {MA[j, j]:+.4e} beta {beta:.4e} (B: {MB[j, j]:+.4e}) "
            f"| {time.time() - t0:.0f} s")
        if beta < 1e-10 * max(1.0, abs(MA[j, j])):
            return MA[:j + 1, :j + 1], MB[:j + 1, :j + 1], V, beta
        v_prev, v, beta_prev = v, (w_c / beta).to(device), beta
        del HA, HB, w
    return MA, MB, V, beta


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM

    device = "cuda"
    t_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model_ckpt, dtype=torch.float32,
                                                 attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()   # checkpointing is only active in train mode; Qwen3 has no dropout

    rows = [json.loads(line) for line in open(args.rollouts)]
    S = Surrogate(model, rows, args.temperature, args.max_tokens_per_mb, device)
    J = float(np.mean(S.rewards))
    print(f"[data] {len(rows)} rollouts, J = {J:.3f}, active (nonzero advantage) A/B = {S.n_active}, "
          f"micro-batches A/B = {[len(b) for b in S.batches]}, n = {S.n}", flush=True)
    result = {"name": args.name, "ckpt": args.model_ckpt, "n_params": S.n, "J": J,
              "n_rollouts": len(rows), "n_active": S.n_active, "eps_max": args.eps, "delta": args.delta,
              "temperature": args.temperature}
    if min(S.n_active) == 0:
        # every prompt's rollouts in a half are all right or all wrong: zero gradient / Hessian estimate
        result["insufficient"] = "a half has no nonzero-advantage rollouts"
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.out_dir) / f"{args.name}_hessian.json").write_text(json.dumps(result, indent=2))
        print("[data] insufficient: no nonzero-advantage rollouts in a half; wrote stub and stopped", flush=True)
        sys.exit(3)
    theta0 = S.get_flat()
    eps_used = []

    def step_for(v):
        return min(args.eps, args.delta / max(S.max_slope(theta0, v, args.probe_eps), 1e-12))

    def hvp(v, e=None):
        e = step_for(v) if e is None else e
        eps_used.append(e)
        S.set_flat(theta0 + e * v)
        gp, _ = S.grads()
        S.set_flat(theta0 - e * v)
        gm, _ = S.grads()
        S.set_flat(theta0)
        return [(a - b) / (2 * e) for a, b in zip(gp, gm)]

    def rand_unit(seed):
        g = torch.Generator(device=device).manual_seed(seed)
        v = torch.randn(S.n, generator=g, device=device)
        return v / v.norm()

    t0 = time.time()
    (gA, gB), (LA, LB) = S.grads()
    t_grad = time.time() - t0
    g = (gA + gB) / 2
    result.update(grad_norm_A=gA.norm().item(), grad_norm_B=gB.norm().item(), grad_norm=g.norm().item(),
                  grad_cos_AB=(gA @ gB / (gA.norm() * gB.norm() + 1e-30)).item(),
                  seconds_per_gradient=t_grad)
    print(f"[grad] |g| = {g.norm().item():.4e} (A {gA.norm().item():.3e}, B {gB.norm().item():.3e}, "
          f"cos(A,B) = {result['grad_cos_AB']:.3f}) | {t_grad:.0f} s per gradient pass", flush=True)

    if args.validate:
        # 1. ratio is exactly 1 at theta: recompute s and compare with s0.
        with torch.no_grad():
            d = max((S._logprob(b) - s0).abs().max().item() for hb, s0h in zip(S.batches, S.s0)
                    for b, s0 in zip(hb, s0h))
        # 2. directional derivative: [L(theta+e u) - L(theta-e u)] / 2e vs g . u
        u, w = rand_unit(101), rand_unit(202)
        e_u = step_for(u)
        S.set_flat(theta0 + e_u * u)
        _, Lp = S.grads()
        S.set_flat(theta0 - e_u * u)
        _, Lm = S.grads()
        S.set_flat(theta0)
        fd = (sum(Lp) - sum(Lm)) / (2 * e_u) / 2
        an = (g @ u).item()
        # 3. eps sensitivity and 4. symmetry
        Hu = hvp(u, e_u)
        Hu3 = hvp(u, e_u / 3)
        Hw = hvp(w)
        HuF, Hu3F, HwF = (Hu[0] + Hu[1]) / 2, (Hu3[0] + Hu3[1]) / 2, (Hw[0] + Hw[1]) / 2
        result["validation"] = {
            "max_abs_logratio_at_theta": d, "eps_random_dir": e_u,
            "directional_derivative_fd": fd, "directional_derivative_grad": an,
            "hvp_eps_vs_eps3_rel_diff": ((HuF - Hu3F).norm() / HuF.norm()).item(),
            "uHu_eps": (u @ HuF).item(), "uHu_eps3": (u @ Hu3F).item(),
            "symmetry_wHu": (w @ HuF).item(), "symmetry_uHw": (u @ HwF).item(),
        }
        print("[validate]", json.dumps(result["validation"]), flush=True)
        del Hu, Hu3, Hw, HuF, Hu3F, HwF

    # ---------------- Lanczos on H_A, Rayleigh-Ritz on H_B and H ----------------
    t0 = time.time()
    MA, MB, V, beta = lanczos(hvp, rand_unit(args.seed + 7), args.lanczos_iters)
    m = MA.shape[0]
    evals, Y = np.linalg.eigh((MA + MA.T) / 2)
    MF = (MA + MB) / 2
    order = np.argsort(evals)                # most negative first
    ritz = []
    for rank, idx in enumerate(order):
        y = Y[:, idx]
        ritz.append({"rank_from_most_negative": rank, "lambda_A_in_sample": float(evals[idx]),
                     "lambda_B_held_out": float(y @ MB @ y), "lambda_full": float(y @ MF @ y),
                     "residual_A": float(beta * abs(y[-1]))})
    result["lanczos"] = {"iters": m, "ritz": ritz, "seconds": time.time() - t0}
    neg, pos = ritz[0], ritz[-1]
    print(f"[lanczos] most negative: A {neg['lambda_A_in_sample']:+.4e} | B held-out "
          f"{neg['lambda_B_held_out']:+.4e} | most positive: A {pos['lambda_A_in_sample']:+.4e} "
          f"| B {pos['lambda_B_held_out']:+.4e}", flush=True)

    # Save unit directions (HF names) for the reward sweeps.
    work = Path(args.work_dir) / args.name
    work.mkdir(parents=True, exist_ok=True)
    saved = {}

    def combine(y):
        out = torch.zeros(S.n)
        for c, Vk in zip(y, V):
            out.add_(Vk, alpha=float(c))
        return out

    def save(tag, flat_cpu):
        flat_cpu = flat_cpu / flat_cpu.norm()
        path = work / f"dir_{tag}.pt"
        torch.save(S.to_named(flat_cpu), path)
        saved[tag] = str(path)

    for r in range(args.save_k):
        y = Y[:, order[r]]
        save(f"eigneg{r + 1}", combine(y))
    save("eigpos1", combine(Y[:, order[-1]]))
    save("grad", g.cpu())
    del V

    # ---------------- Hutchinson trace ----------------
    t0 = time.time()
    tr = []
    for i in range(args.hutchinson):
        gen = torch.Generator(device=device).manual_seed(1000 + i)
        z = torch.randint(0, 2, (S.n,), generator=gen, device=device).float() * 2 - 1
        zn = z / math.sqrt(S.n)                 # unit norm: z^T H z / n = zn^T H zn
        HA, HB = hvp(zn)
        tr.append(((zn @ HA).item(), (zn @ HB).item()))
        print(f"[hutchinson {i + 1}/{args.hutchinson}] A {tr[-1][0]:+.4e} B {tr[-1][1]:+.4e}", flush=True)
    tr = np.array(tr)
    full = tr.mean(axis=1)
    result["trace_over_n"] = {"A": float(tr[:, 0].mean()), "B": float(tr[:, 1].mean()),
                              "full": float(full.mean()),
                              "full_se": float(full.std(ddof=1) / math.sqrt(len(full))),
                              "samples": tr.tolist(), "seconds": time.time() - t0}

    # ---------------- curvature along the gradient and the training displacement ----------------
    def curvature(unit, label):
        HA, HB = hvp(unit)
        result[f"curv_{label}"] = {"A": (unit @ HA).item(), "B": (unit @ HB).item(),
                                   "full": (unit @ (HA + HB) / 2).item()}
        print(f"[curv {label}] {json.dumps(result[f'curv_{label}'])}", flush=True)

    curvature(g / g.norm(), "grad")
    if args.base_ckpt:
        base = AutoModelForCausalLM.from_pretrained(args.base_ckpt, dtype=torch.float32)
        disp = theta0 - torch.cat([p.detach().reshape(-1) for _, p in base.named_parameters()]).to(device)
        del base
        result["displacement_norm"] = disp.norm().item()
        if disp.norm() > 0:
            d_unit = disp / disp.norm()
            result["cos_grad_displacement"] = (g @ d_unit / g.norm()).item()
            curvature(d_unit, "displacement")

    result["eps_used"] = {"min": min(eps_used), "median": float(np.median(eps_used)), "max": max(eps_used),
                          "n": len(eps_used)}
    result["directions"] = saved
    result["seconds_total"] = time.time() - t_start
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}_hessian.json").write_text(json.dumps(result, indent=2))
    print(f"Done {args.name} in {result['seconds_total'] / 60:.1f} min -> {out / (args.name + '_hessian.json')}",
          flush=True)


if __name__ == "__main__":
    main()
