"""Apply theta = theta_ckpt + coefficient * direction inside a running vLLM engine.

The direction file is a dict of HF-named tensors. Inside the vLLM worker it is
mapped onto vLLM's fused parameters (q/k/v -> qkv_proj, gate/up -> gate_up_proj),
and every coefficient is rebuilt from an fp32 CPU copy of the checkpoint, so
there is no accumulated rounding error between alpha points.
"""

import math
from pathlib import Path

import torch


def install_direction_on_vllm(llm, direction_path):
    direction_path = str(Path(direction_path).resolve())

    if not Path(direction_path).is_file():
        raise FileNotFoundError(direction_path)

    def worker_install(model):
        # 重新安装前必须先恢复原点。
        old_coefficient = getattr(
            model, "_reward_landscape_coefficient", 0.0
        )
        if old_coefficient != 0.0:
            raise RuntimeError(
                "Restore coefficient=0 before installing a new direction."
            )

        hf_direction = torch.load(
            direction_path,
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(hf_direction, dict):
            raise TypeError("Direction file must contain a dictionary.")

        params = dict(model.named_parameters())
        mapped_direction = {}
        consumed = set()

        for name, param in params.items():
            if param.dtype not in (
                torch.float32,
                torch.float16,
                torch.bfloat16,
            ):
                raise RuntimeError(
                    f"Unsupported parameter dtype: {name}: {param.dtype}"
                )

            # HF -> vLLM 融合参数映射。
            if ".qkv_proj." in name:
                source_names = [
                    name.replace(".qkv_proj.", ".q_proj."),
                    name.replace(".qkv_proj.", ".k_proj."),
                    name.replace(".qkv_proj.", ".v_proj."),
                ]
            elif ".gate_up_proj." in name:
                source_names = [
                    name.replace(".gate_up_proj.", ".gate_proj."),
                    name.replace(".gate_up_proj.", ".up_proj."),
                ]
            else:
                source_names = [name]

            missing = [
                source_name
                for source_name in source_names
                if source_name not in hf_direction
            ]
            if missing:
                raise RuntimeError(
                    f"Cannot map vLLM parameter {name}.\n"
                    f"Missing HF directions: {missing}"
                )

            parts = []
            for source_name in source_names:
                tensor = hf_direction[source_name]

                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(
                        f"{source_name}: direction must be a tensor."
                    )

                tensor = tensor.detach().to(
                    device="cpu",
                    dtype=torch.float32,
                )

                if not torch.isfinite(tensor).all().item():
                    raise RuntimeError(
                        f"{source_name}: direction contains NaN/Inf."
                    )

                parts.append(tensor)

            # 单 GPU、非量化 Qwen3 的融合权重沿输出维拼接。
            if len(parts) == 1:
                mapped = parts[0]
            else:
                mapped = torch.cat(parts, dim=0)

            if mapped.shape != param.shape:
                raise RuntimeError(
                    f"Shape mismatch for {name}: "
                    f"mapped={tuple(mapped.shape)}, "
                    f"vLLM={tuple(param.shape)}"
                )

            mapped_direction[name] = mapped.contiguous()
            consumed.update(source_names)

        # 防止 HF 方向中有参数被静默丢弃。
        unused = sorted(set(hf_direction) - consumed)

        # Displacement files are built from state_dict(), which also lists
        # lm_head.weight when it is tied to embed_tokens. vLLM exposes the tied
        # weight only once, so drop lm_head if it duplicates the embedding.
        if (
            "lm_head.weight" in unused
            and "lm_head.weight" not in params
            and "model.embed_tokens.weight" in hf_direction
            and torch.equal(
                hf_direction["lm_head.weight"],
                hf_direction["model.embed_tokens.weight"],
            )
        ):
            unused.remove("lm_head.weight")

        if unused:
            raise RuntimeError(
                "Some HF directions were not used. "
                "Check architecture or tied-weight handling:\n"
                + "\n".join(unused)
            )

        # 只在首次安装时保存原点。
        base_state = getattr(
            model, "_reward_landscape_base_state", None
        )

        if base_state is None:
            base_state = {
                name: param.detach().to(
                    device="cpu",
                    dtype=torch.float32,
                ).clone()
                for name, param in params.items()
            }
        else:
            if set(base_state) != set(params):
                raise RuntimeError("Saved base parameter names differ.")

            for name, param in params.items():
                if base_state[name].shape != param.shape:
                    raise RuntimeError(
                        f"Saved base shape differs for {name}."
                    )

        # 完整检查成功后，再安装方向。
        model._reward_landscape_base_state = base_state
        model._reward_landscape_direction = mapped_direction
        model._reward_landscape_direction_path = direction_path
        model._reward_landscape_coefficient = 0.0

        return {
            "hf_tensors_used": len(consumed),
            "vllm_tensors_mapped": len(mapped_direction),
            "vllm_tensors_total": len(params),
        }

    results = llm.apply_model(worker_install)
    print("Direction installed:", results)
    return results


def set_vllm_coefficient(llm, coefficient):
    coefficient = float(coefficient)

    if not math.isfinite(coefficient):
        raise ValueError("coefficient must be finite.")

    def worker_set_coefficient(model):
        direction = getattr(
            model, "_reward_landscape_direction", None
        )
        base_state = getattr(
            model, "_reward_landscape_base_state", None
        )

        if direction is None or base_state is None:
            raise RuntimeError("Install the direction first.")

        params = dict(model.named_parameters())

        if set(params) != set(direction) or set(params) != set(base_state):
            raise RuntimeError("Incomplete parameter mapping.")

        with torch.no_grad():
            for name, param in params.items():
                base = base_state[name]

                if coefficient == 0.0:
                    # 从保存的原始权重直接恢复。
                    updated = base
                else:
                    # CPU FP32 计算，每次都从原点构造。
                    updated = torch.add(
                        base,
                        direction[name],
                        alpha=coefficient,
                    )

                if not torch.isfinite(updated).all().item():
                    raise RuntimeError(
                        f"Non-finite updated weight: {name}"
                    )

                # 最后才转换为模型精度。
                device_weight = updated.to(
                    device=param.device,
                    dtype=param.dtype,
                )

                if not torch.isfinite(device_weight).all().item():
                    raise RuntimeError(
                        f"Weight overflow after dtype conversion: {name}"
                    )

                param.copy_(device_weight)

                del device_weight
                del updated

        model._reward_landscape_coefficient = coefficient

        return {
            "coefficient": coefficient,
            "updated_parameters": len(params),
        }

    return llm.apply_model(worker_set_coefficient)
