"""Check that set_plane_point moves the vLLM weights by exactly alpha (unit direction).

    python landscape/check_applied_norm.py --model-ckpt <ckpt> --alphas 1 5 25
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
import torch
from vllm import LLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlvr_landscape.filter_norm import install_plane, load_fp32_params, set_plane_point, unit_direction  # noqa: E402
from rlvr_landscape.vllm_utils import apply_to_model  # noqa: E402

def snapshot(model):
    model._snap = {n: q.detach().float().cpu().clone() for n, q in model.named_parameters()}
    return len(model._snap)


def dist(model):
    return sum(((q.detach().float().cpu() - model._snap[n]) ** 2).sum().item()
               for n, q in model.named_parameters()) ** 0.5


def requires_grad_split(model):
    """What a `if not parameter.requires_grad: continue` loop (Erin's install) would perturb."""
    on = [(n, q.numel()) for n, q in model.named_parameters() if q.requires_grad]
    total = sum(q.numel() for _, q in model.named_parameters())
    return {"requires_grad_params": sum(c for _, c in on), "total_params": total,
            "requires_grad_tensors": len(on), "examples": [n for n, _ in on[:6]]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-ckpt", required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=[1.0, 5.0, 25.0])
    p.add_argument("--work-dir", default=os.path.join(os.environ.get("WORK", "/tmp"), "basin_tmp", "normcheck"))
    args = p.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    params, _ = load_fp32_params(args.model_ckpt)
    torch.save(params, work / "base.pt")
    torch.save(unit_direction(params, 0), work / "dir0.pt")
    del params

    llm = LLM(model=args.model_ckpt, dtype="float32", gpu_memory_utilization=0.5, max_model_len=4096)
    install_plane(llm, str(work / "base.pt"), [str(work / "dir0.pt")])
    print("requires_grad split:", apply_to_model(llm, requires_grad_split)[0], flush=True)
    set_plane_point(llm, [0.0])
    apply_to_model(llm, snapshot)
    for a in args.alphas:
        set_plane_point(llm, [a])
        print(f"alpha {a:g}: ||theta(alpha) - theta(0)|| in vLLM = {apply_to_model(llm, dist)[0]:.4f}", flush=True)


if __name__ == "__main__":
    main()
