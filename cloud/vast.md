# Running on Vast.ai: rules and lessons

Budget is small, so every rule below is about not paying for idle GPUs and never losing results.

## The rules

1. **Never leave a box running unattended without the watcher.** Every run is followed by
   `cloud/finish_and_destroy.sh` (under `caffeinate` so the laptop does not sleep). It downloads,
   verifies the CSV line count, and only then destroys. If it cannot verify, it *stops* the box
   instead (disk kept, GPU billing ends).
2. **Destroy, don't stop, when done.** Stopped instances still bill for disk.
3. **After every session, check:** `uvx vastai show instances` says 0 instances, and note the credit
   (`uvx vastai show user`).
4. **Dry-run first** (3x3 grid, 4 prompts) on any new code path before a full grid. It costs ~$0.10.
5. **Keys never go in the chat or in git.** Run `wandb login` / `huggingface-cli login` yourself on the
   box. The Vast key lives in `~/.config/vastai/vast_api_key` (`uvx vastai set api-key ...` in your own
   terminal).
6. **Don't mix eager and CUDA-graph runs** in one comparison (they differ by ~0.04 in log-prob). Default
   is graphs (`RLVR_ENFORCE_EAGER=0`).

## Picking a machine

```bash
bash cloud/vast.sh offers A100_SXM4          # or A100_PCIE, H100_SXM
```

- **A100 40GB on-demand** is the default: ~$0.56-0.72/hr. Qwen3-0.6B in fp32 fits easily.
- Need **driver >= 580** (torch 2.13 is CUDA 13). The `offers` filter (`cuda_vers>=13.0`) handles it.
- **On-demand, not interruptible**: interruptible saves ~$0.10/hr but a preemption kills the sweep.
  (The CSV is resumable, but you pay setup + startup again.)
- One GPU. The code is single-GPU; 4 GPUs only help for independent parallel jobs.
- If setup is slow, test the network early (`git clone` taking minutes = bad box). Destroy and pick
  another rather than waiting. A California box in Oct 2026 had ~230KB/s and could not clone.
  Advertised Net_down is not reliable: a Georgia A100 40GB (machine 117301) listed 1650 Mbps
  but downloaded at ~1 MB/s. Quick check on the box:
  `a=$(awk '/eth0/{print $2}' /proc/net/dev); sleep 10; b=$(awk '/eth0/{print $2}' /proc/net/dev); echo $(( (b-a)/10000000 )) MB/s`
  Normal setup (uv sync + model) takes 2-5 min; still downloading after ~5 min = drop the box.

## Lifecycle

```bash
bash cloud/vast.sh create OFFER_ID                  # onstart runs setup_instance.sh automatically
bash cloud/vast.sh log INSTANCE_ID                  # wait for "Setup done" (~3-5 min)
bash cloud/vast.sh ssh INSTANCE_ID
```

On the box, everything is under **`/workspace`** (repo, models, uv/HF/vLLM caches), not `/root`.

Launch detached, with stdin closed so the ssh call returns:

```bash
cd /workspace/rlvr-landscape && source cloud/env.sh
nohup env PYTHONPATH=. python landscape/2d_perturb.py --model-ckpt $BASE_MODEL \
    --output-dir /workspace/OUT --grid-range 0.12 --grid-points 11 \
    > /workspace/run2d.log 2>&1 < /dev/null &
```

Then on the laptop:

```bash
caffeinate -ims bash cloud/finish_and_destroy.sh INSTANCE_ID HOST PORT /workspace/OUT results/OUT
# MAX_HOURS (default 4) and STALL_MIN (default 40) end a run early, keeping partial results.
```

`uvx vastai ssh-url INSTANCE_ID` gives HOST and PORT.

## Killing processes on the box

- Use the bracket trick so `pkill` does not match its own ssh shell:
  `pkill -f '[l]andscape/2d_perturb'`
- vLLM leaves an orphan `VLLM::EngineCore` holding GPU memory when the parent dies:
  `pkill -9 -f 'VLLM::EngineCore'`, then check `nvidia-smi`.

## Sizing (A100 40GB, Qwen3-0.6B, fp32, CUDA graphs)

| Step | Time |
|---|---|
| Setup (uv sync + model downloads) | 3-5 min |
| Canonicalize + checks | ~1 min |
| vLLM startup with CUDA graphs | ~2.5 min (eager: ~30 s) |
| Grid point, 32 prompts x 8 samples, 4096 tok, near the centre | ~1-2 min |
| Grid point far from the centre (outputs loop to max tokens) | up to ~9 min |

- Reward is ~0 beyond radius ~0.12 in filter-normalized units: use `--grid-range 0.12`.
  Corners past that are the slowest points and carry no information.
- Dry run (3x3, 4 prompts x 4, 2048 tok): ~7 min total, ~$0.08.
- Partial 11x11 at r=0.15 (62 points): ~2.5 h, ~$1.8.

## Outputs

- `*.csv` (one row per point and repeat; resumable), `*_config.json`, plots, `run2d.log`.
- `*_rollouts/a<alpha>_b<beta>.jsonl.gz`: completions per point (`--rollouts-per-prompt`, default 2,
  `-1` = all). Read with `gzcat FILE | jq -r '.completion' | less`, or double-click in Finder.
- `results/` is gitignored: back it up yourself.
