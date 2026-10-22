"""Optional Weights & Biases logging.

Logging is skipped (never fatal) when wandb is not installed, WANDB_MODE=disabled,
or no API key is configured. Project defaults to $WANDB_PROJECT or "rlvr-landscape".
"""

import os
from pathlib import Path


def init_run(args, job_type, name=None):
    if os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; skipping logging.")
        return None

    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "rlvr-landscape"),
            entity=os.environ.get("WANDB_ENTITY"),
            job_type=job_type,
            name=name,
            config=vars(args),
        )
    except Exception as error:  # no key, no network, ...
        print(f"[wandb] disabled: {error}")
        return None

    print(f"[wandb] {run.url}")
    return run


def log_point(run, row, direction):
    """Log one alpha point; charts for each direction use alpha as the x-axis."""
    if run is None:
        return
    prefix = f"dir{direction}"
    run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/alpha")
    run.log({
        f"{prefix}/{key}": value
        for key, value in row.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    })


def finish_run(run, files, results_df=None, summary=None):
    """Upload the results table, plots and this run's output files, then close the run."""
    if run is None:
        return
    import wandb

    files = sorted(Path(f) for f in files if Path(f).is_file())  # globs also match rollout dirs
    if results_df is not None:
        run.log({"results": wandb.Table(dataframe=results_df)})
    if summary:
        run.summary.update(summary)
    for path in files:
        if path.suffix == ".png":
            run.log({f"plots/{path.stem}": wandb.Image(str(path))})

    artifact = wandb.Artifact(name=f"{run.job_type}-{run.id}", type="results")
    for path in files:
        artifact.add_file(str(path))
    run.log_artifact(artifact)
    run.finish()
