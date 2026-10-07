### This is an OFFLINE wandb runner, which performs hyperparameter search by its own implementation,
### and then syncs the results with wandb for visualisation and logging.
### Use it whenever working on clusters that have COMPUTING NODES WITHOUT INTERNET CONNECTION.
### This is probably not an efficient way to use wandb, as it doesn't leverage the hyperparameter search functionality,
### there might be a library more suited for this use case.

### This runner does NOT include curriculum learning.

#todo: make it reuse functions from the wandb_runner_offline.py

import argparse
import gc
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
import wandb

from router import AlphaRouterAgent

SEARCH_SPACE = {
    "c_constant": {"distribution": "log_uniform_values", "min": 1.0, "max": 5.0},
    "lr": {"distribution": "uniform", "min": 2e-4, "max": 5e-3},
    "embedding_dim": {"distribution": "int_uniform", "min": 16, "max": 32},
    "num_heads": {"values": [3, 4, 5, 6]},
    "num_layers": {"values": [2, 3, 4]},
    "d_model_const": {"distribution": "int_uniform", "min": 4, "max": 36},
    "lookahead_length": {"values": [32, 36, 40, 44, 48]},
    # "batch_size": {"values": [32, 64]},
    "seed": {"values": [0, 1, 2, 42, 69, 2137]},
    # "curriculum_proceed_threshold": {"distribution": "uniform", "min": 1.5, "max": 2.5},
    # "curriculum_retreat_threshold": {"distribution": "uniform", "min": 4.0, "max": 6.0},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def set_single_thread_runtime() -> None:
    """
    Ensures there is only 1 thread per process.
    Probably not necessary.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Can fail if PyTorch has already initialized the thread pool.
            pass



def sample_config(rng: random.Random) -> dict[str, Any]:
    """
    Sample one hyperparameter configuration from the hyperparameter space.
    Parameters
    ----------
    rng - seed generator

    Returns
    -------
    A dictionary of hyperparameter values.
    """
    cfg: dict[str, Any] = {}

    for key, spec in SEARCH_SPACE.items():
        if "values" in spec:
            cfg[key] = rng.choice(spec["values"])
        elif spec["distribution"] == "uniform":
            cfg[key] = rng.uniform(spec["min"], spec["max"])
        elif spec["distribution"] == "log_uniform_values":
            cfg[key] = math.exp(rng.uniform(math.log(spec["min"]), math.log(spec["max"])))
        elif spec["distribution"] == "int_uniform":
            cfg[key] = rng.randint(spec["min"], spec["max"])
        else:
            raise ValueError(f"Unsupported search distribution for {key}: {spec}")

    return cfg



def build_default_connection_graph() -> nx.Graph:
    """
    Build a small connection graph with 4 qubits and line topology.

    Returns
    -------
    A connection graph
    """
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
    return graph



def add_logs_as_artifact(run: wandb.sdk.wandb_run.Run, path_like: str | os.PathLike[str] | None) -> None:
    """
    Store a file or directory of logs as a Weights & Biases Artifact.

    A W&B Artifact is a versioned collection of files associated with a run,
    commonly used to track outputs such as models, datasets, logs, or other
    reproducibility-related files.

    This function creates an Artifact named ``"run-logs-{run.id}"`` with type
    ``"logs"`` and logs it as an output of ``run``.

    If ``path_like`` points to a directory, all files in the directory are
    added with ``Artifact.add_dir``. If it points to a single file, it is
    added with ``Artifact.add_file``.

    If ``path_like`` is ``None`` or does not exist, the function returns
    without creating an Artifact.
    """
    if path_like is None:
        return

    path = Path(path_like)
    if not path.exists():
        return

    artifact = wandb.Artifact(name=f"run-logs-{run.id}", type="logs")
    if path.is_dir():
        artifact.add_dir(str(path))
    else:
        artifact.add_file(str(path))
    run.log_artifact(artifact)



def train_single_config(config: dict[str, Any], device: str = "cpu") -> None:
    """
    Trains a single agent instance from the given hyperparameter configuration.
    Adds the log as a Weighs & Biases (wandb) artifact.
    Uses a simple line topology with 4 qubits. #todo: make it accept arbitrary connection graphs

    Parameters
    ----------
    config - a dictionary of hyperparameter values
    device - cpu or cuda, etc.
    """
    set_single_thread_runtime()
    set_seed(int(config["seed"]))

    connection_graph = build_default_connection_graph()
    interaction_generator = None

    d_model = int(config["d_model_const"]) * int(config["num_heads"])

    with wandb.init(
        project=os.environ.get("WANDB_PROJECT", "alpharouter"),
        config=config,
        mode=os.environ.get("WANDB_MODE", "online"),
    ) as run:
        run_dir = Path(run.dir)

        agent = AlphaRouterAgent(
            connection_graph=connection_graph,
            interaction_generator=interaction_generator,
            device=device,
            embedding_dim=int(config["embedding_dim"]),
            num_heads=int(config["num_heads"]),
            d_model=d_model,
            ffn_dim=4 * d_model,
            num_layers=int(config["num_layers"]),
            lookahead_length=int(config["lookahead_length"]),
            c_constant=float(config["c_constant"]),
            lr=float(config["lr"]),
            log_path=run_dir,
            episode_limit=120,
            wandb_run=run,
            snapshots=False,
        )

        try:
            # curriculum = CurriculumScheduler(proceed_threshold=int(config["curriculum_proceed_threshold"]),
            #                                  retreat_threshold=int(config["curriculum_retreat_threshold"]),
            #                                  cooldown=20, max_level=10)
            agent.learn(1200, batch_size=64, plot_graphs=True, show_circuits=False,
                        save_model=True, curriculum_scheduler=None)
            add_logs_as_artifact(run, getattr(agent, "dir_path", None))
        finally:
            del agent
            gc.collect()
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()



def _worker_entry(config_json: str, device: str) -> None:
    """
    Retrieves a hyperparameter configuration from json and trains a single agent.

    Parameters
    ----------
    config_json - json with the hyperparameter configuration
    device - cpu or cuda, etc.
    """
    config = json.loads(config_json)
    train_single_config(config=config, device=device)



def run_search(runs: int, workers: int, device: str, search_seed: int) -> None:
    """
    Performs the (possibly) parallelized hyperparameter search.

    Parameters
    ----------

    runs - the number of config sweeps (trials) to run
    workers - number of concurrent processes to use at any given moment.
    device - cpu or cuda, etc.
    """
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    rng = random.Random(search_seed)
    configs = [sample_config(rng) for _ in range(runs)]

    wandb_root = Path(os.environ.get("WANDB_DIR", ".")).resolve()
    wandb_root.mkdir(parents=True, exist_ok=True)
    plan_path = wandb_root / f"planned_configs_{int(time.time())}.json"
    plan_path.write_text(json.dumps(configs, indent=2), encoding="utf-8")

    ctx = torch.multiprocessing.get_context("spawn")
    pending = [json.dumps(cfg) for cfg in configs]
    active: list[tuple[torch.multiprocessing.Process, str]] = []
    failures: list[int] = []

    while pending or active:
        while pending and len(active) < workers:
            cfg_json = pending.pop(0)
            proc = ctx.Process(target=_worker_entry, args=(cfg_json, device))
            proc.start()
            active.append((proc, cfg_json))

        next_active: list[tuple[torch.multiprocessing.Process, str]] = []
        for proc, cfg_json in active:
            proc.join(timeout=0.2)
            if proc.is_alive():
                next_active.append((proc, cfg_json))
                continue
            if proc.exitcode != 0:
                print(f"FAILED CONFIG: {cfg_json}", flush=True)
                failures.append(proc.exitcode or 1)
        active = next_active
        time.sleep(0.2)

    if failures:
        raise RuntimeError(f"{len(failures)} training runs failed during the search.")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline hyperparameter search for DelftBlue + W&B logging.")
    parser.add_argument("--runs", type=int, default=6, help="Total number of sampled configurations to evaluate.")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Maximum number of concurrent training processes.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string, e.g. cpu or cuda:0.")
    parser.add_argument("--search-seed", type=int, default=2137, help="Seed used to sample the search configurations.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_search(runs=args.runs, workers=args.workers, device=args.device, search_seed=args.search_seed)
