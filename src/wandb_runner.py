### This is an ONLINE wandb runner, which leverages the entire weights & biases functionality
### automating hyperparameter search.
### Use it only LOCALLY or on clusters that have COMPUTING NODES WITH INTERNET CONNECTION

import random
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import wandb

from router import AlphaRouterAgent

import gc
import multiprocessing as mp

from src.curriculum_scheduler import CurriculumScheduler
from utils import clear_terminal

# Config for the hyperparameter search
# See wandb documentation for more info
# (https://docs.wandb.ai/models/sweeps/define-sweep-configuration#python-script-or-notebook-2)
sweep_config = {
    "method": "bayes",
    "metric": {"name": "swaps", "goal": "minimize"},
    "parameters": {
        "c_constant": {
            "distribution": "log_uniform_values",
            "min": 1.0,
            "max": 5.0
        },
        "lr": {
            "distribution": "uniform",
            "min": 2e-4,
            "max": 5e-3,
        },
        # "lr_decay": {
        #     "distribution": "uniform",
        #     "min": 0.9,
        #     "max": 1.0,
        # },
        # "weight_decay": {
        #     "distribution": "log_uniform_values",
        #     "min": 1e-8,
        #     "max": 1e-2,
        # },
        "embedding_dim": {
            "distribution": "int_uniform",
            "min": 16,
            "max": 32,
        },

        "num_heads": {"values": [3, 4, 5, 6]},
        "num_layers": {"values": [2, 3, 4]},

        "d_model_const": {
            "distribution": "int_uniform",
            "min": 4,
            "max": 36
        },

        "lookahead_length": {"values": [32, 36, 40, 44, 48]},
        "batch_size": {"values": [32, 64]},
        "curriculum_proceed_threshold": {"distribution": "uniform", "min": 1.25, "max": 2.0},
        "curriculum_retreat_threshold": {"distribution": "uniform", "min": 2.5, "max": 5.0},

        # Run-control
        # "episodes": {
        #     "distribution": "log_uniform_values",
        #     "min": 200,
        #     "max": 1500,
        # },
        "seed": {"values": [0, 1, 2, 42, 69, 2137]},
    },

    # Optional: early termination using Hyperband
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 150,
        "eta": 3,
    },
}



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(connection_graph, interaction_generator=None, device="cpu") -> None:
    """
    Trains a single agent instance.
    Synchronizes the results with Weighs & Biases (wandb)

    Parameters
    ----------
    connection_graph - qubit topology
    interaction_generator - circuit generator (if None, then a BasicInteractionGenerator is used)
    device - cpu or cuda, etc.
    """
    with wandb.init() as run:
        run_dir = Path(run.dir)  # per-run directory  :contentReference[oaicite:5]{index=5}

        cfg = run.config

        set_seed(int(cfg.seed))

        d_model = cfg.d_model_const * cfg.num_heads

        curriculum = CurriculumScheduler(proceed_threshold=cfg.curriculum_proceed_threshold,
                                         retreat_threshold=cfg.curriculum_retreat_threshold)

        agent = AlphaRouterAgent(connection_graph=connection_graph, interaction_generator=interaction_generator, device=device,
                                 embedding_dim=cfg.embedding_dim, num_heads=cfg.num_heads, d_model=d_model,
                                 ffn_dim=4*d_model, num_layers=cfg.num_layers, lookahead_length=cfg.lookahead_length, c_constant=cfg.c_constant,
                                    lr=cfg.lr, log_path=run_dir, episode_limit=120, wandb_run=run)

        # todo: Make timesteps a parameter
        agent.learn(32, batch_size=cfg.batch_size, plot_graphs=False, save_model=True, curriculum_scheduler=curriculum)

        artifact = wandb.Artifact(name="run_logs", type="logs")
        if agent.dir_path is not None:
            artifact.add_file(str(agent.dir_path))
            run.log_artifact(artifact)

        if run is not None:
            wandb.finish()

    clear_terminal()
    del agent
    gc.collect()

    # If CUDA, release cached blocks (won't always drop RSS, but helps OOM)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# def train_subprocess(connection_graph, interaction_generator=None, device="cpu"):
#     """
#     Trains a single agent instance as a child process.
#     Synchronizes the results with Weighs & Biases (wandb).
#
#     Raises a runtime error if the subproccess exited unexpectedly.
#
#     Parameters
#     ----------
#     connection_graph - qubit topology
#     interaction_generator - circuit generator (if None, then a BasicInteractionGenerator is used)
#     device - cpu or cuda, etc.
#     """
#     ctx = mp.get_context("spawn")  # important (esp. on CUDA)
#     training_process = ctx.Process(target=train, args=(connection_graph, interaction_generator, device))
#     training_process.start()
#     training_process.join()
#     if training_process.exitcode != 0:
#         raise RuntimeError(f"train() subprocess failed with exit code {training_process.exitcode}")

def _run_agent(sweep_id, connection_graph, interaction_generator, device, runs) -> None:
    """
    Trains a single agent instance.
    Synchronizes the results with Weighs & Biases (wandb)

    Parameters
    ----------
    sweep_id - wandb id of the sweep
    connection_graph - qubit topology
    interaction_generator - circuit generator (if None, then a BasicInteractionGenerator is used)
    device - cpu or cuda, etc.
    runs - the number of config sweeps (trials) to run
    """
    wandb.agent(
        sweep_id,
        function=lambda: train(connection_graph, interaction_generator, device),
        count=runs,
    )

def run_sweep(connection_graph, interaction_generator=None, runs=25, processes=1, device="cpu") -> None:
    """
    Performs the (possibly) parallelized hyperparameter search.

    Parameters
    ----------
    connection_graph - qubit topology
    interaction_generator - circuit generator (if None, then a BasicInteractionGenerator is used)
    runs - the number of config sweeps (trials) to run
    processes - number of concurrent processes to use at any given moment.
    device - cpu or cuda, etc.
    """
    sweep_id = wandb.sweep(sweep=sweep_config, project="alpharouter")

    runs_per_agent = runs // processes

    agent_processes = [mp.Process(target=_run_agent,
                                  args=(sweep_id, connection_graph, interaction_generator, device, runs_per_agent))
             for _ in range(processes)]
    for agent in agent_processes: agent.start()
    for agent in agent_processes: agent.join()


if __name__ == "__main__":
    connection_graph = nx.Graph()
    # connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9)])
    connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
    # interaction_generator = CustomInteractionGenerator()
    interaction_generator = None

    run_sweep(connection_graph, interaction_generator=interaction_generator, runs=4, processes=4)