import random
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import wandb
from custom_interaction_generator import CustomInteractionGenerator

from main import AlphaRouterAgent

import os
import gc
import multiprocessing as mp

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
        "lr_decay": {
            "distribution": "uniform",
            "min": 0.9,
            "max": 1.0,
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-8,
            "max": 1e-2,
        },
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

        # Run-control
        "episodes": {
            "distribution": "log_uniform_values",
            "min": 200,
            "max": 2500,
        },
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

def train(connection_graph, interaction_generator=None, device="cpu"):
    with wandb.init() as run:
        run_dir = Path(run.dir)  # per-run directory  :contentReference[oaicite:5]{index=5}

        cfg = run.config

        set_seed(int(cfg.seed))

        d_model = cfg.d_model_const * cfg.num_heads

        agent = AlphaRouterAgent(connection_graph=connection_graph, interaction_generator=interaction_generator, device=device,
                                 embedding_dim=cfg.embedding_dim, num_heads=cfg.num_heads, d_model=d_model,
                                 ffn_dim=4*d_model, num_layers=cfg.num_layers, lookahead_length=cfg.lookahead_length, c_constant=cfg.c_constant,
                                 lr=cfg.lr, lr_decay=cfg.lr_decay, weight_decay=cfg.weight_decay, log_path=run_dir,
                                 wandb_run=run, snapshots=False)

        agent.learn(int(cfg.episodes), batch_size=cfg.batch_size, plot=False)

        artifact = wandb.Artifact(name="run_logs", type="logs")
        if agent.dir_path is not None:
            artifact.add_file(str(agent.dir_path))
            run.log_artifact(artifact)

        if run is not None:
            wandb.finish()

    os.system('cls')
    del agent
    gc.collect()

    # If CUDA, release cached blocks (won't always drop RSS, but helps OOM)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def train_subprocess(connection_graph, interaction_generator=None, device="cpu"):
    ctx = mp.get_context("spawn")  # important (esp. on CUDA)
    training_process = ctx.Process(target=train, args=(connection_graph, interaction_generator, device))
    training_process.start()
    training_process.join()
    if training_process.exitcode != 0:
        raise RuntimeError(f"train() subprocess failed with exit code {training_process.exitcode}")

def run_sweep(connection_graph, interaction_generator=None, runs=25, device="cpu"):
    sweep_id = wandb.sweep(sweep=sweep_config, project="alpharouter")
    wandb.agent(
        sweep_id,
        function=lambda: train(connection_graph, interaction_generator, device),
        count=runs,
    )

if __name__ == "__main__":
    connection_graph = nx.Graph()
    connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3,4), (4,5), (5,6), (6,7)])
    interaction_generator = CustomInteractionGenerator()

    run_sweep(connection_graph, interaction_generator=interaction_generator)