# AlphaRouter

**Quantum circuit routing with reinforcement learning and Monte Carlo Tree Search.**

Implementation of the paper by Tang et al.

> **Reference paper:** [*AlphaRouter: Quantum Circuit Routing with Reinforcement Learning and Tree Search*](https://arxiv.org/abs/2410.05115).
> 
> **Project report:** [*Open-Source Reimplementation of AlphaRouter for Quantum Circuit Routing*](Open-Source%20Reimplementation%20of%20AlphaRouter%20for%20Quantum%20Circuit%20Routing.pdf)
> 
> **Authorship notice**: I am not an author of the original AlphaRouter paper. This repository is an independent reimplementation attempt based on the method described by Tang et al.; it should not be presented as the authors’ official implementation. The motivation, implementation choices, deviations, and experimental results of this reimplementation are discussed in the accompanying report.

## Overview

Quantum hardware generally has sparse qubit connectivity, so a logical two-qubit gate may require intermediate **SWAP** operations before its operands become adjacent. AlphaRouter treats SWAP insertion as sequential decision-making and learns a routing policy using a transformer-based agent supervised by Monte Carlo Tree Search (MCTS).

The paper formulates a routing state from the **remaining logical circuit** and **current logical-to-physical qubit mapping**; an action selects a physical edge on which to apply a SWAP. Gates that become topology-compliant are then scheduled. The objective is to complete routing with as few inserted SWAPs as possible.

### Training path

```text
Routing state
    │
    ├─ gate/qubit embedding ──> Transformer ──> SWAP policy logits
    │                                      └─> state-value estimate
    │
    └─ replay buffer ──> MCTS ──> action/value targets
                               │
                               └─> policy + value loss ──> Adam update
```

The implementation includes:

- a trainable physical-qubit embedding and dependency-depth feature encoding for upcoming gates;
- a transformer encoder with a policy decoder and scalar value head;
- a replay buffer and MCTS-generated training targets;
- cross-entropy-style policy loss plus squared value loss
- gradient clipping and optional curriculum learning based on circuit length;
- Qiskit `SABRE` and `BasicSwap` baselines for SWAP-count comparison;
- transformer-only routing at inference time.

### Important repo/paper distinction

This repo is **not a literal copy of the hyperparameters reported in the paper**.

## Installation

The project was created with Python 3.12 in mind. Dependencies are pinned in `requirements.txt` and can be installed with:

```bash         # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Running

The repository contains three ways to launch work, depending on whether you are doing a local/debug run, a Weights & Biases hyperparameter search, or a DelftBlue HPC run.
### 1. Direct/local runs

For development, debugging, or a single manually configured experiment, the low-level entry point is:
python src/router.py
src/router.py contains the experiment configuration in its __main__ block, including the routing topology, model and training hyperparameters, curriculum settings, device selection, checkpointing, and an inference example. Review those values before running it; the checked-in configuration should not be interpreted as a universal reproduction configuration for every experiment reported in the accompanying report.
### 2. Weights & Biases runs

The repository also contains dedicated Weights & Biases (W&B) runner scripts for hyperparameter search and experiment tracking. W&B is used to organize runs, record training metrics, compare hyperparameter configurations, and visualize the resulting experiments.
Two workflows are provided, matching those used in the report:
- Online W&B sweep: runs a Bayesian hyperparameter sweep managed through the W&B API. Use this when the machine has Internet access and can communicate directly with W&B during training.
- Offline W&B sweep: samples hyperparameter configurations locally and records the runs as W&B offline artifacts. This is intended for environments where compute nodes do not have Internet access. The saved runs can be synchronized with W&B after the jobs finish.
The W&B runner scripts contain the corresponding sweep/search configuration, so inspect them before launching a search rather than assuming the checked-in parameter ranges are appropriate for a new experiment. For online runs, authenticate with W&B first if necessary, for example:
wandb login
Offline runs can be synchronized later from a machine with network access using W&B's normal synchronization workflow.
### 3. DelftBlue HPC runs

The scripts/ directory contains the DelftBlue-specific setup and run scripts used to execute the hyperparameter-search workflow on the TU Delft DelftBlue supercomputer. These scripts are separate from the W&B experiment definitions: they handle the HPC-side environment/setup and job execution needed to launch the training runs on DelftBlue.
The report's multi-gate experiment used this offline workflow because DelftBlue compute nodes do not have Internet access. In that setup, training runs are executed on DelftBlue, W&B data is stored locally/offline during the jobs, and the resulting run data can be synchronized with W&B afterwards for analysis and visualization.
Before submitting jobs, inspect the files in scripts/ and adapt the DelftBlue-specific settings to your account and allocation. They should be treated as reproducibility/support scripts for the reported HPC setup, not as portable defaults for arbitrary clusters.

Please note that the Delft HPC scripts have been vibe coded and therefore should be verified before running.

### Outputs

Depending on the selected workflow, training and evaluation may produce checkpoints, logs, plots, and W&B run data. Repository-local outputs are written under directories such as:

```text
models/   model checkpoints
logs/     experiment logs
plots/    generated plots
```

W&B-managed runs additionally store their experiment metadata and metrics through the online or offline W&B workflow described above.

## Core repo layout

```text
requirements.txt
src/
├── router.py                        # agent, replay buffer, training, inference
├── transformer.py                   # positional encoding, transformer, policy/value heads
├── mcts.py                          # MCTS and experimental parallel-MCTS implementation
├── curriculum_scheduler.py          # circuit-length curriculum and performance metric
├── benchmark.py                     # Qiskit SABRE / BasicSwap comparison
├── manhattan_rewarder.py            # distance-based reward utility
├── custom_interaction_generator.py  # custom QGym interaction generator
├── plotting.py                      # training/routing/MCTS visualization helpers
└── utils.py                         # state encoding and routing utilities
```

The active training path in `router.py` instantiates the single-process `MCTS`. `mcts.py` also contains a parallel implementation, but it was not used for the final report because inter-process model-transfer overhead removed the expected speedup.

## Relation to the paper

The attached/reference paper describes the original AlphaRouter method: MCTS supplies action and state-value targets to a transformer agent during training, while inference uses the trained agent without MCTS. It reports **10–20% fewer SWAPs than the best tested heuristic baseline (SABRE) on most evaluated benchmarks**, including unseen benchmark classes, and approximately a **15% lower SWAP-count scaling coefficient** in the reported scaling experiment.

Those numbers are **paper results, not results independently established for this repo**. Do not attribute them to this repo without reproducing the corresponding experiments.

### Results

I have **not** managed to reproduce the results of the paper. This may be due to some mistakes on my part, of which I am not aware, or because of some implementation details, such as hyperparameters, or because of some unimplemented features (the one I am aware of being learning rate warmup & decay).

The exact results I obtained can be found in the report.

## Curriculum learning

The repo includes an optional curriculum scheduler. In the configuration studied in the report, curriculum difficulty is the maximum generated circuit length. Progress is evaluated relative to the mean SWAP count of the SABRE baseline; the scheduler can increase or decrease the curriculum level after a cooldown period.

The implementation is structured so that other curriculum variables can be added later, such as topology size, initial-mapping difficulty, or gate-distance distribution.

## TODO

There are some TODOs left around in the codebase, the most important one being implementing learning rate warm-up & decay as is usual for the transformer architecture.

### Repo maintenance

The author & owner of this repository will NOT maintain it further. If you wish to improve it, you need to do it yourself. Feel free to ask any questions, though.

## Acknowledgements

My sincere thanks to Dr. Sebastian Feld of TU Delft for his supervision, guidance, and support throughout this project!

## License

**GNU General Public License v3.0 (GPL-3.0).**

The GNU General Public License is a free, copyleft license for
software and other kinds of works.

See: [*LICENSE*](LICENSE)
