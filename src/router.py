import pickle
import sys
import time
from collections import deque
from copy import deepcopy
from os import mkdir
from pathlib import Path
from typing import Optional, List

import networkx as nx
import numpy as np
import torch
from numpy import mean
from qgym.envs import Routing
from qgym.envs.routing import RoutingState
import matplotlib.pyplot as plt
import random

from qgym.generators.interaction import InteractionGenerator
from torch import nn, Tensor
from torch.types import Device
from tqdm import tqdm

from benchmark import benchmark
from custom_interaction_generator import CustomInteractionGenerator
from mcts import ParallelMCTS, MCTS
from plotting import moving_average, plot, plot_mapping_and_circuit, render_rgb, plot_subplots_2scales

from gymnasium.vector import SyncVectorEnv

from curriculum_scheduler import CurriculumScheduler, compute_performance, GateLengthCurriculumGenerator
from transformer import Transformer, choose_actions
from utils import NormalizeInfoWrapper, embed_nodes, boost_process_priority, \
    open_binary, get_timestamp, env_surpass_all_possible_gates, env_surpass_all_possible_gates_with_info

EMBEDDING_DIM = 20      # AR paper embedding dimension - 20
LOOKAHEAD_LENGTH = 48   # AR paper lookahead length - 48
NUM_HEADS = 4           # AR paper number of heads in multihead attention - 6
NUM_LAYERS = 3          # AR paper number of transformer layers - 4
D_MODEL = 32            # AR paper uses 216
FFN_DIM = 4 * D_MODEL   # Default hidden layer dimensionality of ffn in transformer
N_ENVS: int = 32
BATCH_SIZE: int = 32
MCTS_PROCESSES: int = 1

REPLAY_BUFFER_MIN_SIZE_MCTS: int = 160 # AR paper uses 320

# LEARNING_RATE = 2e-5
# LEARNING_RATE = 1e-2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0
LR_DECAY_STEP = 40
LR_DECAY = 1
MAX_GRAD_NORM = 1
BUFFER_SIZE = int(1e5)
SNAPSHOT_PERIOD = 64

MODEL_DIR = Path(__file__).resolve().parent.joinpath('../models/')
TEMP_MODEL_DIR = MODEL_DIR.joinpath('temp/')
TEMP_BUFFER_DIR = TEMP_MODEL_DIR.joinpath('buffers/')
TEMP_TRANSFORMER_DIR = TEMP_MODEL_DIR.joinpath('temp_transformer.pt')
TEMP_EMBEDDING_DIR = TEMP_MODEL_DIR.joinpath('temp_embedding.pt')
LOG_DIR = Path(__file__).resolve().parent.joinpath('../logs/')
PLOT_DIR = Path(__file__).resolve().parent.joinpath('../plots/')


def make_env(connection_graph, interaction_generator: InteractionGenerator = None, lookahead_length: int = LOOKAHEAD_LENGTH
             , rewarder=None):
    """
    Environment setup function for the agent with given connection graph, interaction generator, lookahead length, and rewarder.
    Returns a function that yields an environment upon being called.
    """
    def _thunk():
        env = Routing(
            max_observation_reach=lookahead_length,
            observe_legal_surpasses=True,
            observe_connection_graph=True,
            connection_graph=connection_graph,
            render_mode="rgb_array",
            rewarder=rewarder,
            interaction_generator=interaction_generator
        )

        env = NormalizeInfoWrapper(env)
        return env
    return _thunk


class ReplayBuffer:
    """
    A simple replay buffer:
      - stores previously encountered states
      - samples batches of states
    """
    def __init__(self, capacity: int = BUFFER_SIZE):
        """
        :param capacity: maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer   = deque(maxlen=capacity)

    def store(self, state: RoutingState):
        """
        Stores a state.
        """
        self.buffer.append(deepcopy(state))

    def clear(self):
        """
        Clears the buffer.
        """
        self.buffer.clear()

    def sample(self, batch_size: int) -> List[RoutingState]:
        """
        Samples a batch of previously stored states
        """
        batch = random.sample(self.buffer, batch_size)

        return batch

    def save_to_disk(self, path: str | Path, compress: Optional[str] = None) -> None:
        """
        Saves the buffer to disk.
        Parameters
        ----------
        path - filepath to the file to save
        compress - if present, compresses the saved file using specified compression algorithm
        """
        path = path if isinstance(path, Path) else Path(path)
        with open_binary(path, 'wb', compress=compress) as f:
            pickle.dump((self.buffer.maxlen, list(self.buffer)), f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Buffer snapshot saved at {path}")

    def load_from_disk(self, path: str | Path, compress: Optional[str] = None) -> None:
        """
        Loads the buffer from disk.
        Parameters
        ----------
        path - filepath to the file to load
        compress - if present, decompresses the saved file using specified decompression algorithm
        """
        path = path if isinstance(path, Path) else Path(path)
        with open_binary(path, 'rb', compress) as f:
            maxlen, data = pickle.load(f)
        self.buffer = deque(data, maxlen=maxlen)

        print(f"Buffer snapshot loaded from {path}")

    def __len__(self):
        return len(self.buffer)



class AlphaRouterAgent(nn.Module):
    """The AlphaRouter routing agent"""
    def __init__(self, connection_graph: nx.Graph, interaction_generator: InteractionGenerator | None = None,
                 device: Device = torch.device("cpu"), embedding_dim: int = EMBEDDING_DIM, num_heads: int = NUM_HEADS, d_model: int = D_MODEL, ffn_dim: int = FFN_DIM,
                 num_layers: int = NUM_LAYERS, lookahead_length: int = LOOKAHEAD_LENGTH, buffer_size: int = BUFFER_SIZE, episode_limit: int = 100, c_constant: float = 2,
                 lr: float = LEARNING_RATE, lr_decay: float = LR_DECAY, lr_decay_step: int = LR_DECAY_STEP, weight_decay: float = WEIGHT_DECAY,
                 test_seed: int = 42, log_path: Path | None = LOG_DIR, wandb_run = None, *args, **kwargs):
        """
        :param connection_graph: topology of the device
        :param interaction_generator: generator of the quantum circuits. If none, a BasicInteractionGenerator is used.
        :param device: Device to run the agent on (cuda for Nvidia graphics card, cpu for a standard cpu)
        :param embedding_dim: Embedding dimension of the node embeddings (not including the depth embedding)
        :param num_heads: Number of heads in a multihead attention layer
        :param d_model: Dimension of the hidden layer in the transformer
        :param ffn_dim: Dimensionality of the hidden layer in the feed forward layer
        :param num_layers: Number of transformer layers
        :param lookahead_length: The maximum number of the gates ahead that we will analyze
        :param buffer_size: Capacity of the replay buffer
        :param episode_limit: Maximum number of episodes while testing the agent.
        :param c_constant: Exploration/exploitation constant for the MCTS
        :param lr: Learning rate
        :param lr_decay: Learning rate decay
        :param lr_decay_step: Period of learning rate decay (how long we wait until we decrease the learning rate)
        :param weight_decay: Weight decay parameter for the Adam optimizer. Turned off by default.
        :param test_seed: Seed for random number generator for tests.
        :param log_path: Path to the logging directory.
        :param wandb_run: Optional Wandb run object for metric logging / hyperparameter search (if run online)
        """
        self.dir_path = None
        print('Init')
        super().__init__(*args, **kwargs)
        self.device = device
        self.connection_graph = connection_graph
        self.env = make_env(connection_graph, interaction_generator, lookahead_length)
        self.n_qubits = connection_graph.number_of_nodes()
        self.n_edges = connection_graph.number_of_edges()
        self.episode_limit = episode_limit  # The maximal number of steps in a single episode
        self.loss_sum = 0
        self.lookahead_length = lookahead_length
        self.embedding_dim = embedding_dim
        self.qubit_embedding = nn.Embedding(num_embeddings=connection_graph.number_of_nodes(),
                                            embedding_dim=embedding_dim)
        self.qubit_embedding.weight.share_memory_()
        print(self.qubit_embedding.weight)
        self.action_space_size = len(connection_graph.edges)
        self.c_constant = c_constant
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.log_dir = log_path
        self.wandb_run = wandb_run

        embedding_dim = embedding_dim + 1  # Embedding dim + 1 for depth

        # if (embedding_dim) % num_heads != 0:
        #     # Increase the dimension so that it is a multiple of the number of heads
        #     padded_embedding_dim = (embedding_dim // num_heads + 1) * num_heads
        #     self.padding_dim = padded_embedding_dim - embedding_dim
        #     embedding_dim = padded_embedding_dim
        # else:
        #     self.padding_dim = 0
        self.padding_dim = 0

        assert d_model % num_heads == 0

        self.transformer = Transformer(embedding_dim, num_heads, d_model, ffn_dim, num_layers, lookahead_length,
                                       swap_options=self.action_space_size, device=self.device).to(self.device)
        self.replay_buffer = ReplayBuffer(
            buffer_size
        )
        self.mcts: ParallelMCTS | MCTS | None = None

        self.to(self.device)

        self.optimizer = torch.optim.Adam(list(self.transformer.parameters()) + list(self.qubit_embedding.parameters()),
                                          lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_decay_step,
                                                         gamma=lr_decay)
        # todo: check how warmup impacts performance (transformers usually have variable learning rate, see the Attention is All You Need paper)
        # self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=500,num_training_steps=0)

        self.test_seed = test_seed

        self.unfinished_indices = set(range(N_ENVS))

        self._make_envs(connection_graph, interaction_generator, lookahead_length)
        self.curriculum = None

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        TEMP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        TEMP_BUFFER_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        print('Init successful')

    def _make_envs(self, connection_graph, interaction_generator: InteractionGenerator = None,
                   lookahead_length: int = LOOKAHEAD_LENGTH):
        """
        Prepare environments of the agent for the training and testing
        Parameters
        ----------
        connection_graph - topology of the device
        interaction_generator - generator of quantum circuits
        lookahead_length - maximum number of the gates ahead that we will analyze
        """
        env_fns = [make_env(connection_graph, interaction_generator, lookahead_length) for _ in range(N_ENVS)]
        self.vec_env = SyncVectorEnv(env_fns)
        self.vec_env.reset()
        self.test_vec_env = SyncVectorEnv(env_fns)

        self.unfinished_indices = set(range(N_ENVS))

        self.test_vec_env.reset(seed=self.test_seed)
        self.test_states_backup = [deepcopy(env._state) for env in self.test_vec_env.envs]

    def _reset_test_envs(self):
        """
        Reset test environments
        """
        for i in range(len(self.test_vec_env.envs)):
            self.test_vec_env.envs[i]._state = deepcopy(self.test_states_backup[i])

    def batch(self, batch_size: int = BATCH_SIZE) -> tuple[int, int, int]:
        """
        Process one batch of environments, one episode each.
        One episode means trying to solve one particular circuit.

        The loss is averaged over all environments to reduce variance and computation.

        If you want more info, uncomment print statements. #todo: make debug info printable if debug flag is set to truew

        Returns
        -------
        A tuple of 3 elements:
        - Mean loss
        - Mean L1 loss
        - Mean L2 loss
        (see the AlphaRouter paper for the formulae: https://arxiv.org/abs/2410.05115)
        """
        self.transformer.train()
        self.optimizer.zero_grad()

        # print(f'### Unfinished indices: {self.unfinished_indices}')

        state_embeddings = torch.stack(
            [embed_nodes(cur_env._state, self.lookahead_length, self.device, self.qubit_embedding,
                              self.embedding_dim, self.padding_dim) for cur_env in self.vec_env.envs],
            dim=0
        ).to(self.device)

        logits, _ = self.transformer(state_embeddings)
        actions, _ = choose_actions(logits)

        if isinstance(actions, Tensor):
            actions = actions.cpu().numpy()

        observations, rewards, terminated, truncated, info = self.vec_env.step(actions)
        done = np.logical_or(terminated, truncated)

        for i in deepcopy(self.unfinished_indices):
            if done[i]:
                self.unfinished_indices.remove(i)
                continue

            env = self.vec_env.envs[i]
            obs, rew, done[i], info = env_surpass_all_possible_gates_with_info(env, self.action_space_size)

            if done[i]:
                self.unfinished_indices.remove(i)

        for i in deepcopy(self.unfinished_indices):
            self.replay_buffer.store(self.vec_env.envs[i]._state)

        loss = []
        l1 = []
        l2 = []
        mean_loss = 0

        # print(f'### Replay buffer size: {len(self.replay_buffer.buffer)}')

        if len(self.replay_buffer.buffer) > REPLAY_BUFFER_MIN_SIZE_MCTS or (
                len(self.unfinished_indices) == 0 and len(self.replay_buffer.buffer) > batch_size):
            mcts_input = self.replay_buffer.sample(batch_size)
            if MCTS_PROCESSES > 1:
                torch.save(self.qubit_embedding.state_dict(), TEMP_EMBEDDING_DIR)
                self.transformer.cpu()
            mcts_actions, mcts_state_evals = self.mcts.run(mcts_input)

            self.transformer.train()
            if MCTS_PROCESSES > 1:
                self.transformer.to(self.device)

            embeddings = torch.stack(
                [embed_nodes(state, self.lookahead_length, self.device, self.qubit_embedding,
                             self.embedding_dim, self.padding_dim) for state in mcts_input],
                dim=0
            ).to(self.device)

            transformer_logits, state_values = self.transformer(embeddings)
            transformer_actions, transformer_all_log_probs = choose_actions(transformer_logits, sample=False)

            if isinstance(mcts_actions, list):
                l1 = torch.stack([-batch_log_probs[idx] for batch_log_probs, idx in zip(transformer_all_log_probs, mcts_actions)])
            else:
                l1 = -transformer_all_log_probs[0]
            print(f'### l1: {l1}')

            mcts_state_evals = torch.tensor(mcts_state_evals, dtype=transformer_all_log_probs.dtype, device=transformer_all_log_probs.device)
            l2 = (state_values - mcts_state_evals).pow(2)
            print(f'### l2: {l2}')

            self.loss_coefficient = np.sqrt((l1.cpu().detach() ** 2).sum() / (l2.cpu().detach() ** 2).sum())
            # print(f'### loss_coefficient: {self.loss_coefficient}')
            # assert l1.requires_grad
            # assert l2.requires_grad

            loss = (l1 + self.loss_coefficient * l2) / self.action_space_size
            print(f'### loss: {loss}')
            mean_loss = loss.mean()
            print(f'### mean_loss: {mean_loss}')

            try:
                l1 = l1.mean()
                l2 = l2.mean()
            except:
                print('Could not compute the mean of l1 or l2')

            ### UNCOMMENT THE CODE BELOW FOR MORE DEBUG INFO
            # if isinstance(mcts_actions, list):
            #     for state, transformer_action, mcts_action in zip(mcts_input, transformer_actions, mcts_actions):
            #         print(f'### state: {state}')
            #         print(f'### transformer_action: {transformer_action}')
            #         print(f'### mcts_action: {mcts_action}')
            # else:
            #     print(f'### state: {mcts_input[0]}')
            #     print(f'### transformer_action: {transformer_actions[0]}')
            #     print(f'### mcts_action: {mcts_actions}')

            ### BACKPROPAGATION STEP & GRADIENT CLIPPING

            # print(f'Backpropagation step...')
            mean_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()

        if isinstance(mean_loss, torch.Tensor):
            mean_loss = mean_loss.detach().cpu()
        if isinstance(l1, torch.Tensor):
            l1 = l1.detach().cpu()
        elif isinstance(l1, list):
            l1 = None
        if isinstance(l2, torch.Tensor):
            l2 = l2.detach().cpu()
        elif isinstance(l2, list):
            l2 = None

        return mean_loss, l1, l2

    def test(self) -> tuple[float, list[int]]:
        """
        Test the current performance of the agent

        Returns
        -------
        a tuple that contains:
        - mean swaps for all test envs
        - a list of swaps for all test envs
        """
        self._reset_test_envs()

        state_embeddings = torch.stack(
            [embed_nodes(cur_env._state, self.lookahead_length, self.device, self.qubit_embedding,
                         self.embedding_dim, self.padding_dim) for cur_env in self.test_vec_env.envs],
            dim=0
        ).to(self.device)

        unfinished_test_indices = set(range(N_ENVS))

        swaps = [self.episode_limit] * N_ENVS

        for step_idx in range(self.episode_limit):
            logits, _ = self.transformer(state_embeddings)
            actions, _ = choose_actions(logits)

            if isinstance(actions, Tensor):
                actions = actions.cpu().numpy()

            observations, rewards, terminated, truncated, info = self.test_vec_env.step(actions)
            done = np.logical_or(terminated, truncated)

            for i in deepcopy(unfinished_test_indices):
                if done[i]:
                    unfinished_test_indices.remove(i)
                    swaps[i] = step_idx + 1
                    continue

                env = self.test_vec_env.envs[i]
                obs, rew, done[i], info = env_surpass_all_possible_gates_with_info(env, self.action_space_size)

                if done[i]:
                    unfinished_test_indices.remove(i)
                    swaps[i] = step_idx + 1

                if not unfinished_test_indices:
                    break

                state_embeddings = torch.stack(
                    [embed_nodes(cur_env._state, self.lookahead_length, self.device, self.qubit_embedding,
                                 self.embedding_dim, self.padding_dim) for cur_env in self.test_vec_env.envs],
                    dim=0
                ).to(self.device)

        mean_swaps = mean(swaps)
        print(f'### TEST SET mean_swaps: {mean_swaps}')
        print(f'### TEST SET swaps: {swaps}')

        return mean_swaps, swaps

    def benchmark(self, baseline: str = 'SABRE', show_circuits: bool = True) -> tuple[float, list[int]]:
        """
        Benchmark the performance of the agent after the training.

        Parameters
        ----------
        baseline - the baseline agent used to benchmark against. By default: SABRE
        show_circuits - whether to print benchmark circuits or not

        Returns
        -------
        a tuple that contains:
        - mean swaps for all benchmark envs
        - a list of swaps for all benchmark envs
        """
        self._reset_test_envs()

        swaps = []

        for env in self.test_vec_env.envs:
            swaps.append(benchmark(env, seed=None, baseline=baseline, show_circuits=show_circuits))

        mean_swaps = mean(swaps)
        print(f'### BENCHMARK mean_swaps: {mean_swaps}')
        print(f'### BENCHMARK swaps: {swaps}')

        return mean_swaps, swaps


    def save_snapshot(self, idx):
        """
        Save a snapshot of the agent. Used to save the progress during training.

        Parameters
        ----------
        idx - index of the snapshot to save (for file naming)
        """
        path = TEMP_MODEL_DIR.joinpath(f"model_temp_{idx}.pt")
        torch.save(self.state_dict(), path)

        print(f"Model snapshot saved at {path}")

    def learn(self, total_timesteps: int, ma_window: int = 10, batch_size: int = BATCH_SIZE,
              curriculum_scheduler: CurriculumScheduler | None = None,
              plot_graphs: bool = True, show_circuits: bool = True, save_model: bool = False) -> None:
        """
        Conduct the learning process of the agent
        :param total_timesteps: Number of episodes during the learning process
        :param ma_window: The size of the moving average window used for filtering out noise in the graphs
        :param batch_size: The size of the batch used for the learning process
        :param curriculum_scheduler: Curriculum scheduler used for the learning process.
            If None, no curriculum learning is applied. Otherwise, the training proceeds according to the curriculum as
            generated by the scheduler. The idea is to start with simpler examples, and then gradually progress to
            more difficult ones as the agent learns.
        :param plot_graphs: Whether to show the graphs that plot the data from the learning process at the end or not.
            Regardless of this setting, the plots get saved to file.
        :param show_circuits: Whether to print the benchmark circuits.
        :param save_model: Whether to save the model to file or not.
        """
        if curriculum_scheduler is not None:
            current_generator = curriculum_scheduler.make_schedule()
            self._make_envs(self.connection_graph, current_generator, self.lookahead_length)

        self.unfinished_indices = set(range(N_ENVS))
        self.vec_env.reset()

        for env in self.vec_env.envs:
            env_surpass_all_possible_gates(env)
            while env._state.is_done():
                env.reset()
                env_surpass_all_possible_gates(env)

            self.replay_buffer.store(env._state)

        # if MCTS_PROCESSES > 1:
        #     print("Parallel MCTS Init")
        #     # print(self.qubit_embedding.weight)
        #     self.qubit_embedding.cpu()
        #     self.mcts = ParallelMCTS(self.transformer, self.qubit_embedding, self.device, c=self.c_constant,
        #                              swap_options=self.action_space_size, lookahead_length=self.lookahead_length,
        #                              embedding_dim=self.embedding_dim, padding_dim=self.padding_dim, rollouts=200,
        #                              processes=MCTS_PROCESSES, merge_worker_trees=False)
        #     print("Parallel MCTS Init successful")
        #     self.qubit_embedding.to(self.device)
        #     # print(self.qubit_embedding.weight)
        # else:
        print("MCTS Init")
        self.mcts = MCTS(self.transformer, self.qubit_embedding, self.device, c=self.c_constant,
                                 swap_options=self.action_space_size, lookahead_length=self.lookahead_length,
                                 embedding_dim=self.embedding_dim, padding_dim=self.padding_dim, rollouts=200)
        print("MCTS Init successful")

        losses = []
        swaps = []
        swap_lists = []
        returns = []
        losses_l1 = []
        losses_l2 = []
        performances = []
        snapshot = 0

        print(f'### BENCHMARKING THE TEST CIRCUITS')

        benchmark_swaps, benchmark_swap_list = self.benchmark(baseline='SABRE', show_circuits=show_circuits)

        start = time.time()

        for i in tqdm(range(int(total_timesteps))):
            print(f"\n\nEpisode: {i+1}...")
            loss, l1, l2 = self.batch(batch_size=batch_size)
            if ((len(self.replay_buffer.buffer) > REPLAY_BUFFER_MIN_SIZE_MCTS or (
                    (len(self.unfinished_indices) == 0) and len(self.replay_buffer.buffer) > batch_size))
                    and curriculum_scheduler is not None):
                print(f'Running the test set (Episode: {i+1})...')
                ep_swaps, ep_swap_list = self.test()
                performance = compute_performance(ep_swaps, benchmark_swaps)
                print(f"### PERFORMANCE: {performance}")
            else:
                ep_swaps = 0.0
                ep_swap_list = []
                performance = 0.0

            print(f'### BENCHMARK mean_swaps: {benchmark_swaps}')
            print(f'### BENCHMARK swaps: {benchmark_swap_list}')

            if isinstance(loss, Tensor):
                loss = loss.item()
            if isinstance(l1, Tensor):
                l1 = l1.item()
            if isinstance(l2, Tensor):
                l2 = l2.item()

            if self.wandb_run is not None:
                if curriculum_scheduler is None:
                    self.wandb_run.log({
                        "loss": loss,
                        "l1": l1,
                        "l2": l2,
                        "swaps": ep_swaps,
                        "swap_list": ep_swap_list
                    })
                else:
                    self.wandb_run.log({
                        "loss": loss,
                        "l1": l1,
                        "l2": l2,
                        "swaps": ep_swaps,
                        "swap_list": ep_swap_list,
                        "curriculum_level": curriculum_scheduler.current_level,
                        "performance": performance
                    })

            swaps.append(ep_swaps)
            swap_lists.append(ep_swap_list)
            losses.append(loss)
            losses_l1.append(l1)
            losses_l2.append(l2)
            performances.append(performance)

            if ((len(self.replay_buffer.buffer) > REPLAY_BUFFER_MIN_SIZE_MCTS or (len(self.unfinished_indices) == 0))
                    and curriculum_scheduler is not None):
                force_update = len(self.unfinished_indices) == 0 and len(self.replay_buffer.buffer) < batch_size
                if curriculum_scheduler.update_schedule(performance, force_update):
                    print(f"### CURRICULUM UPDATED")
                    current_generator = curriculum_scheduler.make_schedule()
                    self._make_envs(self.connection_graph, current_generator, self.lookahead_length)

                    print(f'### BENCHMARKING THE TEST CIRCUITS UNDER A NEW CURRICULUM')
                    benchmark_swaps, benchmark_swap_list = self.benchmark(baseline='SABRE',
                                                                          show_circuits=show_circuits)


            if self.lr_decay < 1:
                self.scheduler.step()

            if i != 0 and i % SNAPSHOT_PERIOD == 0:
                self.save_snapshot(snapshot)
                snapshot += 1

        if MCTS_PROCESSES > 1:
            self.mcts.close_workers()

        end = time.time()

        print(f'loss_l1: {losses_l1}')
        print(f'loss_l2: {losses_l2}')

        ma_swaps = moving_average(swaps, window=ma_window)
        ma_loss = moving_average(losses, window=ma_window)
        # ma_loss_l1 = moving_average(losses_l1, window=ma_window)
        # ma_loss_l2 = moving_average(losses_l2, window=ma_window)
        # ma_returns = moving_average(returns, window=ma_window)

        if curriculum_scheduler is not None:
            print(f"Level log: {curriculum_scheduler.level_log}")
            print(f"Final curriculum level: {curriculum_scheduler.current_level}")

        self._reset_test_envs()
        test_circuits = [env._state.interaction_circuit for env in self.test_vec_env.envs]
        circuit_lengths = [len(test_circuit) for test_circuit in test_circuits]

        timestamp = get_timestamp()

        path = Path(self.log_dir.joinpath(timestamp + '.txt'))
        self.dir_path = path

        if save_model:
            model_path = MODEL_DIR.joinpath(f"model_{get_timestamp()}.pt")
            torch.save(self.state_dict(), model_path)
            print('Model saved successfully')

        with path.open('w', encoding='utf-8') as f:
            f.write(f'episodes: {total_timesteps}\n')
            f.write(f'Number of parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}\n')
            f.write(f'Number of qubits: {self.n_qubits}\n')
            f.write(f'Number of edges: {self.n_edges}\n')
            f.write(f'learning rate: {self.lr}\n')
            f.write(f'weight decay: {self.weight_decay}\n')
            f.write(f'lr_decay: {LR_DECAY}\n')
            f.write(f'lr_decay_step: {LR_DECAY_STEP}\n')
            f.write(f'max_grad_norm: {MAX_GRAD_NORM}\n')
            f.write(f'N_ENVS: {N_ENVS}\n')
            f.write(f'batch_size: {batch_size}\n')
            f.write(f'N_MCTS_WORKERS: {MCTS_PROCESSES}\n')
            f.write(f'swaps: {swaps}\n')
            f.write(f'ma_swaps: {ma_swaps}\n')
            f.write(f'benchmark_swaps: {benchmark_swaps}\n')
            f.write(f'losses: {losses}\n')
            f.write(f'ma_loss: {ma_loss}\n')
            f.write(f'losses_l1: {losses_l1}\n')
            # f.write(f'ma_loss_l1: {ma_loss_l1}\n')
            f.write(f'losses_l2: {losses_l2}\n')
            # f.write(f'ma_loss_l2: {ma_loss_l2}\n')
            f.write(f'swap_lists: {swap_lists}\n')
            f.write(f'benchmark_swap_lists: {benchmark_swap_list}\n')
            f.write(f'Test circuits: {test_circuits}\n')
            f.write(f'Performances: {performances}\n')
            if curriculum_scheduler is not None:
                f.write(f"Final curriculum level: {curriculum_scheduler.current_level}\n")
                f.write(f"Curriculum level log: {curriculum_scheduler.level_log}\n")
            f.write(f'The training took: {end - start} seconds\n')

        episodes = range(1, total_timesteps + 1)

        plot_dir = PLOT_DIR.joinpath(f'{timestamp}/')
        mkdir(plot_dir)

        plot(episodes, swaps, label='Swaps per Episode', xlabel='Episode', ylabel='Swaps', title='Swaps per Episode', show=plot_graphs, path=plot_dir.joinpath(f'Swaps_per_episode_{timestamp}.png'))
        plot(episodes, losses, label='Loss per Episode', xlabel='Episode', ylabel='Loss', title='Loss per Episode', show=plot_graphs, path=plot_dir.joinpath(f'Loss_per_episode_{timestamp}.png'))
        plot(episodes, losses_l1, label='Loss per Episode', xlabel='Episode', ylabel='Loss', title='Loss L1 per Episode', show=plot_graphs, path=plot_dir.joinpath(f'l1_per_episode_{timestamp}.png'))
        plot(episodes, losses_l2, label='Loss per Episode', xlabel='Episode', ylabel='Loss', title='Loss L2 per Episode', show=plot_graphs, path=plot_dir.joinpath(f'l2_per_episode_{timestamp}.png'))
        # plot(episodes, returns, label=f'Returns per episode', xlabel='Episode', ylabel='Return', title='Returns per episode')
        plot(episodes, ma_swaps, label=f'{ma_window}-Episode Moving Avg', xlabel='Episode', ylabel='Swaps', title=f'Swaps per episode {ma_window}-Episode MA', show=plot_graphs, path=plot_dir.joinpath(f'Swaps_per_episode_{ma_window}_MA_{timestamp}.png'))
        plot(episodes, ma_loss, label=f'{ma_window}-Episode Moving Avg', xlabel='Episode', ylabel='Loss', title=f'Loss per Episode {ma_window}-Episode MA', show=plot_graphs, path=plot_dir.joinpath(f'Loss_per_episode_{ma_window}_MA_{timestamp}.png'))

        circuits = range(N_ENVS)

        plot_subplots_2scales(circuits, [swap_lists[-1], benchmark_swap_list], [circuit_lengths],
                      labels1=['AlphaRouter (final agent)', 'SABRE'], labels2=['Circuit length'],
                      xlabel='Circuit idx', ylabel1='Swaps', ylabel2='Gates', title='Swaps per Circuit',
                      show=plot_graphs, path=plot_dir.joinpath(f'Swaps_per_circuit_{timestamp}.png'))

        if plot_graphs:
            for i, env in enumerate(self.test_vec_env.envs):
                render_rgb(env, title=f'Test circuit #{i}')
        # plot(episodes, ma_loss_l1, label=f'{ma_window}-Episode Moving Avg', xlabel='Game', ylabel='Loss', title=f'Loss L1 per Episode {ma_window}-Episode MA')
        # plot(episodes, ma_loss_l2, label=f'{ma_window}-Episode Moving Avg', xlabel='Game', ylabel='Loss', title=f'Loss L2 per Episode {ma_window}-Episode MA')
        # plot(episodes, ma_returns, label=f'{ma_window}-Episode Moving Avg', xlabel='Episode', ylabel='Return', title='Return per episode 50-Episode MA')

    def forward(self, env: Routing, plot: bool = False) -> List[int]:
        """
        Route a circuit.

        Parameters
        ----------
        env - enviornment that generates circuits, holds device info, etc.
        plot - if yes, plots and displays debug info

        Returns
        -------
        the list of actions the agent chose (i.e. edge indices, where the router inserted a SWAP gate in chronological order)
        """
        state: RoutingState = env._state
        assert len(state.connection_graph.edges) == self.action_space_size

        actions = []

        for i in range(self.episode_limit):
            if plot:
                plot_mapping_and_circuit(env._state.connection_graph, env._state.mapping, env._state.interaction_circuit)
                time.sleep(0.1)
            node_embedding = embed_nodes(env._state, self.lookahead_length, self.device, self.qubit_embedding,
                        self.embedding_dim, self.padding_dim)
            node_embedding = node_embedding.unsqueeze(0)  # now shape (1, T, E)
            logits, _ = self.transformer(node_embedding)
            transformer_action, _ = choose_actions(logits)

            if isinstance(transformer_action, Tensor):
                transformer_action = transformer_action.cpu().item()

            observation, reward, done, truncated, info = env.step(transformer_action)
            actions.append(transformer_action)

            # print(f'action: {transformer_action}')
            # print(f'mapping: {env._state.mapping}')
            # print(f'env._state.interaction_circuit[env._state.position]: {env._state.interaction_circuit[env._state.position]}')
            # print(f'is_legal_surpass: {env._state.is_legal_surpass(*env._state.interaction_circuit[env._state.position])}')

            while env._state.is_legal_surpass(*env._state.interaction_circuit[env._state.position]):
                observation, reward, done, truncated, info = env.step(self.action_space_size)
                if plot:
                    plot_mapping_and_circuit(env._state.connection_graph, env._state.mapping,
                                         env._state.interaction_circuit)
                if done:
                    break

            if done:
                return actions

        print(f'Failed to schedule a correct circuit within the action limit.', file=sys.stderr)

        return actions


if __name__ == '__main__':
    # print("Hello world")

    # boost_process_priority()

    connection_graph = nx.Graph()
    connection_graph.add_edges_from([(0,1), (1,2), (2,3)])
    # connection_graph.add_edges_from([(0,1), (1,2), (2,3), (3,4)])
    # connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3,4), (4,5), (5,0)])
    # connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3,4), (4,5), (5,6), (6,7)])
    # connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9)])
    # connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3,4), (5,6), (6,7), (7,8), (8,9), (10,11),(11,12),(12,13), (13,14), (0,5), (5, 10), (1, 6), (6,11), (2,7), (7,12), (3,8), (8,13), (4,9), (9,14)])
    nx.draw(connection_graph, with_labels=True)
    plt.show()

    print(f"cuda version: {torch.version.cuda}")  # e.g. "11.8"
    print(f"cuda availability: {torch.cuda.is_available()}")
    print(f"device count: {torch.cuda.device_count()}")  # number of GPUs
    print(f"devices: {torch.cuda.get_device_name}")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"device: {device}")
    path = MODEL_DIR.joinpath(f"model_{get_timestamp()}.pt")

    model = AlphaRouterAgent(connection_graph=connection_graph, interaction_generator=None,
                             c_constant=1.2217286199661843, d_model=17 * 5, num_layers=3, num_heads=5, embedding_dim=31,
                             lookahead_length=44, lr=0.00031907082400580507,
                             device=device, episode_limit=100).to(device)

    ### LOAD AND RESUME TRAINING OF A SPECIFIC MODEL INSTEAD OF STARTING FROM SCRATCH???
    # path = MODEL_DIR.joinpath(f"model_2026-02-04T13-22-11-738563.pt")
    # path = TEMP_MODEL_DIR.joinpath(f"model_temp_18.pt")
    # model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    print(path)

    print(f'Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    curriculum_scheduler = CurriculumScheduler(start_level=1, max_level=100, inc_value=1, proceed_threshold=2.5,
                                               retreat_threshold=5.0, cooldown=20,
                                               curriculum_generator=GateLengthCurriculumGenerator())

    start = time.time()
    model.learn(128, batch_size=64, curriculum_scheduler=curriculum_scheduler)
    end = time.time()
    print(f'The training took: {end - start} seconds')

    print('Successful training')

    torch.save(model.state_dict(), path)
    # print(f"Model saved: {model.state_dict()}")
    print('Model saved successfully')

    env = Routing(
        max_observation_reach=LOOKAHEAD_LENGTH,
        observe_legal_surpasses=True,
        observe_connection_graph=True,
        connection_graph=connection_graph,
        render_mode="rgb_array",
        rewarder=None,
        interaction_generator=CustomInteractionGenerator()
    )

    ### LOAD A SPECIFIC MODEL INSTEAD OF THE LAST ONE???
    # path = MODEL_DIR.joinpath(f"model_1028_far_gate_patched_qgym_paper_hyper_2-2025-12-04T11-27-24-601885.pt")

    inference_model = AlphaRouterAgent(connection_graph=connection_graph, device=device, episode_limit=100).to(device)
    inference_model.load_state_dict(torch.load(path, weights_only=True))
    inference_model.eval()
    print('Model loaded successfully')
    # print(f"Model loaded: {inference_model.state_dict()}")

    env.reset()
    actions = inference_model(env, plot=False)
    print('### ### ###')
    print(f'len(actions): {len(actions)}')
    print(f'actions: {actions}')

    print('Successful inference')
