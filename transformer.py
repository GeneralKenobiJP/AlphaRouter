import math

import torch
from torch import nn, Tensor
from torch.amp import autocast


class PositionalEncoding(nn.Module):
    """
    A positional encoding module.
    Takes in node embedding and adds positional information.
    """
    def __init__(self, embedding_dim=24, lookahead_length=48):
        """
        :param embedding_dim: Embedding dimension of the (padded) node embeddings
        :param lookahead_length: The maximal number of the gates ahead that we will analyze
        """
        super().__init__()

        position = torch.arange(lookahead_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-math.log(1e5) / embedding_dim))
        pe = torch.zeros(lookahead_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Rounded up to the nearest even number so that odd indices slicing works
        final_embedding_dim_even = embedding_dim if embedding_dim % 2 == 0 else embedding_dim - 1
        pe[:, 1::2] = torch.cos(position * div_term[::final_embedding_dim_even])
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the positional encoding
        :param x: input tensor
        """
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return x


class TransformerLayer(nn.Module):
    """
    A single instance of Transformer layer of structure:
    - Multihead attention
    - Add & norm layer
    - Feed-forward network
    - Add & norm layer
    """
    def __init__(self, embedding_dim, num_heads, ffn_dim):
        """
        :param embedding_dim: Embedding dimension of the node embeddings
        :param num_heads: Number of heads in a multihead attention layer
        :param ffn_dim: Dimensionality of the hidden layer in the feed forward layer
        """
        super().__init__()

        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """
        Applies the transformer layer
        :param x: input tensor
        """
        attention_output, _ = self.multihead_attention(x, x, x)
        x = self.norm1(x + attention_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


def choose_action(transformer_output) -> (Tensor, Tensor, Tensor):
    # make_dot(transformer_output, params=dict(self.transformer.named_parameters())).render("graph", format='png')

    # Transform the transformer output shape from (number_of_logical_gates x number_of_swap_options) to (number of swap_options)
    # as we want to output the optimal swap action
    swap_scores = transformer_output.sum(dim=1)
    log_probs = torch.log_softmax(swap_scores, dim=-1)
    # softmax_output = torch.softmax(swap_scores, dim=-1)
    # dist = torch.distributions.Categorical(probs=softmax_output, validate_args=False)

    dist = torch.distributions.Categorical(logits=swap_scores, validate_args=False)
    # action = dist.sample()

    action = dist.sample()
    action_log_prob: Tensor = dist.log_prob(action)
    print('log_prob size: {}', action_log_prob.size())

    return action, action_log_prob, swap_scores, log_probs


class Transformer(nn.Module):
    """
    Complete transformer encoder used in the agent with linear decoder
    """
    def __init__(self, embedding_dim, num_heads, ffn_dim, num_layers, lookahead_length, swap_options):
        """
        :param embedding_dim: Embedding dimension of the node embeddings (with the padding and depth)
        :param num_heads: Number of heads in a multihead attention layer
        :param ffn_dim: Dimensionality of the hidden layer in the feed forward layer
        :param num_layers: Number of transformer layers
        :param lookahead_length: The maximal number of the gates ahead that we will analyze
        :param swap_options: Number of possible swaps actions + 1 (to include no swap)
        """
        super().__init__()

        self.positional_encoding = PositionalEncoding(embedding_dim, lookahead_length)

        self.layers = nn.ModuleList([
            TransformerLayer(embedding_dim, num_heads, ffn_dim) for _ in range(num_layers)
        ])

        # Linear decoder that outputs the matrix of shape (number_of_logical_gates x number_of_swap_options)
        self.decoder = nn.Linear(embedding_dim, swap_options)

    def forward(self, x):
        """
        Pads the input accordingly, applies the transformer encoder and linear decoder
        :param x: input tensor
        """
        # with autocast(self.device.type):
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x)
        output = self.decoder(x)
        return choose_action(output)
