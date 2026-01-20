import math

import torch
from torch import nn, Tensor


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

        position = torch.arange(lookahead_length, dtype=torch.float32).unsqueeze(1)  # (T,1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-math.log(1e5) / embedding_dim))  # (d/2,)
        pe = torch.zeros(lookahead_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # position = torch.arange(lookahead_length, dtype=torch.float32).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-math.log(1e5) / embedding_dim))
        # pe = torch.zeros(lookahead_length, embedding_dim)
        # pe[:, 0::2] = torch.sin(position * div_term)
        #
        # # Rounded up to the nearest even number so that odd indices slicing works
        # final_embedding_dim_even = embedding_dim if embedding_dim % 2 == 0 else embedding_dim - 1
        # pe[:, 1::2] = torch.cos(position * div_term[::final_embedding_dim_even])
        # self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the positional encoding
        :param x: input tensor
        """
        T = x.size(1)
        if T > self.pe.size(0):
            raise ValueError(f"T={T} exceeds lookahead_length={self.pe.size(0)}")
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


def choose_action(transformer_output) -> (Tensor, Tensor, Tensor, Tensor):
    # make_dot(transformer_output, params=dict(self.transformer.named_parameters())).render("graph", format='png')

    # Transform the transformer output shape from (number_of_logical_gates x number_of_swap_options) to (number of swap_options)
    # as we want to output the optimal swap action
    swap_scores = transformer_output.mean(dim=1)
    log_probs: Tensor = torch.log_softmax(swap_scores, dim=-1)

    log_probs = torch.nan_to_num(log_probs, nan=-1e-9, posinf=1e9, neginf=-1e9)

    dist = torch.distributions.Categorical(logits=swap_scores, validate_args=False)

    action = dist.sample()
    action_log_prob: Tensor = dist.log_prob(action)

    return action, action_log_prob, swap_scores, log_probs


class Transformer(nn.Module):
    """
    Complete transformer encoder used in the agent with linear decoder
    """
    def __init__(self, embedding_dim, num_heads, d_model, ffn_dim, num_layers, lookahead_length, swap_options, device = torch.device("cpu")):
        """
        :param embedding_dim: Embedding dimension of the node embeddings (with the padding and depth)
        :param num_heads: Number of heads in a multihead attention layer
        :param ffn_dim: Dimensionality of the hidden layer in the feed forward layer
        :param num_layers: Number of transformer layers
        :param lookahead_length: The maximal number of the gates ahead that we will analyze
        :param swap_options: Number of possible swaps actions + 1 (to include no swap)
        """
        super().__init__()

        self.device = device

        self.input_projection = nn.Linear(embedding_dim, d_model)

        self.positional_encoding = PositionalEncoding(d_model, lookahead_length)

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, ffn_dim) for _ in range(num_layers)
        ])

        # 1) Generic defaults for common modules
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Default: Glorot/Xavier uniform for linear layers (good mid-ground for transformers).
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                # LayerNorm: scale=1, bias=0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.MultiheadAttention):
                # MultiheadAttention stores qkv as in_proj_weight/in_proj_bias and out_proj
                if hasattr(m, 'in_proj_weight') and m.in_proj_weight is not None:
                    nn.init.xavier_uniform_(m.in_proj_weight)
                if hasattr(m, 'in_proj_bias') and m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                # out projection
                if hasattr(m, 'out_proj') and m.out_proj is not None:
                    nn.init.xavier_uniform_(m.out_proj.weight)
                    if m.out_proj.bias is not None:
                        nn.init.zeros_(m.out_proj.bias)

        # 2) Override / special-case initializations

        # Input projection: keep small but standard Xavier (helps stable feature magnitudes)
        nn.init.xavier_uniform_(self.input_projection.weight)
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)

        # Linear decoder that outputs the matrix of shape (number_of_logical_gates x number_of_swap_options)
        self.decoder = nn.Linear(d_model, swap_options)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.decoder.bias)

        self.value_head = nn.Linear(d_model, 1)
        # self.value_head.bias.data.fill_(-5.0)
        nn.init.zeros_(self.decoder.bias)
        nn.init.normal_(self.value_head.weight, 0.0, 1e-3)

        # print(f'Transformer parameters initialised to {self.state_dict()}')

    def forward(self, x):
        """
        Pads the input accordingly, applies the transformer encoder and linear decoder
        :param x: input tensor
        """
        # with autocast(self.device.type):
        x = self.input_projection(x)

        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x)
        output = self.decoder(x)
        # print(f'shape: {output.shape}')
        state_values = self.value_head(x)
        state_values = state_values.squeeze(-1)
        state_values = state_values.mean(dim=1)

        return *choose_action(output), state_values
