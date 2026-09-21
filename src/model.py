from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import random
import encoder
# import Encoder from encoder

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.Encoder = encoder.Encoder()

        prob: float = random.random()

        values_size: int = len(self.Encoder.lookuptable())
        max_txt_len: int = 1024
        seq_len = 10
        embedding_dimension: int = 256
        hidden_dimension: int = 1024
        self.embedding_tensor = nn.Parameter(torch.randn(values_size, embedding_dimension))
        self.positional_embedding_tensor = nn.Parameter(torch.randn(max_txt_len, embedding_dimension))
        self.mask_tensor = nn.Linear(1, embedding_dimension)

        self.layers = nn.ModuleList(
            nn.ModuleList([
                    nn.LayerNorm(embedding_dimension),
                    nn.Linear(embedding_dimension, 3*embedding_dimension),

                    nn.LayerNorm(embedding_dimension),
                    nn.Linear(embedding_dimension, hidden_dimension),
                    nn.ReLU(),
                    nn.Linear(hidden_dimension, embedding_dimension)
            ])
            for _ in range(seq_len)
        )

        self.fc = nn.Linear(embedding_dimension, values_size)

    def attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        d_k: int = K.shape[-1]
        return torch.softmax((Q @ K.transpose(-2, -1)) / (d_k**0.5), dim=-1) @ V

    def self_attention(self, x: torch.Tensor):
        return self.attention(x, x, x)

    def embed(self, x: torch.Tensor, probabilities: torch.Tensor):
        t_masked: torch.Tensor = self.mask_tensor(probabilities)
        # print(t_masked.shape)
        return self.embedding_tensor[x] + self.positional_embedding_tensor[:x.shape[-1]] + t_masked

    def forward(self, x: torch.Tensor, probabilities: torch.Tensor):
        # print("begin")
        x_embedded = self.embed(x, probabilities=probabilities)
        for norm1, w_qkv, norm2, fc1, relu, fc2 in self.layers: #type: ignore
            Q, K, V = w_qkv(norm1(x_embedded)).chunk(3, dim=-1)
            # print(Q.shape, K.shape, V.shape)
            x_embedded = x_embedded + self.attention(Q=Q, K=K, V=V)
            x_embedded = x_embedded + fc2(relu(fc1(norm2(x_embedded))))
        x_embedded = self.fc(x_embedded)
        return x_embedded
