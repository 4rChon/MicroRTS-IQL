# modified from source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from iql_microrts.train_config import IQLModelConfig

TensorBatch = list[torch.Tensor]


class SpatialEmbedding(nn.Module):
    def __init__(
        self,
        space_dim: int,
        h: int,
        w: int,
        hidden_size: int
    ):
        super().__init__()
        self.space_dim = space_dim
        self.h = h
        self.w = w
        self.hidden_size = hidden_size

        self.embed_space = nn.Sequential(
            nn.Conv2d(
                self.space_dim,
                hidden_size // 2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(hidden_size // 2),
            nn.GELU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(
                hidden_size // 2,
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(hidden_size),
            nn.GELU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Flatten(),
            nn.Linear(
                int(hidden_size * self.h * self.w * 0.25 * 0.25),
                hidden_size * 2
            ),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        x = x.reshape(batch_size, self.h, self.w, self.space_dim)
        x = x.permute(0, 3, 1, 2)

        return self.embed_space(x).reshape(batch_size, self.hidden_size)


class Squeeze(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims: list[int],
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] | None = None,
        squeeze_output: bool = False,
        dropout: float | None = None,
    ):
        super().__init__()
        n_dims = len(dims)
        assert n_dims >= 2, \
            "MLP must have at least two dimensions (input and output)"

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorPolicy(nn.Module):
    def __init__(
        self,
        state_encoding: nn.Module,
        state_dim: tuple[int, int, int],
        act_dim: list[int],
        hidden_size: int,
        dropout: float | None = None
    ):
        super().__init__()
        self.act_dim = act_dim
        self.h, self.w, self.state_c = state_dim
        self.hidden_size = hidden_size
        self.act_c = sum(act_dim)

        self.state_encoding = state_encoding
        self.mlp = MLP([
            hidden_size,
            hidden_size * 2,
            hidden_size * 4,
            hidden_size * 8,
            self.h * self.w * self.act_c,
        ], dropout=dropout)

        self.register_buffer("mask_value", torch.tensor(-1e8))

    def _sample(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        return torch.searchsorted(
            torch.softmax(logits, dim=-1).cumsum(dim=-1),
            torch.rand((logits.shape[:-1]), device=logits.device)[..., None]
        )

    def forward(
        self,
        x: torch.Tensor,
        action_masks: torch.Tensor
    ) -> torch.Tensor:
        x = self.state_encoding(x)
        logits = self.mlp(x)

        grid_logits = logits.reshape(-1, self.h * self.w, self.act_c)
        action_masks = action_masks.reshape(-1, self.h * self.w, self.act_c)

        grid_logits = torch.where(  # type: ignore
            action_masks.bool(),
            grid_logits,
            self.mask_value  # type: ignore
        )
        split_logits = torch.split(grid_logits, self.act_dim, dim=-1)

        action_probs = torch.cat([
            torch.softmax(logits, dim=-1) for logits
            in split_logits
        ], dim=-1)
        return action_probs.reshape((-1, self.h, self.w, self.act_c))

    @torch.no_grad()
    def act(self, x: torch.Tensor, action_masks: torch.Tensor) -> torch.Tensor:
        x = self.state_encoding(x)
        logits = self.mlp(x)

        grid_logits = logits.reshape(-1, self.h * self.w, self.act_c)
        action_masks = action_masks.reshape(-1, self.h * self.w, self.act_c)

        grid_logits = torch.where(  # type: ignore
            action_masks.bool(),
            grid_logits,
            self.mask_value  # type: ignore
        )

        split_logits = torch.split(grid_logits, self.act_dim, dim=-1)

        return torch.cat([
            self._sample(logits) for logits
            in split_logits
        ], dim=-1)


class QFunction(nn.Module):
    def __init__(
        self,
        state_encoding: nn.Module,
        action_encoding: nn.Module,
        hidden_layers: int = 1,
        hidden_dim: int = 256,
        dropout: float | None = None,
        skip_connection: bool = False
    ):
        super().__init__()

        self.skip_connection = skip_connection
        self.state_encoding = state_encoding
        self.action_encoding = action_encoding

        self.mlps = []
        for _ in range(hidden_layers):
            self.mlps.append(MLP([hidden_dim * 2, hidden_dim * 2],
                                 dropout=dropout))
        self.mlps = nn.ModuleList(self.mlps)

        self.out = MLP(
            [
                hidden_dim * 2,
                hidden_dim,
                1
            ],
            squeeze_output=True,
            dropout=dropout
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        state = self.state_encoding(state)
        action = self.action_encoding(action)
        x = torch.cat([state, action], dim=-1)
        x = x.view(x.shape[0], -1)
        x = F.relu(x)
        for i, mlp in enumerate(self.mlps):
            if self.skip_connection and i > 0:
                x = mlp(x) + x
            else:
                x = mlp(x)
            x = F.relu(x)

        return self.out(x)


class TwinQ(nn.Module):
    def __init__(
        self,
        state_encoding: nn.Module,
        action_encoding: nn.Module,
        hidden_layers: int = 3,
        hidden_dim: int = 256,
        skip_connection: bool = False,
    ):
        super().__init__()
        self.q1 = QFunction(
            state_encoding,
            action_encoding,
            hidden_layers,
            hidden_dim,
            skip_connection=skip_connection
        )
        self.q2 = QFunction(
            state_encoding,
            action_encoding,
            hidden_layers,
            hidden_dim,
            skip_connection=skip_connection
        )

    def both(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action), self.q2(state, action)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(
        self,
        state_encoding: nn.Module,
        hidden_layers: int = 1,
        hidden_dim: int = 256,
        dropout: float | None = None,
        skip_connection: bool = False
    ):

        super().__init__()
        self.skip_connection = skip_connection
        self.state_encoding = state_encoding

        self.fc = MLP(
            [hidden_dim, hidden_dim * 2],
            dropout=dropout
        )

        self.mlps = []
        for _ in range(hidden_layers):
            self.mlps.append(MLP([hidden_dim * 2, hidden_dim * 2],
                                 dropout=dropout))

        self.mlps = nn.ModuleList(self.mlps)

        self.out = MLP(
            [
                hidden_dim * 2,
                hidden_dim,
                1
            ],
            squeeze_output=True,
            dropout=dropout
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.state_encoding(state)

        x = self.fc(x)
        x = F.relu(x)
        for i, mlp in enumerate(self.mlps):
            if self.skip_connection and i > 0:
                x = mlp(x) + x
            else:
                x = mlp(x)
            x = F.relu(x)

        return self.out(x)


class IQLNetwork():
    def __init__(
        self,
        state_dim: tuple[int, int, int],
        action_dim: list[int],
        config: IQLModelConfig,
        device: torch.device
    ):
        print("Initializing IQLNetwork...")
        hidden_layers = config.hidden_layers
        hidden_dim = config.hidden_dim
        dropout = config.dropout

        a_state_encoding = SpatialEmbedding(
            state_dim[2], state_dim[0], state_dim[1], hidden_dim
        ).to(device)
        c_state_encoding = SpatialEmbedding(
            state_dim[2], state_dim[0], state_dim[1], hidden_dim
        ).to(device)
        v_state_encoding = SpatialEmbedding(
            state_dim[2], state_dim[0], state_dim[1], hidden_dim
        ).to(device)
        action_encoding = SpatialEmbedding(
            sum(action_dim), state_dim[0], state_dim[1], hidden_dim
        ).to(device)

        self.actor = ActorPolicy(
            a_state_encoding,
            state_dim,
            action_dim,
            hidden_dim,
            dropout=dropout
        ).to(device)

        self.critic = TwinQ(
            c_state_encoding,
            action_encoding,
            hidden_layers,
            hidden_dim,
            skip_connection=False,
        ).to(device)

        self.value = ValueFunction(
            v_state_encoding,
            hidden_layers,
            hidden_dim,
            skip_connection=False,
        ).to(device)
