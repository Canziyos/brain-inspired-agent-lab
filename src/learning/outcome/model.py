import torch
from torch import nn

from src.learning.outcome.events import EVENT_TYPES
from src.learning.outcome_features import OUTCOME_FEATURE_COUNT


class RateRecurrentOutcomeModel(nn.Module):
    def __init__(
        self,
        neuron_count: int = 24,
        neural_ticks: int = 8,
        leak: float = 0.35,
    ) -> None:
        super().__init__()

        if neuron_count <= 0:
            raise ValueError("neuron_count must be positive")

        if neural_ticks <= 0:
            raise ValueError("neural_ticks must be positive")

        if not 0.0 < leak <= 1.0:
            raise ValueError("leak must be in (0, 1]")

        self.neuron_count = neuron_count
        self.neural_ticks = neural_ticks
        self.leak = leak

        self.input_synapses = nn.Parameter(
            torch.empty(
                neuron_count,
                OUTCOME_FEATURE_COUNT,
            )
        )

        self.recurrent_synapses = nn.Parameter(
            torch.empty(
                neuron_count,
                neuron_count,
            )
        )

        self.neuron_bias = nn.Parameter(torch.zeros(neuron_count))

        self.state_change_head = nn.Linear(neuron_count, 3)
        self.event_head = nn.Linear(neuron_count, len(EVENT_TYPES))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_synapses)
        nn.init.orthogonal_(self.recurrent_synapses)

        with torch.no_grad():
            self.recurrent_synapses.mul_(0.5)

        nn.init.zeros_(self.neuron_bias)

        self.state_change_head.reset_parameters()
        self.event_head.reset_parameters()

    def initial_neural_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.neuron_count,
            device=device,
            dtype=dtype,
        )

    def neural_tick(
        self,
        features: torch.Tensor,
        neural_state: torch.Tensor,
    ) -> torch.Tensor:
        input_drive = features @ self.input_synapses.T
        recurrent_drive = neural_state @ self.recurrent_synapses.T

        candidate_state = torch.tanh(
            input_drive
            + recurrent_drive
            + self.neuron_bias
        )

        return (
            (1.0 - self.leak) * neural_state
            + self.leak * candidate_state
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        neural_state = self.initial_neural_state(
            features.shape[0],
            device=features.device,
            dtype=features.dtype,
        )

        state_changes, event_logits, _ = self.forward_with_state(
            features,
            neural_state,
        )

        return state_changes, event_logits

    def forward_with_state(
        self,
        features: torch.Tensor,
        neural_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, feature]")

        if neural_state.ndim != 2:
            raise ValueError("neural_state must have shape [batch, neuron]")

        if features.shape[0] != neural_state.shape[0]:
            raise ValueError(
                "features and neural_state must have the same batch size"
            )

        for _ in range(self.neural_ticks):
            neural_state = self.neural_tick(features, neural_state)

        state_changes = self.state_change_head(neural_state)
        event_logits = self.event_head(neural_state)

        return state_changes, event_logits, neural_state