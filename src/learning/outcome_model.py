import random
from dataclasses import dataclass

import torch
from torch import nn

from src.core.dynamics_types import EventType
from src.learning.outcome_features import (
    OUTCOME_FEATURE_COUNT,
)
from src.learning.samples import OutcomeSample


EVENT_TYPES = tuple(EventType)

EVENT_TO_INDEX = {
    event: index
    for index, event in enumerate(EVENT_TYPES)
}


@dataclass(frozen=True, slots=True)
class OutcomePrediction:
    energy_change: float
    health_change: float
    curiosity_change: float

    event: EventType
    event_probabilities: tuple[float, ...]

    final_neural_state: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class OutcomeTrainingResult:
    total_loss: float
    state_loss: float
    event_loss: float


class RateRecurrentOutcomeModel(nn.Module):
    def __init__(
        self,
        neuron_count: int = 24,
        neural_ticks: int = 8,
        leak: float = 0.35,
    ) -> None:
        super().__init__()

        if neuron_count <= 0:
            raise ValueError(
                "neuron_count must be positive"
            )

        if neural_ticks <= 0:
            raise ValueError(
                "neural_ticks must be positive"
            )

        if not 0.0 < leak <= 1.0:
            raise ValueError(
                "leak must be in (0, 1]"
            )

        self.neuron_count = neuron_count
        self.neural_ticks = neural_ticks
        self.leak = leak

        # Synapses from encoded inputs to neurons.
        self.input_synapses = nn.Parameter(
            torch.empty(
                neuron_count,
                OUTCOME_FEATURE_COUNT,
            )
        )

        # Synapses between neurons.
        self.recurrent_synapses = nn.Parameter(
            torch.empty(
                neuron_count,
                neuron_count,
            )
        )

        self.neuron_bias = nn.Parameter(
            torch.zeros(neuron_count)
        )

        self.state_change_head = nn.Linear(
            neuron_count,
            3,
        )

        self.event_head = nn.Linear(
            neuron_count,
            len(EVENT_TYPES),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(
            self.input_synapses
        )

        nn.init.orthogonal_(
            self.recurrent_synapses
        )

        # Prevent excessively strong recurrent activity
        # at birth.
        with torch.no_grad():
            self.recurrent_synapses.mul_(0.5)

        nn.init.zeros_(
            self.neuron_bias
        )

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
        input_drive = (
            features
            @ self.input_synapses.T
        )

        recurrent_drive = (
            neural_state
            @ self.recurrent_synapses.T
        )

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
        *,
        return_activity: bool = False,
    ):
        if features.ndim != 2:
            raise ValueError(
                "features must have shape "
                "[batch, feature]"
            )

        neural_state = self.initial_neural_state(
            features.shape[0],
            device=features.device,
            dtype=features.dtype,
        )

        activity: list[torch.Tensor] = []

        for _ in range(self.neural_ticks):
            neural_state = self.neural_tick(
                features,
                neural_state,
            )

            if return_activity:
                activity.append(neural_state)

        state_changes = self.state_change_head(
            neural_state
        )

        event_logits = self.event_head(
            neural_state
        )

        if return_activity:
            activity_trace = torch.stack(
                activity,
                dim=1,
            )

            return (
                state_changes,
                event_logits,
                activity_trace,
            )

        return state_changes, event_logits


def event_index(
    event: EventType,
) -> int:
    return EVENT_TO_INDEX[event]


def predict_outcome(
    model: RateRecurrentOutcomeModel,
    features: tuple[float, ...],
) -> OutcomePrediction:
    model.eval()

    feature_tensor = torch.tensor(
        features,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        (
            state_changes,
            event_logits,
            activity_trace,
        ) = model(
            feature_tensor,
            return_activity=True,
        )

        probabilities = torch.softmax(
            event_logits,
            dim=-1,
        )

    predicted_event_index = int(
        probabilities.argmax(
            dim=-1
        ).item()
    )

    changes = state_changes[0]
    final_state = activity_trace[0, -1]

    return OutcomePrediction(
        energy_change=float(
            changes[0].item()
        ),
        health_change=float(
            changes[1].item()
        ),
        curiosity_change=float(
            changes[2].item()
        ),
        event=EVENT_TYPES[
            predicted_event_index
        ],
        event_probabilities=tuple(
            float(value)
            for value in probabilities[0].tolist()
        ),
        final_neural_state=tuple(
            float(value)
            for value in final_state.tolist()
        ),
    )


def train_outcome_model(
    model: RateRecurrentOutcomeModel,
    optimizer: torch.optim.Optimizer,
    samples: list[OutcomeSample],
    batch_size: int = 8,
    rng: random.Random | None = None,
) -> OutcomeTrainingResult | None:
    if len(samples) < batch_size:
        return None

    random_source = (
        rng
        if rng is not None
        else random
    )

    batch = random_source.sample(
        samples,
        batch_size,
    )

    features = torch.tensor(
        [
            sample.features
            for sample in batch
        ],
        dtype=torch.float32,
    )

    state_changes = torch.tensor(
        [
            sample.state_changes
            for sample in batch
        ],
        dtype=torch.float32,
    )

    event_indices = torch.tensor(
        [
            sample.event_index
            for sample in batch
        ],
        dtype=torch.long,
    )

    model.train()

    (
        predicted_changes,
        event_logits,
    ) = model(features)

    state_loss = nn.functional.mse_loss(
        predicted_changes,
        state_changes,
    )

    event_loss = nn.functional.cross_entropy(
        event_logits,
        event_indices,
    )

    total_loss = (
        state_loss
        + event_loss
    )

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return OutcomeTrainingResult(
        total_loss=float(
            total_loss.item()
        ),
        state_loss=float(
            state_loss.item()
        ),
        event_loss=float(
            event_loss.item()
        ),
    )