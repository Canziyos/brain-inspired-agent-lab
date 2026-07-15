import torch

from src.core.dynamics_types import EventType
from src.learning.outcome import (
    OutcomePrediction,
    OutcomeTrainingResult,
    RateRecurrentOutcomeModel,
    predict_outcome_with_state,
)
from src.learning.types import (
    FeatureVector,
    StateDelta,
)
from src.telemetry.metrics import OutcomeModelMetrics


def mean_absolute_state_delta_error(
    prediction: OutcomePrediction,
    actual: StateDelta,
) -> float:
    return (
        abs(prediction.energy_change - actual[0])
        + abs(prediction.health_change - actual[1])
        + abs(prediction.curiosity_change - actual[2])
    ) / 3.0


def predict_from_reset_state(
    model: RateRecurrentOutcomeModel,
    features: FeatureVector,
    reference_state: torch.Tensor,
) -> OutcomePrediction:
    reset_neural_state = model.initial_neural_state(
        batch_size=1,
        device=reference_state.device,
        dtype=reference_state.dtype,
    )

    prediction, _next_state = predict_outcome_with_state(
        model=model,
        features=features,
        neural_state=reset_neural_state,
    )

    return prediction


def build_outcome_model_metrics(
    prediction: OutcomePrediction,
    reset_prediction: OutcomePrediction,
    actual_event: EventType,
    actual_state_delta: StateDelta,
    training_result: OutcomeTrainingResult | None,
    neural_state: torch.Tensor,
) -> OutcomeModelMetrics:
    state_mae = mean_absolute_state_delta_error(
        prediction=prediction,
        actual=actual_state_delta,
    )

    reset_state_mae = mean_absolute_state_delta_error(
        prediction=reset_prediction,
        actual=actual_state_delta,
    )

    return OutcomeModelMetrics(
        predicted_energy_change=prediction.energy_change,
        actual_energy_change=actual_state_delta[0],

        predicted_health_change=prediction.health_change,
        actual_health_change=actual_state_delta[1],

        predicted_curiosity_change=prediction.curiosity_change,
        actual_curiosity_change=actual_state_delta[2],

        predicted_event=prediction.event,
        actual_event=actual_event,
        event_correct=(
            prediction.event is actual_event
        ),

        state_mae=state_mae,
        reset_state_mae=reset_state_mae,
        persistent_better_than_reset=(
            state_mae < reset_state_mae
        ),

        total_loss=(
            training_result.total_loss
            if training_result is not None
            else None
        ),
        state_loss=(
            training_result.state_loss
            if training_result is not None
            else None
        ),
        event_loss=(
            training_result.event_loss
            if training_result is not None
            else None
        ),

        neural_state_mean=float(
            neural_state.mean().item()
        ),
        neural_state_std=float(
            neural_state.std(
                unbiased=False
            ).item()
        ),
        neural_state_min=float(
            neural_state.min().item()
        ),
        neural_state_max=float(
            neural_state.max().item()
        ),
    )
