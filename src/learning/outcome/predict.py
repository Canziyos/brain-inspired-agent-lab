import torch

from src.learning.outcome.events import EVENT_TYPES
from src.learning.outcome.model import RateRecurrentOutcomeModel
from src.learning.outcome.types import OutcomePrediction
from src.learning.outcome_features import OUTCOME_FEATURE_COUNT
from src.learning.types import FeatureVector


def predict_outcome_with_state(
    model: RateRecurrentOutcomeModel,
    features: FeatureVector,
    neural_state: torch.Tensor,
) -> tuple[OutcomePrediction, torch.Tensor]:
    if len(features) != OUTCOME_FEATURE_COUNT:
        raise ValueError(
            "features must contain exactly "
            f"{OUTCOME_FEATURE_COUNT} values, got {len(features)}"
        )

    was_training = model.training
    model.eval()

    feature_tensor = torch.tensor(
        features,
        dtype=neural_state.dtype,
        device=neural_state.device,
    ).unsqueeze(0)

    with torch.no_grad():
        state_changes, event_logits, updated_neural_state = (
            model.forward_with_state(
                feature_tensor,
                neural_state,
            )
        )

        probabilities = torch.softmax(event_logits, dim=-1)

    if was_training:
        model.train()

    predicted_event_index = int(probabilities.argmax(dim=-1).item())
    changes = state_changes[0]

    prediction = OutcomePrediction(
        energy_change=float(changes[0].item()),
        health_change=float(changes[1].item()),
        curiosity_change=float(changes[2].item()),
        event=EVENT_TYPES[predicted_event_index],
        event_probabilities=tuple(
            float(value)
            for value in probabilities[0].tolist()
        ),
    )

    return prediction, updated_neural_state.detach()