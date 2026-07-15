from src.core.perception import AgentState


LOW_ENERGY = 30.0
MEDIUM_ENERGY = 60.0
HIGH_ENERGY = 85.0

FOOD_SCORE_DESPERATE = 90.0
FOOD_SCORE_HUNGRY = 70.0
FOOD_SCORE_USEFUL = 40.0
FOOD_SCORE_FULL = 12.0

REST_SCORE_EXHAUSTED = 60.0
REST_SCORE_TIRED = 25.0
REST_SCORE_MILD = 8.0
REST_SCORE_NONE = 0.0

CURIOSITY_TO_MYSTERY_SCORE = 0.6

def food_motivation(state: AgentState) -> float:
    if state.energy < LOW_ENERGY:
        return FOOD_SCORE_DESPERATE
    
    if state.energy < MEDIUM_ENERGY:
        return FOOD_SCORE_HUNGRY
    
    if state.energy < HIGH_ENERGY:
        return FOOD_SCORE_USEFUL
    
    return FOOD_SCORE_FULL

def food_action_reason(
    state: AgentState,
) -> str:
    if state.energy < LOW_ENERGY:
        return "desperate for food"

    if state.energy < MEDIUM_ENERGY:
        return "hungry, food useful"

    if state.energy < HIGH_ENERGY:
        return "food nearby"

    return "food nearby, but already full"

def rest_motivation(state: AgentState) -> float:
    if state.energy < LOW_ENERGY:
        return REST_SCORE_EXHAUSTED

    if state.energy < MEDIUM_ENERGY:
        return REST_SCORE_TIRED

    if state.energy < HIGH_ENERGY:
        return REST_SCORE_MILD

    return REST_SCORE_NONE

def mystery_motivation(state: AgentState) -> float:
    return state.curiosity * CURIOSITY_TO_MYSTERY_SCORE