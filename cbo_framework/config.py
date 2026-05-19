"""
Configuration for single-level causal Bayesian optimization on Atlatl.

The experiment searches directly over scenario parameters:

    x* = argmin_x E[y | do(x)]

where y is the Blue AI score. Lower scores indicate stronger adversarial
scenarios against the fixed Blue policy.
"""

import os


ATLATL_SERVER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "atlatl-public-master", "server")
)


# ============================================================
# Unified scenario input space x
# ============================================================

SIDE_CATEGORIES = ["north", "south", "east", "west"]
AI_CATEGORIES = ["passive", "shootback", "pass-agg", "agg"]
OPTIONAL_LLM_AI_CATEGORIES = ["llm-qwen"]
UNIT_TYPE_CATEGORIES = ["infantry", "mechinf", "armor", "artillery"]

SCENARIO_VARS = {
    "n_blue": {
        "type": "integer",
        "low": 1,
        "high": 4,
        "description": "Number of Blue units",
    },
    "blue_side": {
        "type": "categorical",
        "categories": SIDE_CATEGORIES,
        "description": "Blue deployment side",
    },
    "blue_unit_type": {
        "type": "categorical",
        "categories": UNIT_TYPE_CATEGORIES,
        "description": "Unit type used by all Blue units",
    },
    "n_red": {
        "type": "integer",
        "low": 1,
        "high": 4,
        "description": "Number of Red units",
    },
    "red_ai": {
        "type": "categorical",
        "categories": AI_CATEGORIES,
        "description": "Red AI policy",
    },
    "max_phases": {
        "type": "integer",
        "low": 6,
        "high": 20,
        "description": "Maximum number of game phases",
    },
    "p_urban": {
        "type": "continuous",
        "low": 0.0,
        "high": 0.5,
        "description": "Urban terrain probability per hex",
    },
    "p_rough": {
        "type": "continuous",
        "low": 0.0,
        "high": 0.5,
        "description": "Rough terrain probability per hex",
    },
    "p_marsh": {
        "type": "continuous",
        "low": 0.0,
        "high": 0.3,
        "description": "Marsh terrain probability per hex",
    },
    "red_unit_type": {
        "type": "categorical",
        "categories": UNIT_TYPE_CATEGORIES,
        "description": "Unit type used by all Red units",
    },
}

VAR_NAMES = list(SCENARIO_VARS.keys())
VAR_INDICES = {name: i for i, name in enumerate(VAR_NAMES)}

# Backward-compatible aliases for older scripts that imported these names.
DECISION_VARS = SCENARIO_VARS
ADVERSARIAL_VARS = {}


# ============================================================
# Fixed parameters
# ============================================================

MAP_SIZE = 5
BLUE_AI = "pass-agg"
N_EVAL_SEEDS = 3
BASE_SCENARIO_SEED = 42


# ============================================================
# Causal graph edges
# ============================================================

CAUSAL_EDGES = [
    # Inputs -> generated scenario state
    ("n_blue", "force_balance", "initial Blue force count"),
    ("n_red", "force_balance", "initial Red force count"),
    ("blue_unit_type", "force_balance", "unit capability and strength profile"),
    ("red_unit_type", "force_balance", "unit capability and strength profile"),
    ("blue_side", "deployment", "Blue setup region"),
    ("n_blue", "deployment", "number of Blue placements"),
    ("n_red", "deployment", "number of Red placements"),
    ("p_urban", "terrain_map", "probabilistic terrain assignment"),
    ("p_rough", "terrain_map", "probabilistic terrain assignment"),
    ("p_marsh", "terrain_map", "probabilistic terrain assignment"),
    ("red_ai", "red_behavior", "policy class selection"),

    # Scenario state -> game mechanisms
    ("force_balance", "posture", "pass_agg posture from current strengths"),
    ("terrain_map", "mobility", "mobility.cost lookup"),
    ("blue_unit_type", "mobility", "movement budget and terrain costs"),
    ("red_unit_type", "mobility", "movement budget and terrain costs"),
    ("terrain_map", "fire_effect", "combat.terrain_multiplier lookup"),
    ("blue_unit_type", "fire_effect", "combat.firepower and range lookup"),
    ("red_unit_type", "fire_effect", "combat.firepower and range lookup"),
    ("deployment", "engagement", "initial contact geometry"),
    ("red_behavior", "engagement", "Red action policy"),
    ("posture", "engagement", "Blue pass-agg attack/defense mode"),
    ("mobility", "engagement", "reachable positions over time"),
    ("fire_effect", "engagement", "damage opportunities over time"),
    ("max_phases", "engagement", "trajectory horizon"),

    # Game mechanisms -> score components
    ("engagement", "attrition_score", "who fires at whom over the trajectory"),
    ("fire_effect", "attrition_score", "damage calculation"),
    ("mobility", "city_score", "ability to occupy urban hexes"),
    ("terrain_map", "city_score", "urban hex distribution"),
    ("max_phases", "city_score", "per-phase accumulation window"),

    # Score components -> outcome
    ("attrition_score", "y", "+1 Red strength lost, lossPenalty Blue strength lost"),
    ("city_score", "y", "city control score accumulated by phase"),
]

CAUSAL_LAYERS = {
    "inputs": VAR_NAMES,
    "generated_state": ["force_balance", "terrain_map", "deployment", "red_behavior"],
    "mechanisms": ["posture", "mobility", "fire_effect", "engagement"],
    "scores": ["attrition_score", "city_score"],
    "output": ["y"],
}

PATHWAY_GROUPS = {
    "force_balance": ["n_blue", "n_red", "blue_unit_type", "red_unit_type"],
    "deployment": ["blue_side", "n_blue", "n_red"],
    "terrain": ["p_urban", "p_rough", "p_marsh"],
    "red_policy": ["red_ai"],
    "horizon": ["max_phases"],
    "unit_terrain": ["blue_unit_type", "red_unit_type", "p_urban", "p_rough", "p_marsh"],
}
