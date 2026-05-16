"""
Configuration for Causal Bayesian Optimization on Atlatl.
Defines decision variables (d), adversarial variables (u),
and the pre-specified causal graph structure.
"""

import os

# 自动检测路径：假设 atlatl-public-master/ 和 cbo_framework/ 在同一父目录
ATLATL_SERVER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "atlatl-public-master", "server")
)

# ============================================================
# Decision variables d (Blue-controllable)
# ============================================================
DECISION_VARS = {
    "n_blue": {
        "type": "integer",
        "low": 1,
        "high": 4,
        "description": "Number of blue infantry units"
    },
    "blue_side": {
        "type": "categorical",
        "categories": ["north", "south", "east", "west"],
        "description": "Blue deployment side"
    },
}

# ============================================================
# Adversarial variables u (Environment / Red-controllable)
# ============================================================
ADVERSARIAL_VARS = {
    "n_red": {
        "type": "integer",
        "low": 1,
        "high": 4,
        "description": "Number of red infantry units"
    },
    "red_ai": {
        "type": "categorical",
        "categories": ["passive", "shootback", "pass-agg", "agg"],
        "description": "Red AI policy type"
    },
    "max_phases": {
        "type": "integer",
        "low": 6,
        "high": 20,
        "description": "Maximum game phases (scenario length)"
    },
    "p_urban": {
        "type": "continuous",
        "low": 0.0,
        "high": 0.5,
        "description": "Probability of urban terrain per hex"
    },
    "p_rough": {
        "type": "continuous",
        "low": 0.0,
        "high": 0.5,
        "description": "Probability of rough terrain per hex"
    },
    "p_marsh": {
        "type": "continuous",
        "low": 0.0,
        "high": 0.3,
        "description": "Probability of marsh terrain per hex"
    },
    "scenario_seed": {
        "type": "integer",
        "low": 0,
        "high": 99999,
        "description": "Random seed for scenario generation"
    },
}

# ============================================================
# Fixed parameters (not optimized)
# ============================================================
MAP_SIZE = 5          # 5x5 hex grid
BLUE_AI = "pass-agg"  # Blue AI under test (fixed)
N_EVAL_SEEDS = 3      # Number of seeds to average over per evaluation

# ============================================================
# Causal graph edges (derived from Atlatl code)
# Format: (parent, child, mechanism_description)
# ============================================================
CAUSAL_EDGES = [
    # Layer 0 -> Layer 1: Inputs to intermediate mechanisms
    ("n_blue",   "force_ratio",   "direct: n_blue / n_red"),
    ("n_red",    "force_ratio",   "direct: n_blue / n_red"),
    ("p_urban",  "terrain_map",   "scenario_gen: hex terrain assignment"),
    ("p_rough",  "terrain_map",   "scenario_gen: hex terrain assignment"),
    ("p_marsh",  "terrain_map",   "scenario_gen: hex terrain assignment"),
    ("scenario_seed", "terrain_map", "scenario_gen: randomization"),
    ("red_ai",   "red_behavior",  "airegistry: selects AI class"),

    # Layer 1 -> Layer 2: Mechanisms to game dynamics
    ("force_ratio",  "posture",     "pass_agg.getPosture: atk if ratio>=1"),
    ("terrain_map",  "mobility",    "mobility.cost: terrain x unit_type"),
    ("terrain_map",  "fire_effect", "combat.terrain_multiplier"),
    ("red_behavior", "engagement",  "AI decision loop"),
    ("force_ratio",  "engagement",  "determines numerical advantage"),

    # Layer 2 -> Layer 3: Dynamics to score components
    ("posture",    "attrition_score", "attack posture -> engage enemies"),
    ("fire_effect","attrition_score", "damage = str x fp x terrain x 0.5"),
    ("engagement", "attrition_score", "red actions cause blue losses"),
    ("mobility",   "city_score",     "movement -> city capture"),
    ("max_phases", "city_score",     "city score accumulates per phase"),

    # Layer 3 -> Output
    ("attrition_score", "y", "+1 per red str lost, -2 per blue str lost"),
    ("city_score",      "y", "24/n_cities per phase per owned city"),
]

# Causal graph node layers for visualization / kernel design
CAUSAL_LAYERS = {
    "inputs_d":    ["n_blue", "blue_side"],
    "inputs_u":    ["n_red", "red_ai", "max_phases", "p_urban", "p_rough",
                    "p_marsh", "scenario_seed"],
    "mediators_1": ["force_ratio", "terrain_map", "red_behavior"],
    "mediators_2": ["posture", "mobility", "fire_effect", "engagement"],
    "scores":      ["attrition_score", "city_score"],
    "output":      ["y"],
}
