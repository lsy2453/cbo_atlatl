"""
Direct synchronous evaluator for Atlatl games.

This module maps a 10-dimensional scenario vector x into an Atlatl scenario,
runs the fixed Blue AI against the selected Red AI, and returns the Blue score.
"""

import json
import os
import random
import sys
import types

import numpy as np

import config

_atlatl_path = config.ATLATL_SERVER_PATH
if _atlatl_path not in sys.path:
    sys.path.insert(0, _atlatl_path)

import map as atlatl_map
import unit as atlatl_unit
from game import Game

try:
    import websockets  # noqa: F401
except ModuleNotFoundError:
    # The synchronous evaluator never uses the websocket client helpers in
    # Atlatl AI modules, but several modules import websockets at top level.
    sys.modules["websockets"] = types.SimpleNamespace(connect=None)


def create_scenario(n_blue, blue_side, blue_unit_type, n_red, red_unit_type,
                    max_phases, p_urban, p_rough, p_marsh, scenario_seed,
                    map_size=None):
    """
    Generate an Atlatl scenario dict from scenario parameters.

    The terrain probabilities are interpreted in order urban -> rough -> marsh
    -> clear. Candidate generation enforces p_urban + p_rough + p_marsh <= 1.
    """
    if map_size is None:
        map_size = config.MAP_SIZE

    rng = random.Random(scenario_seed)

    mapData = atlatl_map.MapData()
    mapData.createHexGrid(map_size, map_size)

    for hex_obj in mapData.hexes():
        r = rng.random()
        if r < p_urban:
            hex_obj.terrain = "urban"
        elif r < p_urban + p_rough:
            hex_obj.terrain = "rough"
        elif r < p_urban + p_rough + p_marsh:
            hex_obj.terrain = "marsh"

    side_map = {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
    }
    red_side = side_map[blue_side]

    blue_hexes = _get_setup_hexes(map_size, blue_side)
    red_hexes = _get_setup_hexes(map_size, red_side)

    unitData = atlatl_unit.UnitData()

    def place_units(faction, unit_type, hexes, count):
        available = list(hexes)
        rng.shuffle(available)
        placed = min(count, len(available))
        for i in range(placed):
            u_param = {
                "hex": available[i],
                "type": unit_type,
                "longName": str(i),
                "faction": faction,
                "currentStrength": 100,
            }
            atlatl_unit.Unit(u_param, unitData, mapData)

    place_units("blue", blue_unit_type, blue_hexes, n_blue)
    place_units("red", red_unit_type, red_hexes, n_red)

    scenario = {
        "map": mapData.toPortable(),
        "units": unitData.toPortable(),
        "score": {
            "maxPhases": max_phases,
            "lossPenalty": -2,
            "cityScore": 24,
        },
    }
    scenario["map"]["fogOfWar"] = False
    return scenario


def _get_setup_hexes(size, side, margin=1):
    """Get valid hex IDs for a deployment side."""
    import math

    low = math.floor(size / 2) - margin
    high = math.floor(size / 2) + margin
    hexes = []
    if side == "north":
        for i in range(size):
            for j in range(low + 1):
                hexes.append(f"hex-{i}-{j}")
    elif side == "south":
        for i in range(size):
            for j in range(high, size):
                hexes.append(f"hex-{i}-{j}")
    elif side == "east":
        for i in range(high, size):
            for j in range(size):
                hexes.append(f"hex-{i}-{j}")
    elif side == "west":
        for i in range(low + 1):
            for j in range(size):
                hexes.append(f"hex-{i}-{j}")
    return hexes


def _create_ai(ai_name, role):
    """Instantiate an AI by name from the Atlatl AI modules."""
    if ai_name == "passive":
        from ai.passive import AI
        return AI(role, {})
    if ai_name == "shootback":
        from ai.shootback import AI
        return AI(role, {})
    if ai_name == "pass-agg":
        from ai.pass_agg import AI
        return AI(role, {})
    if ai_name == "agg":
        from ai.pass_agg import AI
        return AI(role, {"mode": "agg"})
    if ai_name == "pass":
        from ai.pass_agg import AI
        return AI(role, {"mode": "pass"})
    if ai_name == "random":
        from ai.random_actor import AI
        return AI(role, {})
    if ai_name.startswith("llm"):
        from llm_ai import LLM_AI
        kwargs = {"backend": "mock"}
        if ai_name in ("llm-qwen", "llm-dashscope"):
            kwargs = {"backend": "qwen"}
        elif ai_name in ("llm-openai", "llm-compatible"):
            kwargs = {"backend": "openai-compatible"}
        return LLM_AI(role, kwargs)
    if ai_name == "pascal":
        from ai.dl_alpha_beta import AI
        return AI(role, {
            "debug": False,
            "neuralNet": "ai/pass-v-pass-g3",
            "depthLimit": "1",
        })
    if ai_name in ("pass-agg-fp", "pass-agg-fog", "field", "dijkstra",
                   "mcts1k", "burtplus", "stomp"):
        from airegistry import ai_registry
        cls, kwargs = ai_registry[ai_name]
        return cls(role, kwargs)
    raise ValueError(f"Unknown AI: {ai_name}")


def run_game(scenario, blue_ai_name, red_ai_name):
    """Run a single Atlatl game synchronously and return the Blue score."""
    game = Game(scenario)
    try:
        import current_game_access
        current_game_access.server = types.SimpleNamespace(game=game)
    except Exception:
        pass
    state = game.initial_state()

    blue_ai = _create_ai(blue_ai_name, "blue")
    red_ai = _create_ai(red_ai_name, "red")

    param_msg = json.dumps({"type": "parameters", "parameters": scenario})
    blue_ai.process(param_msg)
    red_ai.process(param_msg)

    while not game.is_terminal(state):
        on_move = game.on_move(state)
        ai = blue_ai if on_move == "blue" else red_ai
        obs = game.observation(state, on_move)
        obs_msg = json.dumps({"type": "observation", "observation": obs})
        response_str = ai.process(obs_msg)
        if response_str is None:
            action = {"type": "pass"}
        else:
            response = json.loads(response_str)
            action = response.get("action", {"type": "pass"})
        state = game.transition(state, action)

    return game.score(state)


def evaluate(x_vars, n_seeds=None, base_seed=None):
    """
    Evaluate a scenario configuration x.

    scenario_seed is treated as an exogenous Monte Carlo seed, not an optimized
    scenario coordinate. The returned mean estimates E[y | do(x)].
    """
    if n_seeds is None:
        n_seeds = config.N_EVAL_SEEDS
    if base_seed is None:
        base_seed = config.BASE_SCENARIO_SEED

    scores = []
    for i in range(n_seeds):
        seed = base_seed + i * 1000
        scenario = create_scenario(
            n_blue=x_vars["n_blue"],
            blue_side=x_vars["blue_side"],
            blue_unit_type=x_vars["blue_unit_type"],
            n_red=x_vars["n_red"],
            red_unit_type=x_vars["red_unit_type"],
            max_phases=x_vars["max_phases"],
            p_urban=x_vars["p_urban"],
            p_rough=x_vars["p_rough"],
            p_marsh=x_vars["p_marsh"],
            scenario_seed=seed,
        )
        scores.append(run_game(scenario, config.BLUE_AI, x_vars["red_ai"]))

    scores = np.array(scores, dtype=np.float64)
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scores": scores.tolist(),
        "n_eval": n_seeds,
    }


def _cat_to_index(value, categories):
    return categories.index(value)


def _index_to_cat(value, categories):
    idx = int(round(float(value)))
    idx = min(max(idx, 0), len(categories) - 1)
    return categories[idx]


def vars_to_vector(x_vars):
    """Convert scenario variable dict to a numeric vector for GP modeling."""
    return np.array([
        x_vars["n_blue"],
        _cat_to_index(x_vars["blue_side"], config.SIDE_CATEGORIES),
        _cat_to_index(x_vars["blue_unit_type"], config.UNIT_TYPE_CATEGORIES),
        x_vars["n_red"],
        _cat_to_index(x_vars["red_ai"], config.AI_CATEGORIES),
        x_vars["max_phases"],
        x_vars["p_urban"],
        x_vars["p_rough"],
        x_vars["p_marsh"],
        _cat_to_index(x_vars["red_unit_type"], config.UNIT_TYPE_CATEGORIES),
    ], dtype=np.float64)


def vector_to_vars(vec):
    """Convert a numeric vector back to typed scenario variables."""
    vec = np.asarray(vec, dtype=np.float64)
    p_urban = float(np.clip(vec[6], 0.0, 0.5))
    p_rough = float(np.clip(vec[7], 0.0, 0.5))
    p_marsh = float(np.clip(vec[8], 0.0, 0.3))
    total = p_urban + p_rough + p_marsh
    if total > 1.0:
        p_urban, p_rough, p_marsh = [
            p_urban / total,
            p_rough / total,
            p_marsh / total,
        ]

    return {
        "n_blue": int(np.clip(round(vec[0]), 1, 4)),
        "blue_side": _index_to_cat(vec[1], config.SIDE_CATEGORIES),
        "blue_unit_type": _index_to_cat(vec[2], config.UNIT_TYPE_CATEGORIES),
        "n_red": int(np.clip(round(vec[3]), 1, 4)),
        "red_ai": _index_to_cat(vec[4], config.AI_CATEGORIES),
        "max_phases": int(np.clip(round(vec[5]), 6, 20)),
        "p_urban": p_urban,
        "p_rough": p_rough,
        "p_marsh": p_marsh,
        "red_unit_type": _index_to_cat(vec[9], config.UNIT_TYPE_CATEGORIES),
    }


N_DIM = len(config.VAR_NAMES)
X_INDICES = list(range(N_DIM))
FORCE_INDICES = [
    config.VAR_INDICES["n_blue"],
    config.VAR_INDICES["blue_unit_type"],
    config.VAR_INDICES["n_red"],
    config.VAR_INDICES["red_unit_type"],
]
TERRAIN_INDICES = [
    config.VAR_INDICES["p_urban"],
    config.VAR_INDICES["p_rough"],
    config.VAR_INDICES["p_marsh"],
]
UNIT_TYPE_INDICES = [
    config.VAR_INDICES["blue_unit_type"],
    config.VAR_INDICES["red_unit_type"],
]
AI_INDEX = config.VAR_INDICES["red_ai"]
PHASE_INDEX = config.VAR_INDICES["max_phases"]


if __name__ == "__main__":
    x = {
        "n_blue": 3,
        "blue_side": "east",
        "blue_unit_type": "infantry",
        "n_red": 3,
        "red_ai": "pass-agg",
        "max_phases": 10,
        "p_urban": 0.1,
        "p_rough": 0.1,
        "p_marsh": 0.05,
        "red_unit_type": "infantry",
    }
    result = evaluate(x, n_seeds=2)
    print(f"Score: {result['mean']:.2f} +/- {result['std']:.2f}")
    print(f"Individual scores: {result['scores']}")
