"""
Direct synchronous evaluator for Atlatl games.
Bypasses the async server/websocket architecture.
Runs Game + AI.process() in a simple loop.
"""

import sys
import os
import json
import random
import copy
import numpy as np

# Add Atlatl server to path
# When run via run.py, path is already set.
# When run standalone, use config.
import config
_atlatl_path = config.ATLATL_SERVER_PATH
if _atlatl_path not in sys.path:
    sys.path.insert(0, _atlatl_path)

import map as atlatl_map
import unit as atlatl_unit
import combat
import mobility
from game import Game
import config


def create_scenario(n_blue, blue_side, n_red, max_phases,
                    p_urban, p_rough, p_marsh, scenario_seed,
                    map_size=None):
    """
    Generate an Atlatl scenario dict from CBO parameters.
    Returns a scenario dict compatible with Game().
    """
    if map_size is None:
        map_size = config.MAP_SIZE

    rng = random.Random(scenario_seed)

    # Build hex grid
    mapData = atlatl_map.MapData()
    mapData.createHexGrid(map_size, map_size)

    # Assign terrain based on probabilities
    for hex_obj in mapData.hexes():
        r = rng.random()
        if r < p_urban:
            hex_obj.terrain = "urban"
        elif r < p_urban + p_rough:
            hex_obj.terrain = "rough"
        elif r < p_urban + p_rough + p_marsh:
            hex_obj.terrain = "marsh"
        # else: stays "clear" (default)

    # Determine sides
    side_map = {
        "north": "south", "south": "north",
        "east": "west", "west": "east"
    }
    red_side = side_map[blue_side]

    # Get setup hexes for each side
    blue_hexes = _get_setup_hexes(map_size, blue_side)
    red_hexes = _get_setup_hexes(map_size, red_side)

    # Place units
    unitData = atlatl_unit.UnitData()

    def place_units(faction, hexes, count):
        available = list(hexes)
        rng.shuffle(available)
        placed = min(count, len(available))
        for i in range(placed):
            hex_id = available[i]
            u_param = {
                "hex": hex_id,
                "type": "infantry",
                "longName": str(i),
                "faction": faction,
                "currentStrength": 100
            }
            atlatl_unit.Unit(u_param, unitData, mapData)

    place_units("blue", blue_hexes, n_blue)
    place_units("red", red_hexes, n_red)

    score_params = {
        "maxPhases": max_phases,
        "lossPenalty": -2,
        "cityScore": 24
    }

    scenario = {
        "map": mapData.toPortable(),
        "units": unitData.toPortable(),
        "score": score_params
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
    """Instantiate an AI by name from the registry."""
    # Import AI classes directly to avoid full registry import
    if ai_name == "passive":
        from ai.passive import AI
        return AI(role, {})
    elif ai_name == "shootback":
        from ai.shootback import AI
        return AI(role, {})
    elif ai_name == "pass-agg":
        from ai.pass_agg import AI
        return AI(role, {})
    elif ai_name == "agg":
        from ai.pass_agg import AI
        return AI(role, {"mode": "agg"})
    elif ai_name == "pass":
        from ai.pass_agg import AI
        return AI(role, {"mode": "pass"})
    elif ai_name == "random":
        from ai.random_actor import AI
        return AI(role, {})
    elif ai_name.startswith("llm"):
        from llm_ai import LLM_AI
        kwargs = {"backend": "mock"}
        if ai_name == "llm-claude":
            kwargs = {"backend": "claude", "model": "claude-sonnet-4-20250514"}
        elif ai_name == "llm-openai":
            kwargs = {"backend": "openai"}
        return LLM_AI(role, kwargs)
    else:
        raise ValueError(f"Unknown AI: {ai_name}")


def run_game(scenario, blue_ai_name, red_ai_name):
    """
    Run a single Atlatl game synchronously.
    Returns the final score (from Blue's perspective).
    """
    game = Game(scenario)
    state = game.initial_state()

    # Create AI instances
    blue_ai = _create_ai(blue_ai_name, "blue")
    red_ai = _create_ai(red_ai_name, "red")

    # Send parameters to AIs
    param_msg = json.dumps({
        "type": "parameters",
        "parameters": scenario
    })
    blue_ai.process(param_msg)
    red_ai.process(param_msg)

    # Game loop
    while not game.is_terminal(state):
        on_move = game.on_move(state)

        if on_move == "blue":
            ai = blue_ai
        else:
            ai = red_ai

        # Create observation message
        obs = game.observation(state, on_move)
        obs_msg = json.dumps({"type": "observation", "observation": obs})

        # Get AI response
        response_str = ai.process(obs_msg)
        if response_str is None:
            # AI chose not to respond (terminal or not on move)
            action = {"type": "pass"}
        else:
            response = json.loads(response_str)
            action = response.get("action", {"type": "pass"})

        # Transition game state
        state = game.transition(state, action)

    return game.score(state)


def evaluate(d_vars, u_vars, n_seeds=None):
    """
    Evaluate a (d, u) configuration.
    Runs multiple games with different seeds and returns statistics.

    Parameters
    ----------
    d_vars : dict
        Decision variables: {n_blue, blue_side}
    u_vars : dict
        Adversarial variables: {n_red, red_ai, max_phases,
                                 p_urban, p_rough, p_marsh, scenario_seed}
    n_seeds : int
        Number of evaluation seeds to average over.

    Returns
    -------
    dict with keys: mean, std, scores, individual results
    """
    if n_seeds is None:
        n_seeds = config.N_EVAL_SEEDS

    blue_ai_name = config.BLUE_AI
    red_ai_name = u_vars["red_ai"]

    scores = []
    base_seed = u_vars.get("scenario_seed", 42)

    for i in range(n_seeds):
        seed = base_seed + i * 1000

        scenario = create_scenario(
            n_blue=d_vars["n_blue"],
            blue_side=d_vars["blue_side"],
            n_red=u_vars["n_red"],
            max_phases=u_vars["max_phases"],
            p_urban=u_vars["p_urban"],
            p_rough=u_vars["p_rough"],
            p_marsh=u_vars["p_marsh"],
            scenario_seed=seed,
        )

        score = run_game(scenario, blue_ai_name, red_ai_name)
        scores.append(score)

    scores = np.array(scores)
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scores": scores.tolist(),
        "n_eval": n_seeds,
    }


def vars_to_vector(d_vars, u_vars):
    """Convert variable dicts to a numeric vector for GP."""
    # Encode categoricals as integers
    side_map = {"north": 0, "south": 1, "east": 2, "west": 3}
    ai_map = {"passive": 0, "shootback": 1, "pass-agg": 2, "agg": 3}

    vec = [
        d_vars["n_blue"],
        side_map[d_vars["blue_side"]],
        u_vars["n_red"],
        ai_map[u_vars["red_ai"]],
        u_vars["max_phases"],
        u_vars["p_urban"],
        u_vars["p_rough"],
        u_vars["p_marsh"],
        u_vars["scenario_seed"] / 99999.0,  # normalize
    ]
    return np.array(vec, dtype=np.float64)


def vector_to_vars(vec):
    """Convert numeric vector back to variable dicts."""
    side_list = ["north", "south", "east", "west"]
    ai_list = ["passive", "shootback", "pass-agg", "agg"]

    d_vars = {
        "n_blue": int(round(vec[0])),
        "blue_side": side_list[int(round(vec[1]))],
    }
    u_vars = {
        "n_red": int(round(vec[2])),
        "red_ai": ai_list[int(round(vec[3]))],
        "max_phases": int(round(vec[4])),
        "p_urban": float(vec[5]),
        "p_rough": float(vec[6]),
        "p_marsh": float(vec[7]),
        "scenario_seed": int(round(vec[8] * 99999)),
    }
    return d_vars, u_vars


# Variable indices for kernel decomposition
D_INDICES = [0, 1]           # n_blue, blue_side
U_INDICES = [2, 3, 4, 5, 6, 7, 8]  # n_red, red_ai, max_phases, p_*, seed
FORCE_INDICES = [0, 2]       # n_blue, n_red (force ratio group)
TERRAIN_INDICES = [5, 6, 7]  # p_urban, p_rough, p_marsh
AI_INDEX = 3                 # red_ai
PHASE_INDEX = 4              # max_phases


if __name__ == "__main__":
    # Quick test
    d = {"n_blue": 3, "blue_side": "east"}
    u = {"n_red": 3, "red_ai": "pass-agg", "max_phases": 10,
         "p_urban": 0.1, "p_rough": 0.1, "p_marsh": 0.05,
         "scenario_seed": 42}
    print("Testing evaluator...")
    result = evaluate(d, u, n_seeds=2)
    print(f"Score: {result['mean']:.2f} +/- {result['std']:.2f}")
    print(f"Individual scores: {result['scores']}")
