"""
LLM-based AI agent for Atlatl.
Uses Claude API to make tactical decisions based on game state.

This is the third AI category for CBO analysis:
  1. Rule-based (pass-agg): deterministic heuristics
  2. Neural network (Pascal): learned from training data
  3. LLM (this): general reasoning, zero-shot

The LLM receives the full game state as structured text and reasons
about the best action. Its failure modes are fundamentally different:
- May misunderstand spatial hex relationships
- May over/under-value certain terrain or unit types
- May exhibit inconsistent strategies across similar states
- Decision quality depends on prompt framing
"""

import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__),
                "..", "atlatl-public-master", "server"))

import map as atlatl_map
import unit as atlatl_unit
import status


# ============================================================
# Option 1: Use Anthropic API (requires API key)
# ============================================================
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# ============================================================
# Option 2: Use OpenAI-compatible API (e.g., local LLM)
# ============================================================
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


SYSTEM_PROMPT = """You are a tactical AI commanding Blue forces in a hex-grid wargame.

GAME RULES:
- Turn-based: Blue and Red alternate. Each unit gets one action per phase.
- Actions: move to adjacent hex, fire at enemy in range, or pass.
- Unit types: infantry (range 1, moves 1 hex), armor (range 1, moves 1-2),
  mechanized infantry (range 1, moves 1-2), artillery (range 2, moves 1-2).
- Terrain effects on DEFENDER: urban/rough = 0.5x damage to infantry;
  marsh = 2x damage to armor/mechinf/artillery. Clear = no modifier.
- Mobility: infantry moves 1 hex on any terrain; armor/mechinf move 2 on clear,
  1 on rough/marsh/urban; artillery cannot enter marsh.
- Damage = attacker_strength * firepower_table * terrain_multiplier * 0.5
- Unit becomes ineffective below 50% strength.
- Score: +1 per red strength destroyed, -2 per blue strength lost,
  +/- city_score per phase for each city controlled.

STRATEGY GUIDANCE:
- Prioritize force preservation (losing blue costs 2x vs killing red 1x).
- Capture cities for ongoing score accumulation.
- Use terrain defensively (urban/rough halves damage to infantry).
- Concentrate fire on weak targets to get kills.
- Avoid exposing units to multiple enemies.

Respond with ONLY a valid JSON action object. Examples:
  {"type": "move", "mover": "blue 0", "destination": "hex-2-3"}
  {"type": "fire", "source": "blue 0", "target": "red 1"}
  {"type": "pass"}
"""


def format_game_state(obs, map_data, role):
    """Convert Atlatl observation to readable text for LLM."""
    lines = []
    lines.append(f"=== GAME STATE (you are {role.upper()}) ===")

    st = obs.get("status", {})
    lines.append(f"Phase: {st.get('phaseCount', '?')} | "
                 f"Score: {st.get('score', '?')} | "
                 f"On move: {st.get('onMove', '?')}")

    # Map info
    cities = st.get("cityOwner", {})
    if cities:
        city_str = ", ".join(f"{cid}({owner})" for cid, owner in cities.items())
        lines.append(f"Cities: {city_str}")

    # Units
    lines.append("\nYOUR UNITS:")
    for u in obs.get("units", []):
        if u["faction"] == role and not u.get("ineffective", False):
            status_str = "can_move" if u.get("canMove") else "already_moved"
            lines.append(f"  {u['faction']} {u['longName']} ({u['type']}) "
                        f"at {u.get('hex', '?')} "
                        f"str={u.get('currentStrength', '?')} "
                        f"[{status_str}]")

    lines.append("\nENEMY UNITS:")
    for u in obs.get("units", []):
        if u["faction"] != role and not u.get("ineffective", False):
            hex_loc = u.get("hex", "?")
            if hex_loc == "fog":
                continue
            lines.append(f"  {u['faction']} {u['longName']} ({u['type']}) "
                        f"at {hex_loc} "
                        f"str={u.get('currentStrength', '?')}")

    # Available actions hint
    lines.append("\nChoose ONE action for ONE of your units that can_move.")

    return "\n".join(lines)


def parse_llm_response(response_text, legal_actions):
    """
    Parse LLM response into a valid Atlatl action.
    Falls back to pass if parsing fails.
    """
    # Try to extract JSON from response
    text = response_text.strip()

    # Handle markdown code blocks
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            action = json.loads(text[start:end])
            # Validate against legal actions
            if _is_legal(action, legal_actions):
                return action
        except json.JSONDecodeError:
            pass

    # Fallback: pass
    return {"type": "pass"}


def _is_legal(action, legal_actions):
    """Check if action matches any legal action."""
    for legal in legal_actions:
        match = True
        for key in legal:
            if key not in action or action[key] != legal[key]:
                match = False
                break
        if match:
            return True
    return False


class LLM_AI:
    """
    LLM-based Atlatl AI agent.

    Supports multiple backends:
    - "claude": Anthropic Claude API
    - "openai": OpenAI-compatible API (works with local LLMs)
    - "mock": Deterministic mock for testing without API
    """

    def __init__(self, role, kwargs=None):
        if kwargs is None:
            kwargs = {}
        self.role = role
        self.mapData = None
        self.unitData = None
        self.param = None
        self.backend = kwargs.get("backend", "mock")
        self.model = kwargs.get("model", "claude-sonnet-4-20250514")
        self.api_key = kwargs.get("api_key", os.environ.get("ANTHROPIC_API_KEY"))
        self.temperature = kwargs.get("temperature", 0.3)
        self.call_count = 0

        # Initialize API client
        if self.backend == "claude" and HAS_ANTHROPIC and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif self.backend == "openai" and HAS_OPENAI:
            self.client = openai.OpenAI(
                api_key=kwargs.get("openai_key", os.environ.get("OPENAI_API_KEY")),
                base_url=kwargs.get("base_url", None)  # For local LLMs
            )
        else:
            self.client = None
            if self.backend != "mock":
                print(f"Warning: {self.backend} backend unavailable, using mock")
                self.backend = "mock"

    def _call_llm(self, state_text, legal_actions):
        """Call LLM and return action."""
        legal_str = json.dumps(legal_actions[:10], indent=1)  # Show some examples
        prompt = (f"{state_text}\n\nLegal actions (sample):\n{legal_str}\n\n"
                  f"Choose the best action. Respond with ONLY a JSON object.")

        if self.backend == "claude":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        elif self.backend == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=200,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content

        else:  # mock
            return self._mock_decision(legal_actions)

    def _mock_decision(self, legal_actions):
        """
        Mock LLM: simple heuristic that mimics LLM-like behavior.
        Prioritizes: fire > move toward enemy/city > pass.
        More "thoughtful" than pass-agg but less optimal.
        Introduces LLM-like failure modes:
        - Sometimes ignores terrain advantages
        - Occasionally over-commits to city capture
        - May split forces instead of concentrating
        """
        import random
        random.seed(self.call_count)
        self.call_count += 1

        fire_actions = [a for a in legal_actions if a["type"] == "fire"]
        move_actions = [a for a in legal_actions if a["type"] == "move"]

        if fire_actions:
            # LLM-like: usually picks the best target but sometimes wrong
            if random.random() < 0.8:
                return json.dumps(fire_actions[0])
            else:
                return json.dumps(random.choice(fire_actions))

        if move_actions:
            # LLM-like: tends to move toward cities (over-values them)
            # but sometimes makes spatial reasoning errors
            if random.random() < 0.7:
                # Move toward city if possible, otherwise random
                city_moves = []
                for a in move_actions:
                    dest = a["destination"]
                    if self.mapData and dest in self.mapData.hexIndex:
                        if self.mapData.hexIndex[dest].terrain == "urban":
                            city_moves.append(a)
                if city_moves:
                    return json.dumps(random.choice(city_moves))

            return json.dumps(random.choice(move_actions))

        return json.dumps({"type": "pass"})

    def process(self, message, response_fn=None):
        """Process Atlatl message and return action."""
        msgD = json.loads(message)

        if msgD['type'] == "parameters":
            self.param = msgD['parameters']
            self.mapData = atlatl_map.MapData()
            self.unitData = atlatl_unit.UnitData()
            atlatl_map.fromPortable(self.param['map'], self.mapData)
            atlatl_unit.fromPortable(self.param['units'], self.unitData,
                                     self.mapData)
            return json.dumps({"type": "role-request", "role": self.role})

        elif msgD['type'] == 'observation':
            obs = msgD['observation']
            if (not obs['status']['isTerminal'] and
                    obs['status']['onMove'] == self.role):

                if obs['status']['setupMode']:
                    return json.dumps({"type": "action",
                                       "action": {"type": "pass"}})

                # Update unit positions
                for unitObs in obs['units']:
                    uid = unitObs['faction'] + " " + unitObs['longName']
                    if uid in self.unitData.unitIndex:
                        un = self.unitData.unitIndex[uid]
                        un.partialObsUpdate(unitObs, self.unitData,
                                           self.mapData)

                # Get legal actions
                from game import Game
                game = Game(self.param)
                state = {"units": obs["units"], "status": obs["status"]}
                legal = game.legal_actions(state)

                # Format state for LLM
                state_text = format_game_state(obs, self.mapData, self.role)

                # Call LLM
                llm_response = self._call_llm(state_text, legal)

                # Parse response
                action = parse_llm_response(llm_response, legal)

                return json.dumps({"type": "action", "action": action})

        elif msgD['type'] == 'reset':
            return None

        return None


# ============================================================
# Register with Atlatl evaluator
# ============================================================
AI = LLM_AI  # For compatibility with Atlatl AI registry


if __name__ == "__main__":
    """Test the LLM AI standalone."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                    "..", "atlatl-public-master", "server"))

    from atlatl_evaluator import create_scenario, run_game

    # Test with mock backend (no API key needed)
    print("Testing LLM AI (mock backend)...")

    scenario = create_scenario(
        n_blue=3, blue_side="east", n_red=3,
        max_phases=10, p_urban=0.1, p_rough=0.1,
        p_marsh=0.05, scenario_seed=42
    )

    # Monkey-patch the evaluator to use LLM AI
    import atlatl_evaluator
    original_create_ai = atlatl_evaluator._create_ai

    def patched_create_ai(ai_name, role):
        if ai_name == "llm":
            return LLM_AI(role, {"backend": "mock"})
        return original_create_ai(ai_name, role)

    atlatl_evaluator._create_ai = patched_create_ai

    score = run_game(scenario, "llm", "pass-agg")
    print(f"LLM (blue) vs pass-agg (red): score = {score}")

    score2 = run_game(scenario, "pass-agg", "llm")
    print(f"pass-agg (blue) vs LLM (red): score = {score2}")
