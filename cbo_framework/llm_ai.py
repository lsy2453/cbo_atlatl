"""
LLM-based AI agent for Atlatl.

The agent implements the same process(message) interface as the bundled Atlatl
heuristic AIs. It supports:

- mock: deterministic local fallback for tests
- qwen/openai-compatible: DashScope Qwen or any OpenAI-compatible endpoint

Configure Qwen with environment variables, never hard-code API keys:

    LLM_API_KEY      required for qwen/openai-compatible
                     or DASHSCOPE_API_KEY for DashScope/Qwen
    LLM_BASE_URL     optional, defaults to DashScope compatible-mode endpoint
    LLM_MODEL        optional, defaults to qwen-plus
    LLM_CACHE_PATH   optional, defaults to cbo_framework/llm_cache.jsonl
"""

import hashlib
import json
import os
import random
import sys
import urllib.error
import urllib.request

sys.path.append(os.path.join(os.path.dirname(__file__),
                             "..", "atlatl-public-master", "server"))

import map as atlatl_map
import unit as atlatl_unit

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    OpenAI = None
    HAS_OPENAI = False


DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen-plus"


SYSTEM_PROMPT = """You are a tactical AI commanding one side in a hex-grid wargame.

Rules:
- Turn-based: Blue and Red alternate. Each surviving unit gets one action per phase.
- Legal actions are exactly the JSON objects supplied by the user.
- Actions include pass, move, or fire.
- Infantry has range 1 and stable movement across clear/rough/marsh/urban terrain.
- Armor and mechanized infantry move faster on clear terrain but slower on rough,
  marsh, and urban terrain.
- Artillery has range 2 but cannot enter marsh.
- Urban and rough terrain reduce damage against infantry. Marsh increases damage
  against armor, mechanized infantry, and artillery.
- Score is from Blue's perspective: Blue gains points for Red losses and loses
  twice as many points for Blue losses. City control also accumulates per phase.

Choose one tactically reasonable legal action. Preserve friendly strength, exploit
range and terrain, and avoid moving into unsupported contact.

Respond with ONLY one valid JSON action object copied from the supplied legal actions.
Do not include reasoning, markdown, or extra text.
"""


def format_game_state(obs, role):
    """Convert Atlatl observation to compact text for the LLM."""
    lines = [f"ROLE: {role}"]
    status = obs.get("status", {})
    lines.append(
        f"phase={status.get('phaseCount')} score={status.get('score')} "
        f"on_move={status.get('onMove')}"
    )

    city_owner = status.get("cityOwner", {})
    if city_owner:
        lines.append("cities=" + json.dumps(city_owner, sort_keys=True))

    friendly = []
    enemy = []
    for u in obs.get("units", []):
        if u.get("ineffective", False):
            continue
        unit_line = (
            f"{u.get('faction')} {u.get('longName')} type={u.get('type')} "
            f"hex={u.get('hex')} str={u.get('currentStrength')} "
            f"canMove={u.get('canMove')}"
        )
        if u.get("faction") == role:
            friendly.append(unit_line)
        elif u.get("hex") != "fog":
            enemy.append(unit_line)

    lines.append("FRIENDLY:")
    lines.extend(f"  {line}" for line in friendly)
    lines.append("ENEMY:")
    lines.extend(f"  {line}" for line in enemy)
    return "\n".join(lines)


def _is_legal(action, legal_actions):
    """Check if action exactly matches any legal action by required keys."""
    if not isinstance(action, dict):
        return False
    for legal in legal_actions:
        if all(action.get(key) == value for key, value in legal.items()):
            return True
    return False


def parse_llm_response(response_text, legal_actions):
    """Parse LLM text into a legal Atlatl action, falling back to pass."""
    text = (response_text or "").strip()

    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            action = json.loads(text[start:end])
            if _is_legal(action, legal_actions):
                return action
        except json.JSONDecodeError:
            pass

    for action in legal_actions:
        if action.get("type") == "pass":
            return action
    return {"type": "pass"}


class JsonlActionCache:
    """Small persistent cache keyed by role, state text, and legal actions."""

    def __init__(self, path):
        self.path = path
        self.items = {}
        self._load()

    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    self.items[row["key"]] = row["action"]
                except (KeyError, json.JSONDecodeError):
                    continue

    def make_key(self, role, state_text, legal_actions, model):
        payload = {
            "role": role,
            "state": state_text,
            "legal": legal_actions,
            "model": model,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key):
        return self.items.get(key)

    def set(self, key, action):
        if not self.path:
            return
        self.items[key] = action
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "action": action},
                               ensure_ascii=False) + "\n")


class LLM_AI:
    """
    LLM-driven Atlatl AI.

    backend values:
    - mock
    - qwen
    - openai-compatible
    """

    def __init__(self, role, kwargs=None):
        kwargs = kwargs or {}
        self.role = role
        self.mapData = None
        self.unitData = None
        self.param = None
        self.call_count = 0

        self.backend = kwargs.get("backend", "mock")
        if self.backend in ("openai", "dashscope"):
            self.backend = "openai-compatible"

        self.temperature = float(kwargs.get(
            "temperature", os.environ.get("LLM_TEMPERATURE", 0.1)
        ))
        self.max_tokens = int(kwargs.get(
            "max_tokens", os.environ.get("LLM_MAX_TOKENS", 200)
        ))
        self.model = kwargs.get("model") or os.environ.get(
            "LLM_MODEL", DEFAULT_QWEN_MODEL
        )
        self.base_url = kwargs.get("base_url") or os.environ.get(
            "LLM_BASE_URL", DEFAULT_QWEN_BASE_URL
        )
        self.api_key = (
            kwargs.get("api_key")
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
        )
        cache_path = kwargs.get("cache_path") or os.environ.get(
            "LLM_CACHE_PATH",
            os.path.join(os.path.dirname(__file__), "llm_cache.jsonl"),
        )
        self.cache = JsonlActionCache(cache_path)

        self.client = None
        if self.backend in ("qwen", "openai-compatible"):
            if HAS_OPENAI and self.api_key:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            elif self.api_key:
                self.client = "raw-http"
            else:
                print("Warning: LLM backend unavailable; falling back to mock. "
                      "Set LLM_API_KEY to use Qwen.")
                self.backend = "mock"

    def _mock_decision(self, legal_actions):
        """Deterministic low-cost fallback for local tests."""
        random.seed(self.call_count)
        self.call_count += 1
        fire_actions = [a for a in legal_actions if a.get("type") == "fire"]
        move_actions = [a for a in legal_actions if a.get("type") == "move"]
        if fire_actions:
            return fire_actions[0]
        if move_actions:
            return random.choice(move_actions)
        return {"type": "pass"}

    def _call_openai_compatible(self, state_text, legal_actions):
        legal_text = json.dumps(legal_actions, ensure_ascii=False, indent=1)
        user_prompt = (
            f"{state_text}\n\n"
            f"Legal actions:\n{legal_text}\n\n"
            "Return exactly one legal action JSON object."
        )
        if self.client == "raw-http":
            return self._call_openai_compatible_raw(user_prompt)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    def _call_openai_compatible_raw(self, user_prompt):
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        parsed = json.loads(body)
        return parsed["choices"][0]["message"]["content"]

    def _choose_action(self, state_text, legal_actions):
        if self.backend == "mock":
            return self._mock_decision(legal_actions)

        key = self.cache.make_key(self.role, state_text, legal_actions, self.model)
        cached = self.cache.get(key)
        if cached is not None and _is_legal(cached, legal_actions):
            return cached

        try:
            response_text = self._call_openai_compatible(state_text, legal_actions)
            action = parse_llm_response(response_text, legal_actions)
        except Exception as exc:
            print(f"Warning: LLM call failed ({exc}); using pass fallback.")
            action = parse_llm_response("", legal_actions)

        self.cache.set(key, action)
        return action

    def process(self, message, response_fn=None):
        """Process Atlatl JSON message and return an Atlatl action wrapper."""
        msg = json.loads(message)

        if msg["type"] == "parameters":
            self.param = msg["parameters"]
            self.mapData = atlatl_map.MapData()
            self.unitData = atlatl_unit.UnitData()
            atlatl_map.fromPortable(self.param["map"], self.mapData)
            atlatl_unit.fromPortable(self.param["units"], self.unitData,
                                     self.mapData)
            return json.dumps({"type": "role-request", "role": self.role})

        if msg["type"] == "observation":
            obs = msg["observation"]
            status = obs["status"]
            if status["isTerminal"] or status["onMove"] != self.role:
                return None
            if status["setupMode"]:
                return json.dumps({"type": "action", "action": {"type": "pass"}})

            for unit_obs in obs["units"]:
                uid = unit_obs["faction"] + " " + unit_obs["longName"]
                if uid in self.unitData.unitIndex:
                    unit_obj = self.unitData.unitIndex[uid]
                    unit_obj.partialObsUpdate(unit_obs, self.unitData, self.mapData)

            from game import Game
            game = Game(self.param)
            state = {"units": obs["units"], "status": obs["status"]}
            legal_actions = game.legal_actions(state)
            state_text = format_game_state(obs, self.role)
            action = self._choose_action(state_text, legal_actions)
            return json.dumps({"type": "action", "action": action})

        if msg["type"] == "reset":
            return None

        return None


AI = LLM_AI
