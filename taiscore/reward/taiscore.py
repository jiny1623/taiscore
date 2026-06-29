"""
TAIScore: Targeted Actionable Improvement Score

Evaluates a critique-guided revision rollout (x, y0, c, y1):
  - Does the critique identify a real weakness?
  - Does the actor follow it?
  - Does the intended aspect improve?

Used as the GRPO reward signal for critic training.
Compatible with the VeRL batch reward manager interface.
"""

import ast
import concurrent.futures
import json
import re
import threading
import time
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import requests
from requests.adapters import HTTPAdapter


# ---------------------------------------------------------------------------
# TAIScore judge rubric
# ---------------------------------------------------------------------------

TAISCORE_RUBRIC = """
You are evaluating a writing critique as a training reward.

You will be given:
- User prompt
- Response A: initial response
- Critique of Response A
- Response B: revised response after the critique

Judge whether the critique was a useful intervention.

Score these dimensions from 1 to 10. All scores must be integers.
1. critique_quality: Is the critique faithful, specific, important, and actionable?
2. critique_uptake: Does Response B actually follow the critique?
3. quality_gain: Does Response B improve over Response A on the issue targeted by the critique?
4. prompt_faithfulness: Do the critique and Response B stay aligned with the user prompt?

Final score, also an integer from 1 to 10:
Give high scores only when the critique identifies important, fixable issues and Response B improves by following it.
Give low scores when the critique is vague, generic, unfaithful, or when Response B improves for reasons unrelated to the critique.
Penalize Response B if it becomes worse, overlong, irrelevant, or less faithful to the prompt.
Do not reward unnecessary length or superficial paraphrasing.

Return valid JSON only:
{
  "critique_quality": {"score": ..., "reason": "..."},
  "critique_uptake": {"score": ..., "reason": "..."},
  "quality_gain": {"score": ..., "reason": "..."},
  "prompt_faithfulness": {"score": ..., "reason": "..."},
  "score": ...
}
""".strip()


# ---------------------------------------------------------------------------
# Refiner prompt (generates y1 from x, y0, c)
# ---------------------------------------------------------------------------

REFINE_PROMPT = """
You are a helpful writing assistant.

Given a writing prompt, a previous response, and critique feedback, revise the response into a better answer.

Use the critique as guidance, but keep the final response natural, coherent, and faithful to the original prompt.
Return the full revised response, not only the changed parts.
Do not explain your changes.

### Prompt:
{prompt}

### Previous Response:
{response}

### Critique:
{critique}

### Revised Response:
""".strip()


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class OpenAICompatClient:
    """Thread-safe HTTP client for any OpenAI-compatible API endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        timeout_sec: int = 120,
        use_env_proxy: bool = False,
        pool_maxsize: int = 128,
    ):
        base = (base_url or "").rstrip("/")
        if base.endswith("/v1"):
            self.url = f"{base}/chat/completions"
        elif base.endswith("/chat/completions"):
            self.url = base
        else:
            self.url = f"{base}/v1/chat/completions"

        self.api_key = api_key or "EMPTY"
        self.timeout_sec = int(timeout_sec)
        self.use_env_proxy = bool(use_env_proxy)
        self.pool_maxsize = int(pool_maxsize)
        self._tls = threading.local()

    def _get_session(self) -> requests.Session:
        session = getattr(self._tls, "session", None)
        if session is None:
            session = requests.Session()
            session.trust_env = self.use_env_proxy
            adapter = HTTPAdapter(
                pool_connections=self.pool_maxsize,
                pool_maxsize=self.pool_maxsize,
            )
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._tls.session = session
        return session

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.2,
        top_p: float = 0.95,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        resp = self._get_session().post(
            self.url, headers=headers, json=payload, timeout=self.timeout_sec
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"API error status={resp.status_code} body={resp.text[:300]}"
            )
        return resp.json()["choices"][0]["message"]["content"]


@lru_cache(maxsize=8)
def _get_client(
    base_url: str,
    api_key: str,
    timeout_sec: int,
    use_env_proxy: bool = False,
    pool_maxsize: int = 128,
) -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url=base_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
        use_env_proxy=use_env_proxy,
        pool_maxsize=pool_maxsize,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def strip_think(text: str) -> str:
    """Remove <think>...</think> block produced by reasoning models."""
    if not isinstance(text, str):
        return ""
    marker = "</think>"
    lower = text.lower()
    if marker in lower:
        idx = lower.find(marker)
        text = text[idx + len(marker):]
    return text.strip()


def _try_parse_json(text: str) -> Optional[Dict]:
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    else:
        l, r = s.find("{"), s.rfind("}")
        if l != -1 and r > l:
            s = s[l: r + 1]
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def _parse_taiscore_response(text: str) -> Optional[float]:
    """Parse the final integer score (1–10) from the judge JSON response."""
    if not isinstance(text, str) or not text.strip():
        return None
    parsed = _try_parse_json(text)
    if isinstance(parsed, dict) and "score" in parsed:
        v = parsed["score"]
        if isinstance(v, bool) or v is None:
            return None
        try:
            f = float(v)
        except Exception:
            return None
        if not f.is_integer():
            return None
        score = int(f)
        if 1 <= score <= 10:
            return float(score)
    # Fallback: bare integer
    s = text.strip()
    if re.fullmatch(r"(?:10|[1-9])", s):
        return float(int(s))
    return None


def _zscore(values: Sequence[float], eps: float = 1e-6) -> List[float]:
    """Group-relative z-score normalization for GRPO advantages."""
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = var ** 0.5
    if std < eps:
        return [0.0] * len(values)
    return [(v - mean) / (std + eps) for v in values]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_judge_prompt(user_prompt: str, y0: str, critique: str, y1: str) -> str:
    return "\n\n".join([
        TAISCORE_RUBRIC,
        "---",
        f"User prompt:\n{user_prompt.strip()}",
        f"Response A:\n{y0.strip()}",
        f"Critique:\n{critique.strip()}",
        f"Response B:\n{y1.strip()}",
        "Return valid JSON only. Every score field must be an integer from 1 to 10.",
    ])


def _build_refine_prompt(user_prompt: str, y0: str, critique: str) -> str:
    return REFINE_PROMPT.format(prompt=user_prompt, response=y0, critique=critique)


# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------

def _call_api(
    client: OpenAICompatClient,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    retries: int,
    parser_fn: Optional[Callable[[str], Optional[float]]] = None,
) -> Tuple[Optional[str], Optional[float]]:
    """Single API call with retries. Returns (raw_text, parsed_score)."""
    messages = [{"role": "user", "content": prompt}]
    for _ in range(max(1, retries)):
        try:
            reply = client.chat(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception:
            continue
        if parser_fn is None:
            return reply, None
        score = parser_fn(reply)
        if score is not None:
            return reply, score
    return None, None


def compute_score_batched(
    data_sources: Sequence[str],
    solution_strs: Sequence[str],
    ground_truths: Sequence[Any],
    extra_infos: Sequence[Dict[str, Any]],
    # Actor (refiner) endpoint
    refiner_base_url: str = "http://localhost:8000",
    refiner_api_key: str = "EMPTY",
    refiner_model: str = "Qwen/Qwen3-8B",
    refiner_timeout_sec: int = 120,
    refiner_use_env_proxy: bool = False,
    refiner_max_tokens: int = 2048,
    refiner_temperature: float = 0.2,
    refiner_top_p: float = 0.95,
    refiner_retries: int = 2,
    refiner_max_parallel: int = 32,
    refiner_pool_maxsize: int = 128,
    # Judge endpoint
    judge_base_url: str = "http://localhost:8001",
    judge_api_key: str = "EMPTY",
    judge_model: str = "openai/gpt-oss-120b",
    judge_timeout_sec: int = 120,
    judge_use_env_proxy: bool = False,
    judge_max_tokens: int = 2048,
    judge_temperature: float = 0.2,
    judge_top_p: float = 1.0,
    judge_retries: int = 2,
    judge_max_parallel: int = 64,
    judge_pool_maxsize: int = 256,
    # Misc
    invalid_score: float = 0.0,
    debug: bool = False,
) -> List[Dict[str, float]]:
    """
    Compute TAIScore for a batch of critique rollouts.

    Each rollout is (x, y0, c) from extra_infos + solution_strs.
    The refiner generates y1; the judge scores (x, y0, c, y1).
    Returns group-relative z-scores as GRPO advantages.

    extra_infos fields:
      - user_prompt (str): the instruction x
      - draft_y0 (str): the initial response y0
      - example_id (str): used to group rollouts for z-score normalization
    solution_strs: the generated critique c
    """
    n = len(solution_strs)
    if n == 0:
        return []

    refiner_client = _get_client(
        refiner_base_url, refiner_api_key, int(refiner_timeout_sec),
        bool(refiner_use_env_proxy), max(32, int(refiner_pool_maxsize)),
    )
    judge_client = _get_client(
        judge_base_url, judge_api_key, int(judge_timeout_sec),
        bool(judge_use_env_proxy), max(32, int(judge_pool_maxsize)),
    )

    # Parse inputs
    user_prompts, y0s, critiques = [], [], []
    refine_prompts: List[Optional[str]] = []
    valid_indices: List[int] = []

    for i in range(n):
        extra = extra_infos[i] if i < len(extra_infos) and isinstance(extra_infos[i], dict) else {}
        x = str(extra.get("user_prompt", "")).strip()
        y0 = str(extra.get("draft_y0", "")).strip()
        c = strip_think(solution_strs[i] if i < len(solution_strs) else "")

        user_prompts.append(x)
        y0s.append(y0)
        critiques.append(c)

        if x and y0 and c:
            refine_prompts.append(_build_refine_prompt(x, y0, c))
            valid_indices.append(i)
        else:
            refine_prompts.append(None)

    # Group by example_id for z-score normalization
    group_ids: List[str] = []
    for i in range(n):
        extra = extra_infos[i] if i < len(extra_infos) and isinstance(extra_infos[i], dict) else {}
        gid = str(extra.get("example_id", "")).strip() or f"row_{i}"
        group_ids.append(gid)

    # Generate revisions (y1) and score in parallel
    raw_scores: List[Optional[float]] = [None] * n
    judge_valid: List[float] = [0.0] * n
    t0 = time.perf_counter()

    def _refine(idx: int) -> Tuple[int, Optional[str]]:
        prompt = refine_prompts[idx]
        if not prompt:
            return idx, None
        text, _ = _call_api(
            refiner_client, refiner_model, prompt,
            refiner_max_tokens, refiner_temperature, refiner_top_p, refiner_retries,
        )
        return idx, strip_think(text) if text else None

    def _judge(idx: int, y1: str) -> Tuple[int, Optional[float]]:
        prompt = _build_judge_prompt(user_prompts[idx], y0s[idx], critiques[idx], y1)
        _, score = _call_api(
            judge_client, judge_model, prompt,
            judge_max_tokens, judge_temperature, judge_top_p, judge_retries,
            parser_fn=_parse_taiscore_response,
        )
        return idx, score

    n_refiner = min(len(valid_indices), max(1, int(refiner_max_parallel)))
    n_judge = min(len(valid_indices), max(1, int(judge_max_parallel)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_refiner) as refiner_pool, \
         concurrent.futures.ThreadPoolExecutor(max_workers=n_judge) as judge_pool:
        refine_futures = {refiner_pool.submit(_refine, i): i for i in valid_indices}
        judge_futures: List[concurrent.futures.Future] = []

        for fut in concurrent.futures.as_completed(refine_futures):
            try:
                idx, y1 = fut.result()
            except Exception:
                continue
            if y1:
                judge_futures.append(judge_pool.submit(_judge, idx, y1))

        for fut in concurrent.futures.as_completed(judge_futures):
            try:
                idx, score = fut.result()
            except Exception:
                continue
            if score is not None:
                raw_scores[idx] = score
                judge_valid[idx] = 1.0

    if debug:
        n_valid = len(valid_indices)
        n_scored = sum(1 for s in raw_scores if s is not None)
        print(
            f"[taiscore] n={n} valid={n_valid} scored={n_scored} "
            f"elapsed={time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    # Group-relative z-score normalization
    group_to_indices: Dict[str, List[int]] = {}
    for i, gid in enumerate(group_ids):
        group_to_indices.setdefault(gid, []).append(i)

    advantage: List[float] = [0.0] * n
    for _, idxs in group_to_indices.items():
        group_raw = [raw_scores[j] if raw_scores[j] is not None else float(invalid_score) for j in idxs]
        group_z = _zscore(group_raw)
        for local_i, global_i in enumerate(idxs):
            advantage[global_i] = group_z[local_i]

    return [
        {
            "score": advantage[i],
            "raw_score": float(raw_scores[i]) if raw_scores[i] is not None else float(invalid_score),
            "judge_valid": judge_valid[i],
        }
        for i in range(n)
    ]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, float]:
    """Single-sample wrapper (VeRL naive reward manager interface)."""
    results = compute_score_batched(
        data_sources=[data_source],
        solution_strs=[solution_str],
        ground_truths=[ground_truth],
        extra_infos=[extra_info or {}],
        **kwargs,
    )
    if results:
        out = {k: float(v) for k, v in results[0].items()}
        # A single-sample call cannot form a GRPO group, so the normalized
        # advantage from compute_score_batched is always zero. Expose the raw
        # judge score as the scalar reward for naive single-sample callers.
        out["score"] = out.get("raw_score", out.get("score", 0.0))
        return out
    return {"score": 0.0, "raw_score": 0.0, "judge_valid": 0.0}
