#!/usr/bin/env python3
"""Score one critique-guided transition with OpenAI-compatible endpoints."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taiscore.reward import compute_score


PROMPT = "Write a concise opening paragraph for a mystery story set in a library."
DRAFT_Y0 = "The library was quiet. Someone walked in and looked around."
CRITIQUE = (
    "The response is too generic. Add a concrete sensory detail and a clearer "
    "hint of mystery while keeping the paragraph concise."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refiner-base-url", default="http://localhost:8000")
    parser.add_argument("--judge-base-url", default="http://localhost:8001")
    parser.add_argument("--refiner-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--judge-model", default="openai/gpt-oss-120b")
    args = parser.parse_args()

    score = compute_score(
        data_source="writing",
        solution_str=CRITIQUE,
        ground_truth=None,
        extra_info={
            "user_prompt": PROMPT,
            "draft_y0": DRAFT_Y0,
            "example_id": "example-0001",
        },
        refiner_base_url=args.refiner_base_url,
        refiner_model=args.refiner_model,
        judge_base_url=args.judge_base_url,
        judge_model=args.judge_model,
    )
    print(json.dumps(score, indent=2))


if __name__ == "__main__":
    main()
