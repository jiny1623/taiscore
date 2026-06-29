import json
import unittest
from unittest import mock

import taiscore.reward.taiscore as t


class TAIScoreTests(unittest.TestCase):
    def test_strip_think_removes_reasoning_prefix(self):
        self.assertEqual(t.strip_think("<think>hidden</think> visible"), "visible")

    def test_parse_json_score(self):
        reply = json.dumps({"score": 8})
        self.assertEqual(t._parse_taiscore_response(reply), 8.0)

    def test_parse_fenced_json_score(self):
        reply = '```json\n{"score": 7}\n```'
        self.assertEqual(t._parse_taiscore_response(reply), 7.0)

    def test_rejects_non_integer_score(self):
        self.assertIsNone(t._parse_taiscore_response('{"score": 7.5}'))

    def test_batch_returns_group_normalized_scores(self):
        def fake_call(client, model, prompt, max_tokens, temperature, top_p, retries, parser_fn=None):
            if parser_fn is None:
                return "revised", None
            score = 9 if "HIGH_MARKER" in prompt else 3
            reply = json.dumps({"score": score})
            return reply, parser_fn(reply)

        with mock.patch.object(t, "_call_api", side_effect=fake_call):
            scores = t.compute_score_batched(
                data_sources=["writing", "writing"],
                solution_strs=["LOW_MARKER critique", "HIGH_MARKER critique"],
                ground_truths=[None, None],
                extra_infos=[
                    {"user_prompt": "prompt", "draft_y0": "draft", "example_id": "same"},
                    {"user_prompt": "prompt", "draft_y0": "draft", "example_id": "same"},
                ],
            )

        self.assertEqual([s["raw_score"] for s in scores], [3.0, 9.0])
        self.assertLess(scores[0]["score"], 0)
        self.assertGreater(scores[1]["score"], 0)

    def test_single_score_uses_raw_score(self):
        def fake_call(client, model, prompt, max_tokens, temperature, top_p, retries, parser_fn=None):
            if parser_fn is None:
                return "revised", None
            reply = json.dumps({"score": 6})
            return reply, parser_fn(reply)

        with mock.patch.object(t, "_call_api", side_effect=fake_call):
            score = t.compute_score(
                data_source="writing",
                solution_str="critique",
                ground_truth=None,
                extra_info={"user_prompt": "prompt", "draft_y0": "draft", "example_id": "one"},
            )

        self.assertEqual(score["score"], 6.0)
        self.assertEqual(score["raw_score"], 6.0)


if __name__ == "__main__":
    unittest.main()
