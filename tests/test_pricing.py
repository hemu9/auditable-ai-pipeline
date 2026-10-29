"""Cost estimates come from API-reported tokens and published prices, or are unavailable."""

import unittest

from auditable_ai.pricing import PRICE_SOURCE, PRICES_RETRIEVED_ON, estimate_cost


class CostEstimates(unittest.TestCase):
    def test_reported_tokens_are_priced_at_the_published_rate(self):
        usage = {"input_tokens": 900, "output_tokens": 300, "cache_read_input_tokens": 0}
        cost = estimate_cost(usage, "claude-opus-5-5", stop_reason="end_turn")
        self.assertEqual((cost["status"], cost["usd"]), ("estimated", 0.0096))
        self.assertEqual(
            (cost["basis"]["source"], cost["basis"]["retrieved_on"]),
            (PRICE_SOURCE, PRICES_RETRIEVED_ON),
        )

    def test_each_attempt_is_priced_at_the_model_that_ran_it(self):
        usage = {
            "input_tokens": 1000,
            "output_tokens": 200,
            "iterations": [
                {
                    "type": "message",
                    "model": "claude-opus-5-5",
                    "input_tokens": 1000,
                    "output_tokens": 0,
                },
                {
                    "type": "fallback_message",
                    "model": "claude-opus-4-8",
                    "input_tokens": 1000,
                    "output_tokens": 200,
                },
            ],
        }
        cost = estimate_cost(usage, "claude-opus-4-8", stop_reason="end_turn")
        self.assertEqual(cost["usd"], 0.014)
        self.assertEqual([line["usd"] for line in cost["attempts"]], [0.004, 0.01])

    def test_cache_writes_are_priced_by_duration(self):
        usage = {
            "input_tokens": 100,
            "output_tokens": 0,
            "cache_read_input_tokens": 2000,
            "cache_creation_input_tokens": 1000,
            "cache_creation": {"ephemeral_5m_input_tokens": 600, "ephemeral_1h_input_tokens": 400},
        }
        self.assertEqual(
            estimate_cost(usage, "claude-opus-5-5", stop_reason="end_turn")["usd"], 0.007
        )

    def test_unmodeled_cases_are_reported_as_unavailable_rather_than_guessed(self):
        tokens = {"input_tokens": 10, "output_tokens": 10}
        cases = (
            (None, "claude-opus-5-5", "end_turn", "no usage"),
            (tokens, "claude-not-in-the-table", "end_turn", "no recorded list price"),
            (tokens, "claude-opus-5-5", "refusal", "refusal"),
            ({**tokens, "speed": "fast"}, "claude-opus-5-5", "end_turn", "speed"),
            ({**tokens, "inference_geo": "us"}, "claude-opus-5-5", "end_turn", "geography"),
            ({**tokens, "service_tier": "priority"}, "claude-opus-5-5", "end_turn", "tier"),
            (
                {**tokens, "cache_creation_input_tokens": 50},
                "claude-opus-5-5",
                "end_turn",
                "duration",
            ),
        )
        for usage, model, stop_reason, phrase in cases:
            with self.subTest(reason=phrase):
                cost = estimate_cost(usage, model, stop_reason=stop_reason)
                self.assertEqual((cost["status"], cost["usd"]), ("unavailable", None))
                self.assertIn(phrase, cost["reason"])

    def test_malformed_usage_makes_the_estimate_unavailable_instead_of_raising(self):
        cases = {
            "a list as a count": {"input_tokens": [None], "output_tokens": 0},
            "a negative count": {"input_tokens": -1},
            "a boolean count": {"output_tokens": True},
            "iterations that are not a list": {"input_tokens": 1, "iterations": {"model": "x"}},
            "an iteration that is not an object": {"input_tokens": 1, "iterations": [7]},
            "an unhashable model": {"iterations": [{"model": ["claude-opus-5-5"]}]},
            "cache durations that are not an object": {
                "cache_creation_input_tokens": 5,
                "cache_creation": "5m",
            },
            "a count too large to convert": {"input_tokens": 10**400},
            "a total too large to price": {"input_tokens": 10**308},
        }
        for name, usage in cases.items():
            with self.subTest(name):
                cost = estimate_cost(usage, "claude-opus-5-5", stop_reason="end_turn")
                self.assertEqual((cost["status"], cost["usd"]), ("unavailable", None))
