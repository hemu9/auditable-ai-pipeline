"""Claude adapter: request shape, response interpretation, retries and SDK wiring, all offline."""

import importlib.util
import json
import unittest
from unittest import mock

from auditable_ai.anthropic_adapter import (
    AnthropicTransport,
    ClaudeExtractionAdapter,
    ClaudeSettings,
    ModelOutputError,
    ProviderCallError,
    build_request,
    interpret_response,
    parse_output,
)
from auditable_ai.dataset import load_dataset
from auditable_ai.documents import RawDocumentContent
from auditable_ai.runs import run_model_extraction
from tests.provider_fixtures import (
    ScriptedTransport,
    call_center_output,
    fixed_clock,
    message_body,
    response,
    transient_error,
)

HAS_SDK = importlib.util.find_spec("anthropic") is not None


def call_center():
    return load_dataset().document("eval-call-center-v1")


class RequestConstruction(unittest.TestCase):
    def test_request_uses_structured_output_and_explicit_effort_without_sampling_parameters(self):
        request = build_request(call_center(), ClaudeSettings())
        self.assertEqual(request["model"], "claude-opus-5-5")
        self.assertEqual(request["output_config"]["effort"], "medium")
        self.assertEqual(request["output_config"]["format"]["type"], "json_schema")
        self.assertTrue({"temperature", "top_p", "top_k", "thinking"}.isdisjoint(request))
        self.assertEqual(request["betas"], ["server-side-fallback-2026-07-01"])
        self.assertEqual(request["fallbacks"], "default")
        self.assertIn("[eval-call-center-v1:s3]", request["messages"][0]["content"])

    def test_fallbacks_can_be_disabled_to_keep_every_response_on_the_requested_model(self):
        request = build_request(call_center(), ClaudeSettings(fallbacks=None))
        self.assertTrue({"betas", "fallbacks"}.isdisjoint(request))

    def test_settings_are_checked_when_created(self):
        for field, value in (
            ("model", ""),
            ("effort", None),
            ("max_tokens", 0),
            ("max_attempts", True),
            ("timeout_s", float("nan")),
            ("fallbacks", 3),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                ClaudeSettings(**{field: value})


class ResponseInterpretation(unittest.TestCase):
    def test_output_usage_and_fallback_switches_are_read_without_the_sdk(self):
        switch = {"type": "fallback", "from": {"model": "claude-opus-5-5"}, "to": {"model": "x"}}
        interpretation = interpret_response(message_body({"facts": []}, blocks=[switch], model="x"))
        self.assertEqual(interpretation.served_model, "x")
        self.assertEqual(interpretation.fallbacks, ({"from": "claude-opus-5-5", "to": "x"},))
        self.assertEqual(interpretation.usage["input_tokens"], 900)
        self.assertEqual(parse_output(interpretation), {"facts": []})

    def test_unusable_responses_are_classified_by_kind(self):
        cases = {
            "refused": message_body(stop_reason="refusal"),
            "truncated": message_body(text='{"facts": [', stop_reason="max_tokens"),
            "no_text": message_body(),
            "invalid_json": message_body(text="The report mentions 9,320 calls."),
        }
        for kind, body in cases.items():
            with self.subTest(kind=kind):
                with self.assertRaises(ModelOutputError) as caught:
                    parse_output(interpret_response(body))
                self.assertEqual(caught.exception.kind, kind)
        for body in ("not json", json.dumps({"type": "error"})):
            with self.subTest(body=body):
                with self.assertRaises(ModelOutputError) as caught:
                    interpret_response(body)
                self.assertEqual(caught.exception.kind, "unexpected_response")

    def test_metadata_of_an_unexpected_type_reads_as_absent(self):
        body = json.dumps(
            {
                "type": "message",
                "id": 7,
                "model": ["claude-opus-5-5"],
                "stop_reason": {"reason": "end_turn"},
                "content": [
                    {"type": "fallback", "from": 1, "to": {"model": 2}},
                    {"type": "text", "text": "{}"},
                ],
                "usage": "plenty",
            }
        )
        interpretation = interpret_response(body)
        self.assertEqual(
            (
                interpretation.message_id,
                interpretation.served_model,
                interpretation.stop_reason,
                interpretation.usage,
            ),
            (None, None, None, None),
        )
        self.assertEqual(interpretation.fallbacks, ({"from": None, "to": None},))
        self.assertEqual(parse_output(interpretation), {})

    def test_non_standard_or_unparseable_json_is_classified_not_raised(self):
        body = message_body({"facts": []}).replace('"output_tokens": 300', '"output_tokens": NaN')
        for unparseable in (body, "[" * 100_000):
            with self.subTest(body=unparseable[:20]), self.assertRaises(ModelOutputError) as caught:
                interpret_response(unparseable)
            self.assertEqual(caught.exception.kind, "unexpected_response")
        for text in ('{"confidence": NaN}', '{"confidence": -Infinity}', "[" * 100_000):
            with self.subTest(text=text[:20]), self.assertRaises(ModelOutputError) as caught:
                parse_output(interpret_response(message_body(text=text)))
            self.assertEqual(caught.exception.kind, "invalid_json")

    def test_input_too_deeply_nested_to_parse_is_classified_not_raised(self):
        # How deep the JSON parser can go depends on the Python version, so this
        # makes it fail the way it does on deeply nested input.
        interpretation = interpret_response(message_body(text="[]"))
        too_deep = RecursionError("maximum recursion depth exceeded while decoding")
        with mock.patch("auditable_ai.anthropic_adapter.strict_loads", side_effect=too_deep):
            for kind, parse in (
                ("unexpected_response", lambda: interpret_response(message_body(text="[]"))),
                ("invalid_json", lambda: parse_output(interpretation)),
            ):
                with self.subTest(kind=kind), self.assertRaises(ModelOutputError) as caught:
                    parse()
                self.assertEqual(caught.exception.kind, kind)


class RetryPolicy(unittest.TestCase):
    def setUp(self):
        self.document = call_center()
        self.content = RawDocumentContent(self.document.text)

    def test_transient_errors_are_retried_and_only_the_final_attempt_is_timed(self):
        delays = []
        transport = ScriptedTransport(
            transient_error(), response(message_body(call_center_output()))
        )
        adapter = ClaudeExtractionAdapter(
            self.document, transport=transport, sleep=delays.append, jitter=lambda: 0.25
        )
        self.assertEqual(adapter(self.content)["source_ids"], ["eval-call-center-v1"])
        attempts = adapter.exchange.attempts
        self.assertEqual([attempt["outcome"] for attempt in attempts], ["error", "response"])
        self.assertEqual(adapter.exchange.latency_ms, attempts[-1]["latency_ms"])
        self.assertEqual((delays, len(transport.requests)), ([1.25], 2))

    def test_retry_after_is_respected_and_attempts_are_capped(self):
        limited = ProviderCallError(
            "rate limited",
            error_type="RateLimitError",
            status_code=429,
            retryable=True,
            retry_after_s=7,
        )
        delays = []
        adapter = ClaudeExtractionAdapter(
            self.document,
            transport=ScriptedTransport(limited, limited, limited),
            sleep=delays.append,
            jitter=lambda: 0.0,
        )
        with self.assertRaises(ProviderCallError):
            adapter(self.content)
        self.assertEqual((delays, len(adapter.exchange.attempts)), ([7, 7], 3))
        self.assertIsNone(adapter.exchange.raw_response)
        self.assertEqual(adapter.exchange.error["error_type"], "RateLimitError")

    def test_a_non_retryable_error_is_not_retried(self):
        rejected = ProviderCallError("bad request", error_type="BadRequestError", status_code=400)
        transport = ScriptedTransport(rejected)
        adapter = ClaudeExtractionAdapter(self.document, transport=transport, sleep=self.fail)
        with self.assertRaises(ProviderCallError):
            adapter(self.content)
        self.assertEqual((len(transport.requests), adapter.exchange.error["status_code"]), (1, 400))

    def test_credentials_are_redacted_from_recorded_messages(self):
        error = ProviderCallError(
            "invalid key sk-ant-api03-ABCDEFGHIJKL; header Bearer abcdefghijkl",
            error_type="AuthenticationError",
            status_code=401,
        )
        self.assertEqual(error.describe()["message"], "invalid key [REDACTED]; header [REDACTED]")

    def test_the_adapter_refuses_text_from_a_different_document(self):
        adapter = ClaudeExtractionAdapter(self.document, transport=ScriptedTransport())
        with self.assertRaises(ValueError):
            adapter(RawDocumentContent("another document"))


@unittest.skipUnless(HAS_SDK, "the optional anthropic SDK is not installed")
class SdkTransport(unittest.TestCase):
    """The real SDK over a mocked HTTP layer: request shape and error mapping, no network."""

    def transport(self, handler, *, api_key="sk-ant-scripted-test-key-0000"):
        import anthropic
        import httpx2

        client = anthropic.Anthropic(
            api_key=api_key,
            max_retries=0,
            http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        )
        return AnthropicTransport(client, origin="scripted_fixture")

    def test_the_documented_request_is_sent_and_the_exact_body_is_returned(self):
        import httpx2

        seen = {}
        body = message_body(call_center_output())

        def handler(request):
            seen.update(
                path=request.url.path,
                beta=request.headers.get("anthropic-beta"),
                payload=json.loads(request.content),
            )
            return httpx2.Response(200, headers={"request-id": "req_mock_1"}, text=body)

        raw = self.transport(handler).send(build_request(call_center(), ClaudeSettings()))
        self.assertEqual((raw.text, raw.request_id, raw.status_code), (body, "req_mock_1", 200))
        self.assertEqual(seen["path"], "/v1/messages")
        self.assertEqual(seen["beta"], "server-side-fallback-2026-07-01")
        self.assertEqual(seen["payload"]["fallbacks"], "default")
        self.assertEqual(seen["payload"]["output_config"]["format"]["type"], "json_schema")
        self.assertTrue({"temperature", "betas"}.isdisjoint(seen["payload"]))

    def test_requests_without_fallbacks_use_the_standard_endpoint(self):
        import httpx2

        seen = {}

        def handler(request):
            seen["beta"] = request.headers.get("anthropic-beta")
            return httpx2.Response(200, headers={"request-id": "req_mock_2"}, text=message_body({}))

        self.transport(handler).send(build_request(call_center(), ClaudeSettings(fallbacks=None)))
        self.assertIsNone(seen["beta"])

    def test_api_errors_become_classified_provider_errors(self):
        import httpx2

        cases = (
            (429, {"retry-after": "3"}, "RateLimitError", True, 3.0),
            (529, {}, "OverloadedError", True, None),
            (400, {}, "BadRequestError", False, None),
        )
        for status, headers, error_type, retryable, retry_after in cases:

            def handler(request, status=status, headers=headers):
                return httpx2.Response(
                    status,
                    headers={"request-id": f"req_{status}", **headers},
                    json={"type": "error", "error": {"type": "scripted", "message": "scripted"}},
                )

            with self.subTest(status=status):
                with self.assertRaises(ProviderCallError) as caught:
                    self.transport(handler).send(build_request(call_center(), ClaudeSettings()))
                error = caught.exception
                self.assertEqual(
                    (error.error_type, error.status_code, error.retryable, error.retry_after_s),
                    (error_type, status, retryable, retry_after),
                )
                self.assertEqual(error.request_id, f"req_{status}")

    def test_connection_failures_are_retryable(self):
        import httpx2

        def handler(request):
            raise httpx2.ConnectError("offline")

        with self.assertRaises(ProviderCallError) as caught:
            self.transport(handler).send(build_request(call_center(), ClaudeSettings()))
        self.assertEqual(caught.exception.error_type, "APIConnectionError")
        self.assertTrue(caught.exception.retryable)

    def test_a_full_run_through_the_sdk_never_records_the_api_key(self):
        import httpx2

        key = "sk-ant-scripted-secret-0123456789"

        def handler(request):
            body = message_body(call_center_output())
            return httpx2.Response(200, headers={"request-id": "req_mock_3"}, text=body)

        record = run_model_extraction(
            call_center(), transport=self.transport(handler, api_key=key), clock=fixed_clock
        )
        serialized = json.dumps(record)
        self.assertEqual(record["status"], "completed")
        self.assertNotIn(key, serialized)
        self.assertNotIn("authorization", serialized.lower())
