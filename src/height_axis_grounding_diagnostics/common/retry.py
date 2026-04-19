from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RETRYABLE_STATUS_CODES = (408, 409, 429, 500, 502, 503, 504, 529)
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_INITIAL_DELAY_SECONDS = 2.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0
DEFAULT_MAX_DELAY_SECONDS = 60.0
DEFAULT_JITTER_RATIO = 0.2

TRANSIENT_ERROR_MARKERS = (
    "api connection error",
    "apiconnectionerror",
    "api timeout error",
    "apitimeouterror",
    "bad gateway",
    "badgateway",
    "connection refused",
    "connection reset",
    "deadline exceeded",
    "deadline_exceeded",
    "gateway timeout",
    "internal server error",
    "internal_server_error",
    "overloaded",
    "overloaded_error",
    "rate limit",
    "ratelimiterror",
    "server_error",
    "service unavailable",
    "serviceunavailable",
    "temporarily unavailable",
    "timeout",
    "timed out",
    "too many requests",
    "unavailable",
)


class ApiErrorGuardError(Exception):
    pass


@dataclass(frozen=True)
class RetryPolicy:
    enabled: bool = True
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    initial_delay_seconds: float = DEFAULT_INITIAL_DELAY_SECONDS
    backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER
    max_delay_seconds: float = DEFAULT_MAX_DELAY_SECONDS
    jitter_ratio: float = DEFAULT_JITTER_RATIO
    retryable_status_codes: tuple[int, ...] = DEFAULT_RETRYABLE_STATUS_CODES

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_attempts": self.max_attempts,
            "initial_delay_seconds": self.initial_delay_seconds,
            "backoff_multiplier": self.backoff_multiplier,
            "max_delay_seconds": self.max_delay_seconds,
            "jitter_ratio": self.jitter_ratio,
            "retryable_status_codes": list(self.retryable_status_codes),
        }


@dataclass(frozen=True)
class RetryDecision:
    retryable: bool
    reason: str | None = None


def parse_retry_status_codes(value: str | None) -> tuple[int, ...]:
    if value is None or not value.strip():
        return DEFAULT_RETRYABLE_STATUS_CODES
    status_codes: list[int] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            status_code = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid retry status code: {item}") from exc
        if status_code < 100 or status_code > 599:
            raise ValueError(f"Retry status code must be 100-599: {status_code}")
        status_codes.append(status_code)
    if not status_codes:
        raise ValueError("At least one retry status code is required")
    return tuple(sorted(set(status_codes)))


def validate_retry_policy(policy: RetryPolicy) -> None:
    if policy.max_attempts < 1:
        raise ValueError("--retry-max-attempts must be at least 1")
    if policy.initial_delay_seconds < 0:
        raise ValueError("--retry-initial-delay-seconds must be non-negative")
    if policy.backoff_multiplier < 1:
        raise ValueError("--retry-backoff-multiplier must be at least 1")
    if policy.max_delay_seconds < 0:
        raise ValueError("--retry-max-delay-seconds must be non-negative")
    if policy.jitter_ratio < 0:
        raise ValueError("--retry-jitter-ratio must be non-negative")


def add_retry_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--disable-retry", action="store_true", help="Disable provider API retry attempts.")
    parser.add_argument(
        "--retry-max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Total attempts for retryable provider API errors. Default: {DEFAULT_MAX_ATTEMPTS}.",
    )
    parser.add_argument(
        "--retry-initial-delay-seconds",
        type=float,
        default=DEFAULT_INITIAL_DELAY_SECONDS,
        help=f"Delay after the first retryable failure. Default: {DEFAULT_INITIAL_DELAY_SECONDS}.",
    )
    parser.add_argument(
        "--retry-backoff-multiplier",
        type=float,
        default=DEFAULT_BACKOFF_MULTIPLIER,
        help=f"Exponential backoff multiplier. Default: {DEFAULT_BACKOFF_MULTIPLIER}.",
    )
    parser.add_argument(
        "--retry-max-delay-seconds",
        type=float,
        default=DEFAULT_MAX_DELAY_SECONDS,
        help=f"Maximum retry delay. Default: {DEFAULT_MAX_DELAY_SECONDS}.",
    )
    parser.add_argument(
        "--retry-jitter-ratio",
        type=float,
        default=DEFAULT_JITTER_RATIO,
        help=f"Random jitter ratio applied to retry delay. Default: {DEFAULT_JITTER_RATIO}.",
    )
    parser.add_argument(
        "--retry-status-codes",
        default=",".join(str(code) for code in DEFAULT_RETRYABLE_STATUS_CODES),
        help="Comma-separated HTTP status codes treated as retryable provider API errors.",
    )
    parser.add_argument(
        "--allow-api-errors",
        action="store_true",
        help="Allow a run or analysis to finish even if retryable provider API errors remain.",
    )


def build_retry_policy_from_args(args: argparse.Namespace) -> RetryPolicy:
    policy = RetryPolicy(
        enabled=not bool(getattr(args, "disable_retry", False)),
        max_attempts=args.retry_max_attempts,
        initial_delay_seconds=args.retry_initial_delay_seconds,
        backoff_multiplier=args.retry_backoff_multiplier,
        max_delay_seconds=args.retry_max_delay_seconds,
        jitter_ratio=args.retry_jitter_ratio,
        retryable_status_codes=parse_retry_status_codes(args.retry_status_codes),
    )
    validate_retry_policy(policy)
    return policy


def extract_status_code_from_error(error_text: str) -> int | None:
    for match in re.finditer(r"(?<![\d.])([1-5]\d{2})(?![\d.])", error_text):
        return int(match.group(1))
    return None


def normalize_status_code(status_code: Any, error_text: str) -> int | None:
    if isinstance(status_code, int):
        return status_code
    return extract_status_code_from_error(error_text)


def classify_api_error(
    *,
    status_code: int | None,
    error_text: str,
    policy: RetryPolicy,
) -> RetryDecision:
    if status_code in policy.retryable_status_codes:
        return RetryDecision(True, f"retryable_status_code:{status_code}")

    lowered = error_text.lower()
    for marker in TRANSIENT_ERROR_MARKERS:
        if marker in lowered:
            return RetryDecision(True, f"transient_error_marker:{marker}")

    return RetryDecision(False, None)


def is_retryable_api_error(
    *,
    status_code: int | None,
    error_text: str,
    policy: RetryPolicy,
) -> bool:
    return classify_api_error(status_code=status_code, error_text=error_text, policy=policy).retryable


def calculate_retry_delay(attempt_number: int, policy: RetryPolicy) -> float:
    exponent = max(0, attempt_number - 1)
    delay = policy.initial_delay_seconds * (policy.backoff_multiplier ** exponent)
    delay = min(delay, policy.max_delay_seconds)
    if policy.jitter_ratio > 0 and delay > 0:
        jitter = delay * policy.jitter_ratio
        delay = random.uniform(max(0.0, delay - jitter), delay + jitter)
        delay = min(delay, policy.max_delay_seconds)
    return round(delay, 3)


def build_api_error_rows(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        retry_attempts = attempt.get("retry_attempts")
        if not isinstance(retry_attempts, list) or not retry_attempts:
            continue
        final_retry_attempt = retry_attempts[-1]
        retry_summary = attempt.get("retry_summary") or {}
        if final_retry_attempt.get("error_type") != "api_error":
            continue
        if retry_summary.get("final_outcome") not in {"failed_after_retries", "failed_without_retry"}:
            continue

        rows.append(
            {
                "attempt_id": attempt.get("attempt_id", ""),
                "provider": attempt.get("provider", ""),
                "model": attempt.get("model", ""),
                "case_id": attempt.get("case_id", ""),
                "image_id": attempt.get("image_id", ""),
                "answer_key": attempt.get("answer_key", ""),
                "package_slug": attempt.get("package_slug", ""),
                "variant_slug": attempt.get("variant_slug", ""),
                "prompt_name": attempt.get("prompt_name", ""),
                "repeat_index": attempt.get("repeat_index", ""),
                "status_code": final_retry_attempt.get("status_code"),
                "retry_count": attempt.get("retry_count", 0),
                "attempt_count": retry_summary.get("attempt_count", len(retry_attempts)),
                "final_outcome": retry_summary.get("final_outcome", ""),
                "error_message": final_retry_attempt.get("error_message") or "",
            }
        )
    return rows


def assert_no_api_errors(api_error_rows: list[dict[str, Any]], *, allow_api_errors: bool, run_dir: Path) -> None:
    if allow_api_errors or not api_error_rows:
        return
    raise ApiErrorGuardError(
        f"Run completed with {len(api_error_rows)} provider API error(s) after retries. "
        f"See {run_dir / 'api_errors.csv'}."
    )


def get_run_api_error_count(run_dir: Path) -> int:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = None
        if isinstance(summary, dict) and isinstance(summary.get("api_error_count"), int):
            return summary["api_error_count"]

    api_errors_path = run_dir / "api_errors.csv"
    if api_errors_path.exists():
        with api_errors_path.open("r", encoding="utf-8", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))

    attempts_dir = run_dir / "attempts"
    if not attempts_dir.exists():
        return 0
    attempts: list[dict[str, Any]] = []
    for attempt_path in attempts_dir.rglob("run-*.json"):
        try:
            attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(attempt, dict):
            attempts.append(attempt)
    return len(build_api_error_rows(attempts))


def raise_if_run_has_api_errors(run_dir: Path, *, allow_api_errors: bool) -> None:
    if allow_api_errors:
        return
    api_error_count = get_run_api_error_count(run_dir)
    if api_error_count:
        raise ApiErrorGuardError(
            f"Run has {api_error_count} provider API error(s) after retries. "
            "Re-run those cases or pass --allow-api-errors to analyze anyway."
        )
