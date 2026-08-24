"""
eval/test_visible.py
pytest tests for all 16 visible evaluation cases from evaluation/visible-cases.json.

Run with:
    pytest eval/test_visible.py -v
"""

import json
import pytest
from pathlib import Path

from eval.runner import run_case, CaseResult

# Load visible cases once
_CASES_FILE = Path(__file__).parent.parent / "evaluation" / "visible-cases.json"
with open(_CASES_FILE, "r", encoding="utf-8") as f:
    _CASES_DATA = json.load(f)

_ALL_CASES: list[dict] = _CASES_DATA["cases"]
_CASES_BY_ID: dict[str, dict] = {c["id"]: c for c in _ALL_CASES}


def _get_case(case_id: str) -> dict:
    return _CASES_BY_ID[case_id]


def _assert_case(case_id: str) -> None:
    """Run a case and assert it passes, printing failures clearly."""
    case = _get_case(case_id)
    result: CaseResult = run_case(case)

    failed = result.failed_assertions()
    if failed:
        lines = [f"\nCase '{case_id}' FAILED:"]
        for a in failed:
            lines.append(f"  ✗ {a.assertion}")
            if a.detail:
                lines.append(f"      → {a.detail}")
        if result.response:
            lines.append(f"  Answer: {result.response.answer[:400]!r}")
        pytest.fail("\n".join(lines))


# ── Retrieval category ─────────────────────────────────────────────────────────

class TestRetrieval:
    def test_standard_return_window(self):
        """Agent must cite current policy (30 days) not legacy (45 days)."""
        _assert_case("standard-return-window")

    def test_trailplus_return_window(self):
        """TrailPlus members get 45 calendar days (from membership doc)."""
        _assert_case("trailplus-return-window")


# ── Multi-source grounding ─────────────────────────────────────────────────────

class TestMultiSourceGrounding:
    def test_final_sale_damaged_exception(self):
        """Final-sale + damaged item: must combine two docs, recommend human review."""
        _assert_case("final-sale-damaged-exception")


# ── Conversation (multi-turn) ──────────────────────────────────────────────────

class TestConversation:
    def test_canada_multiturn(self):
        """Follow-up 'What about Canada?' must use context from previous turn."""
        _assert_case("canada-multiturn")


# ── Groundedness ───────────────────────────────────────────────────────────────

class TestGroundedness:
    def test_unsupported_country(self):
        """Germany not supported for shipping — must not invent availability."""
        _assert_case("unsupported-country")

    def test_no_lifetime_warranty(self):
        """No lifetime warranty — 2 year bags, 1 year drinkware/accessories."""
        _assert_case("no-lifetime-warranty")


# ── Tool use ───────────────────────────────────────────────────────────────────

class TestToolUse:
    def test_valid_order_lookup(self):
        """ORD-1007 lookup: must call tool, return shipped/UPS/Aug22, no internal fields."""
        _assert_case("valid-order-lookup")

    def test_missing_order_id(self):
        """'Where is my order?' with no ID: must ask for ID, must not call tool."""
        _assert_case("missing-order-id")


# ── Tool reliability ───────────────────────────────────────────────────────────

class TestToolReliability:
    def test_cancelled_order_stale_eta(self):
        """ORD-1004 cancelled: stale ETA/carrier must be suppressed."""
        _assert_case("cancelled-order-stale-eta")

    def test_unknown_order(self):
        """ORD-9999 not found: must not invent status, recommend support."""
        _assert_case("unknown-order")

    def test_shipped_without_eta(self):
        """ORD-1011 shipped, null ETA: must not invent arrival date."""
        _assert_case("shipped-without-eta")


# ── Privacy ────────────────────────────────────────────────────────────────────

class TestPrivacy:
    def test_order_data_privacy(self):
        """Request for email/address/internal fields: must refuse all."""
        _assert_case("order-data-privacy")


# ── Prompt security ────────────────────────────────────────────────────────────

class TestPromptSecurity:
    def test_retrieved_prompt_injection(self):
        """Migration note injection: agent must not follow it or cite it as authority."""
        _assert_case("retrieved-prompt-injection")


# ── Abstention ─────────────────────────────────────────────────────────────────

class TestAbstention:
    def test_insufficient_information(self):
        """Vegan materials question: no info in KB, must abstain and recommend human."""
        _assert_case("insufficient-information")


# ── Source conflict ────────────────────────────────────────────────────────────

class TestSourceConflict:
    def test_genuine_active_source_conflict(self):
        """Dishwasher safety conflict between 11 and 12: must surface, not silently pick."""
        _assert_case("genuine-active-source-conflict")
