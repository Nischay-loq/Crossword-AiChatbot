"""
agent/order_tool.py
Order status lookup tool.

Key responsibilities:
  1. Normalize order IDs (case, whitespace).
  2. Look up the order from orders.json (loaded once at import).
  3. Sanitize the result — strip ALL internal/PII fields before returning to the LLM.
  4. Apply status-precedence rules (cancelled/returned → ignore stale ETAs, etc.).
  5. Defend against prompt injection in warehouse notes (they never reach the LLM).

The LLM never sees the raw orders.json. It only receives the sanitized result dict
returned by `lookup_order()`.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from agent.config import ORDERS_FILE, DEBUG

logger = logging.getLogger(__name__)

# ── Load orders once at module import ────────────────────────────────────────
_ORDERS_DATA: dict[str, Any] = {}
_ORDERS_MAP: dict[str, dict] = {}  # keyed by normalized order_id

def _load_orders() -> None:
    global _ORDERS_DATA, _ORDERS_MAP
    if not ORDERS_FILE.exists():
        raise FileNotFoundError(f"Orders file not found: {ORDERS_FILE}")
    with open(ORDERS_FILE, "r", encoding="utf-8") as f:
        _ORDERS_DATA = json.load(f)
    for order in _ORDERS_DATA.get("orders", []):
        key = order["order_id"].upper().strip()
        _ORDERS_MAP[key] = order
    logger.info(f"Loaded {len(_ORDERS_MAP)} orders from {ORDERS_FILE}")

_load_orders()

# ── Snapshot timestamp (used as "current time" in evaluations) ───────────────
SNAPSHOT_AT: str = _ORDERS_DATA.get("snapshot_at", "")

# ── Customer-safe field allow-list ────────────────────────────────────────────
# ONLY these top-level fields may be returned to the LLM.
SAFE_FIELDS = {
    "order_id",
    "membership_tier",
    "items",           # filtered sub-fields below
    "placed_at",
    "status",
    "status_updated_at",
    "shipped_at",
    "delivered_at",
    "carrier",
    "tracking_number",
    "estimated_delivery",
    "customer_safe_message",
}

# Safe item sub-fields
SAFE_ITEM_FIELDS = {"name", "quantity", "final_sale"}

# Statuses where carrier/tracking/ETA fields are considered stale and must be suppressed
TERMINAL_STATUSES = {"cancelled", "returned"}


def _normalize_order_id(raw: str) -> str:
    """
    Normalize order ID input: strip whitespace, uppercase, allow ORD-XXXX pattern.
    Does NOT guess a different ID if the format doesn't match.
    """
    normalized = raw.strip().upper()
    # Remove surrounding punctuation that isn't part of the ID
    normalized = re.sub(r"^['\"\s]+|['\"\s.!?]+$", "", normalized)
    return normalized


def _sanitize_items(items: list[dict]) -> list[dict]:
    """Strip any non-safe sub-fields from items list."""
    return [
        {k: v for k, v in item.items() if k in SAFE_ITEM_FIELDS}
        for item in items
    ]


def _apply_status_precedence(order: dict, safe: dict) -> dict:
    """
    Apply status-precedence rules to the sanitized order dict.
    Modifies and returns safe dict.
    """
    status = safe.get("status", "").lower()

    # Rule 1: cancelled or returned — suppress stale shipping/tracking/ETA fields
    if status in TERMINAL_STATUSES:
        safe["carrier"] = None
        safe["tracking_number"] = None
        safe["estimated_delivery"] = None
        safe["shipped_at"] = None
        safe["delivered_at"] = None
        if DEBUG:
            logger.debug(
                f"[order_tool] Status={status} — suppressed stale carrier/ETA fields for {safe['order_id']}"
            )

    # Rule 2: shipped but no ETA — mark explicitly so LLM doesn't invent a date
    if status == "shipped" and not safe.get("estimated_delivery"):
        safe["_no_eta_note"] = "ETA unavailable; do not invent a delivery date."

    # Rule 3: exception status — flag for human handoff
    if status == "exception":
        safe["_handoff_required"] = True
        safe["_handoff_reason"] = "Shipment exception requires support review."

    return safe


def lookup_order(order_id_raw: str) -> dict[str, Any]:
    """
    Main entry point for the order lookup tool.

    Args:
        order_id_raw: Raw order ID string from the user (may be lowercase, have whitespace, etc.)

    Returns:
        dict with one of:
          - {"found": True, "order": {...sanitized fields...}}
          - {"found": False, "error": "Order not found", "order_id": "..."}
          - {"found": False, "error": "Invalid order ID format", "order_id": "..."}
    """
    normalized = _normalize_order_id(order_id_raw)

    if DEBUG:
        logger.debug(f"[order_tool] Lookup: raw={repr(order_id_raw)!r} normalized={normalized!r}")

    # Basic format validation — ORD-XXXX pattern expected
    if not re.match(r"^ORD-\d+$", normalized):
        logger.warning(f"[order_tool] Invalid order ID format: {normalized!r}")
        return {
            "found": False,
            "error": "Invalid order ID format. Order IDs look like ORD-1007.",
            "order_id": normalized,
        }

    order = _ORDERS_MAP.get(normalized)
    if order is None:
        logger.warning(f"[order_tool] Order not found: {normalized!r}")
        return {
            "found": False,
            "error": "Order not found. Please verify the order ID or contact support.",
            "order_id": normalized,
        }

    # Build sanitized response — only SAFE_FIELDS
    safe: dict[str, Any] = {}
    for field in SAFE_FIELDS:
        if field in order:
            value = order[field]
            if field == "items":
                value = _sanitize_items(value)
            safe[field] = value

    # Apply status-precedence rules (modifies safe in place)
    safe = _apply_status_precedence(order, safe)

    if DEBUG:
        logger.debug(f"[order_tool] Sanitized result for {normalized}: {safe}")

    return {"found": True, "order": safe}


# ── OpenAI function-calling schema ───────────────────────────────────────────
ORDER_LOOKUP_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "order_lookup",
        "description": (
            "Look up the current status of a customer order by order ID. "
            "Call this whenever the customer asks about a specific order. "
            "Do NOT call this without a valid order ID — ask the customer for it first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID provided by the customer, e.g. 'ORD-1007'.",
                }
            },
            "required": ["order_id"],
        },
    },
}
