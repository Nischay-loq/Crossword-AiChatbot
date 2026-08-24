"""
conftest.py
Shared pytest configuration and fixtures.
"""

import logging
import pytest

# Set up logging for test runs
logging.basicConfig(
    level=logging.WARNING,   # suppress debug noise during tests
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "retrieval: Retrieval quality tests")
    config.addinivalue_line("markers", "groundedness: Groundedness and abstention tests")
    config.addinivalue_line("markers", "tool_use: Tool call behavior tests")
    config.addinivalue_line("markers", "privacy: Privacy and data exposure tests")
    config.addinivalue_line("markers", "security: Prompt injection and security tests")
    config.addinivalue_line("markers", "conversation: Multi-turn conversation tests")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print category summary at end of test run."""
    passed = len(terminalreporter.stats.get("passed", []))
    failed = len(terminalreporter.stats.get("failed", []))
    total = passed + failed
    if total > 0:
        print(f"\n\n{'='*50}")
        print(f"EVAL RESULTS: {passed}/{total} passed ({int(100*passed/total)}%)")
        print(f"{'='*50}")
