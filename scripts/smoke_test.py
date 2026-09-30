#!/usr/bin/env python3
"""
Quick end-to-end smoke test: invokes the adapter once against the real Claude Code CLI.

Usage:
    python scripts/smoke_test.py            # uses CLAUDE_TEST_MODEL or "haiku"
    python scripts/smoke_test.py sonnet     # explicit model
"""

import os
import sys
import time
from pathlib import Path

# Allow running from a source checkout without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    model_name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CLAUDE_TEST_MODEL", "haiku")

    try:
        from langchain_core.messages import AIMessage

        from claude_code_langchain import ClaudeCodeChatModel, __version__
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Install the package first: pip install -e .")
        return 1

    print(f"🧪 claude-code-langchain {__version__} smoke test (model: {model_name})")

    try:
        model = ClaudeCodeChatModel(model=model_name)

        start = time.monotonic()
        response = model.invoke("What is 2+2? Reply with just the number.")
        elapsed = time.monotonic() - start

        assert isinstance(response, AIMessage), f"unexpected type {type(response)}"
        assert response.content.strip(), "empty response"

        print(f"   invoke  ✅ {response.content.strip()!r} in {elapsed:.1f}s")
        print(f"   model   ✅ {response.response_metadata.get('model_name')}")
        print(f"   usage   ✅ {response.usage_metadata}")

        chunks = [c.content for c in model.stream("Count from 1 to 5, separated by spaces.")]
        text_chunks = [c for c in chunks if c]
        assert text_chunks, "no streamed content"
        print(f"   stream  ✅ {len(text_chunks)} text chunks: {''.join(text_chunks).strip()!r}")
    except Exception as e:  # noqa: BLE001 - report any failure to the user
        print(f"❌ {type(e).__name__}: {e}")
        return 1

    print("🎉 Smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
