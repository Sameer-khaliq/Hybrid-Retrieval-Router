"""
Quick API-key verification script.

Run this on YOUR machine (E:\\hybrid-retrieval-router), not in Claude's
sandbox - your real keys live in your local .env file only.

Checks, for each of Groq / Gemini / Tavily:
  1. Does Settings actually load a non-empty key from .env?
  2. Does a minimal real API call succeed with that key?

Usage:
    uv run python scripts/verify_api_keys.py

(If you don't have a scripts/ folder yet, just drop this file anywhere
in the project root and run: uv run python verify_api_keys.py)
"""

import sys
from pathlib import Path

# Make sure the project root (parent of this scripts/ folder) is on
# sys.path, so `from config.settings import settings` resolves no matter
# which directory this script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings


def check_key_present(name: str, value: str) -> bool:
    if not value:
        print(f"  [FAIL] {name}: empty - not loaded from .env")
        return False
    masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
    print(f"  [OK]   {name}: loaded ({masked})")
    return True


def check_groq() -> bool:
    print("\n--- Groq ---")
    if not check_key_present("GROQ_API_KEY", settings.groq_api_key):
        return False
    try:
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)
        resp = client.chat.completions.create(
            model=settings.groq_fast_model,
            messages=[{"role": "user", "content": "Say 'ok' and nothing else."}],
            max_tokens=5,
            temperature=0,
        )
        text = resp.choices[0].message.content
        print(f"  [OK]   API call succeeded, model replied: {text!r}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] API call failed: {e}")
        return False


def check_gemini() -> bool:
    print("\n--- Gemini (Google AI) ---")
    if not check_key_present("GOOGLE_API_KEY", settings.google_api_key):
        return False
    try:
        from google import genai
        client = genai.Client(api_key=settings.google_api_key)
        resp = client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents="hello world",
        )
        dim = len(resp.embeddings[0].values)
        print(f"  [OK]   API call succeeded, embedding dimension returned: {dim}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] API call failed: {e}")
        return False


def check_tavily() -> bool:
    print("\n--- Tavily ---")
    if not check_key_present("TAVILY_API_KEY", settings.tavily_api_key):
        return False
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.tavily_api_key)
        resp = client.search(query="test", max_results=1)
        n = len(resp.get("results", []))
        print(f"  [OK]   API call succeeded, {n} result(s) returned")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] API call failed: {e}")
        return False


def main():
    print("Verifying API keys from .env ...")
    results = {
        "Groq": check_groq(),
        "Gemini": check_gemini(),
        "Tavily": check_tavily(),
    }

    print("\n=== Summary ===")
    all_ok = True
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")
        all_ok = all_ok and ok

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()