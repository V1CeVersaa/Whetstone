"""Deprecated alias for scripts/run_eval.py.

Kept so documented commands (CLAUDE.md, docs/) keep working; all behavior,
including the new CLI overrides, lives in run_eval.py.
"""

from run_eval import main

if __name__ == "__main__":
    main()
