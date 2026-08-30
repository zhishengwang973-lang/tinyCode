"""Entry point for TinyCode. Run with: python -m tinyCode"""

import sys

from tinyCode.main import entry_point as _app_entry_point


def entry_point() -> None:
    """Dispatch the interactive app or the non-interactive evaluation CLI."""
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        from tinyCode.evals.cli import entry_point as eval_entry_point

        eval_entry_point(sys.argv[2:])
        return
    _app_entry_point()

if __name__ == "__main__":
    entry_point()
