# src/cli.py
from __future__ import annotations
import sys
from src.commands.install_cmd import run_install
from src.commands.test_cmd import run_tests
from src.commands.url_file_cmd import run_eval

def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        sys.stderr.write("Usage: ./run {install|test|URL_FILE}\n")
        sys.exit(1)

    sub = argv[0]

    if sub == "install":
        run_install()
        return

    if sub == "test":
        run_tests()
        return

    # NEW: handle './run eval <file>' and also './run <file>'
    if sub == "eval":
        if len(argv) < 2:
            sys.stderr.write("Usage: ./run eval URL_FILE\n")
            sys.exit(1)
        run_eval(argv[1])
        return

    # Fallback: treat first arg as URL file path
    run_eval(sub)

if __name__ == "__main__":
    main()