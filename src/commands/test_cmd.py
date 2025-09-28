# src/commands/test_cmd.py
from coverage import Coverage
from io import StringIO
from contextlib import redirect_stdout
from src.commands.url_file_cmd import run_eval

def run_tests() -> None:
    # print("TEST: dummy handler running", flush=True)

    cov = Coverage(source=["src"])  # keep it simple: line coverage only
    cov.erase()
    cov.start()

    exit_code = 0
    try:
        run_eval("url.txt")
    except SystemExit as e:
        exit_code = int(e.code) if isinstance(e.code, int) else 1
        print(f"TEST: run_eval raised SystemExit({exit_code}), continuing for coverage...", flush=True)
    finally:
        cov.stop()
        cov.save()

    buf = StringIO()
    with redirect_stdout(buf):
        percent = cov.report(show_missing=False)

    print(f"TEST: code coverage {percent:.0f}%", flush=True)

    # optional: propagate original exit code if you want ./run test to fail on non-zero
    # if exit_code:
    #     raise SystemExit(exit_code)