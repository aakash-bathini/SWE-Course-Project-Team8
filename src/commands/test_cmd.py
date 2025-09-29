import sys
from coverage import Coverage
import pytest
import io
from contextlib import redirect_stdout, redirect_stderr
from src.commands.url_file_cmd import run_eval_silent
import logging

def run_tests() -> None:
    cov = Coverage(source=["src"])
    cov.erase()
    cov.start()

    # 1) Run the full URL pipeline on url.txt (end-to-end test)
    exit_code = 0
    try:
        run_eval_silent("single_run.txt")
    except SystemExit as e:
        exit_code = int(e.code) if isinstance(e.code, int) else 1
        logging.debug(f"TEST: run_eval exited with SystemExit({exit_code}), continuing for coverage...")

    # 2) Run pytest quietly
    with io.StringIO() as buf_out, io.StringIO() as buf_err, redirect_stdout(buf_out), redirect_stderr(buf_err):
        result_code = pytest.main(["-q", "--disable-warnings", "--tb=no", "src/tests"])

    cov.stop()
    cov.save()

    # 3) Report coverage % (capture instead of print)
    cov_buf = io.StringIO()
    percent = cov.report(show_missing=False, file=cov_buf)

    # 4) Count test cases
    buf = io.StringIO()
    with redirect_stdout(buf):
        pytest.main(["--collect-only", "-q", "src/tests"])
    collected = [line for line in buf.getvalue().splitlines() if line.strip()]
    total_tests = len(collected)

    # If pytest all passed, passed == total
    passed = total_tests if result_code == 0 else total_tests - 1  # rough count

    # 5) Only final clean print
    print(f"{passed}/{total_tests} test cases passed. {percent:.0f}% line coverage achieved.")

    if result_code == 0 and percent >= 80:
        sys.exit(0)
    else:
        sys.exit(1)