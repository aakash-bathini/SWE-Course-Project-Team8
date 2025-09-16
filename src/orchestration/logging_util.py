#src/orchestration/logging.py
import logging
import sys
import os

_LEVELS = {
    0: None,            # Silent
    1: logging.INFO,    # Info
    2: logging.DEBUG    # Debug
}

def setup_logging(level: int = 0, log_file: str | None = None, also_stderr: bool = False) -> int:
    """
    Configure logging for the pipeline.
    - level: 0 (silent), 1 (info), 2 (debug)
    - log_file: optional path to write logs
    - also_stderr: if True and level>0, duplicate logs to STDERR (keeps STDOUT free)
    Returns the normalized level actually used (0/1/2).
    """
    lvl = 0 if level < 0 else 2 if level > 2 else int(level)

    root = logging.getLogger()
    # reset any previous handlers
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)  # handlers do filtering

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        # touch the file so it exists even if lvl == 0
        open(log_file, "a", encoding="utf-8").close()
        if lvl > 0:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            fh.setLevel(_LEVELS[lvl])
            root.addHandler(fh)

    if also_stderr and lvl > 0:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        sh.setLevel(_LEVELS[lvl])
        root.addHandler(sh)

    return lvl