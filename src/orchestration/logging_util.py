import logging
import sys
import os

_LEVELS = {
    0: None,            # Silent
    1: logging.INFO,    # Info
    2: logging.DEBUG    # Debug
}

def setup_logging_util(also_stderr: bool = False) -> int:
    """
    Configure logging for the pipeline.
    - Reads $LOG_FILE and $LOG_LEVEL from environment variables.
    - Default level is 0 (silent).
    - also_stderr: if True and level > 0, duplicate logs to STDERR.
    Returns the normalized level actually used (0/1/2).
    """

    # Read environment variables
    log_file = os.getenv("LOG_FILE")
    try:
        level = int(os.getenv("LOG_LEVEL", "0"))
    except ValueError:
        level = 0  # fallback if env var is invalid
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    # Normalize log level
    lvl = 0 if level < 0 else 2 if level > 2 else level

    root = logging.getLogger()
    # reset any previous handlers
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)  # handlers do filtering

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")

    # File logging based on environment variable
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        # ensure the file exists even if lvl == 0
        open(log_file, "a", encoding="utf-8").close()
        if lvl > 0:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            fh.setLevel(_LEVELS[lvl])
            root.addHandler(fh)

    # Optional STDERR logging
    if also_stderr and lvl > 0:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        sh.setLevel(_LEVELS[lvl])
        root.addHandler(sh)

    return lvl