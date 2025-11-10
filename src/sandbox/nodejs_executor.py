"""
Node.js v24 sandbox executor for running JavaScript monitoring programs.
Milestone 5.1 - Neal's component
"""

import subprocess
import json
import tempfile
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Node.js v24 executable path (must be in PATH or specify full path)
NODEJS_BIN = "node"
JS_EXECUTION_TIMEOUT = 30  # seconds


def execute_js_program(
    program_code: str,
    model_name: str,
    uploader_username: str,
    downloader_username: str,
    zip_file_path: str,
) -> Dict[str, Any]:
    """
    Execute JavaScript program in Node.js v24 sandbox with monitoring.

    Args:
        program_code: JavaScript code to execute
        model_name: Name of the model being downloaded
        uploader_username: Username of model uploader
        downloader_username: Username of model downloader
        zip_file_path: Path to the ZIP file being downloaded

    Returns:
        {
            "exit_code": int,
            "stdout": str,
            "stderr": str,
            "success": bool (exit_code == 0),
            "blocked": bool (exit_code != 0)
        }

    Raises:
        RuntimeError: If Node.js execution fails or times out
    """
    try:
        # Create wrapper script that injects CLI arguments
        wrapper_code = f"""
// Injected CLI arguments
const MODEL_NAME = {json.dumps(model_name)};
const UPLOADER_USERNAME = {json.dumps(uploader_username)};
const DOWNLOADER_USERNAME = {json.dumps(downloader_username)};
const ZIP_FILE_PATH = {json.dumps(zip_file_path)};

// User's monitoring program
{program_code}
"""

        # Write wrapper to temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".js", delete=False, encoding="utf-8"
        ) as f:
            f.write(wrapper_code)
            temp_js_file = f.name

        try:
            # Execute JavaScript with timeout
            result = subprocess.run(
                [NODEJS_BIN, temp_js_file],
                capture_output=True,
                text=True,
                timeout=JS_EXECUTION_TIMEOUT,
            )

            exit_code = result.returncode
            stdout = result.stdout
            stderr = result.stderr

            logger.info(
                f"JS program executed: model={model_name}, "
                f"uploader={uploader_username}, downloader={downloader_username}, "
                f"exit_code={exit_code}"
            )

            return {
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "success": exit_code == 0,
                "blocked": exit_code != 0,
            }

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_js_file)
            except Exception as e:
                logger.warning(f"Failed to clean up temp JS file: {e}")

    except subprocess.TimeoutExpired:
        logger.error(
            f"JS program execution timed out after {JS_EXECUTION_TIMEOUT}s: "
            f"model={model_name}, downloader={downloader_username}"
        )
        raise RuntimeError(f"JavaScript execution timed out after {JS_EXECUTION_TIMEOUT} seconds")

    except FileNotFoundError:
        logger.error(f"Node.js not found at {NODEJS_BIN}")
        raise RuntimeError(
            f"Node.js v24 not found. Please ensure Node.js is installed and in PATH. "
            f"Tried: {NODEJS_BIN}"
        )

    except Exception as e:
        logger.error(f"Unexpected error executing JS program: {e}")
        raise RuntimeError(f"Failed to execute JavaScript program: {str(e)}")
