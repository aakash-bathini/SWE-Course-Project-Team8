"""
Reproducibility Metric - Phase 2
Evaluates whether model demonstration code runs successfully

Scoring:
- 0.0: No code found or code doesn't run
- 0.5: Code runs with debugging/modifications
- 1.0: Code runs perfectly without changes
"""

import logging
import re
import subprocess
import tempfile
import os
from typing import Optional
from src.models.model_types import EvalContext

logger = logging.getLogger(__name__)


async def metric(context: EvalContext) -> float:
    """
    Calculate reproducibility score based on demo code execution
    """
    try:
        # Extract demo code from HF data (model card)
        demo_code = _extract_demo_code(context)

        if not demo_code:
            logger.info("No demo code found in model card")
            return -1.0  # Not applicable (like reviewedness when no GitHub repo)

        # Try to execute the code
        execution_result = _test_code_execution(demo_code)

        if execution_result == "perfect":
            return 1.0
        elif execution_result == "partial":
            return 0.5
        else:
            return 0.0

    except Exception as e:
        logger.error(f"Reproducibility metric error: {e}")
        return 0.0


def _extract_demo_code(context: EvalContext) -> Optional[str]:
    """
    Extract Python demo code from model card
    """
    try:
        if not context.hf_data or len(context.hf_data) == 0:
            return None

        hf_info = context.hf_data[0]
        readme_text = hf_info.get("readme_text", "")

        if not readme_text:
            return None

        # Find Python code blocks in markdown
        # Pattern: ```python ... ``` or ```py ... ```
        code_pattern = r"```(?:python|py)\s*\n(.*?)```"
        matches = re.findall(code_pattern, readme_text, re.DOTALL | re.IGNORECASE)

        if not matches:
            return None

        # Return the first substantial code block (>20 chars)
        for code in matches:
            if len(code.strip()) > 20:
                return code.strip()

        return None

    except Exception as e:
        logger.error(f"Error extracting demo code: {e}")
        return None


def _test_code_execution(code: str) -> str:
    """
    Test if code executes successfully
    Returns: 'perfect', 'partial', or 'failed'
    """
    try:
        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            temp_file = f.name
            # Add basic error handling and imports
            test_code = f"""
import sys
import warnings
warnings.filterwarnings('ignore')

try:
{_indent_code(code, 4)}
    sys.exit(0)
except ImportError as e:
    # Missing dependencies - consider partial success
    sys.exit(50)
except Exception as e:
    # Other errors - failure
    sys.exit(1)
"""
            f.write(test_code)

        try:
            # Run with timeout to prevent hanging
            result = subprocess.run(
                ["python", temp_file], capture_output=True, timeout=5, text=True
            )

            if result.returncode == 0:
                return "perfect"
            elif result.returncode == 50:
                # Import errors suggest it would work with dependencies
                return "partial"
            else:
                return "failed"

        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    except subprocess.TimeoutExpired:
        logger.warning("Code execution timeout - considering as partial success")
        return "partial"
    except Exception as e:
        logger.error(f"Code execution test failed: {e}")
        return "failed"


def _indent_code(code: str, spaces: int) -> str:
    """Add indentation to code block"""
    indent = " " * spaces
    return "\n".join(indent + line if line.strip() else line for line in code.split("\n"))
