"""
Unit tests for readme_parser helper functions to increase meaningful coverage.
"""

import re
from src.config_parsers_nlp.readme_parser import extract_license_block, extract_section, find_spdx_ids


def test_extract_license_block_and_section():
    md = """
# Project

## License
This project is licensed under the Apache-2.0 License.

## Usage
Do stuff.
"""
    # Should extract the license block content
    block = extract_license_block(md)
    assert block is not None
    assert "Apache-2.0" in block

    # Generic section extraction for "Usage"
    sec = extract_section(md, re.compile(r"(?i)^##\s+Usage", re.M))
    assert sec is not None
    assert "Do stuff." in sec


def test_find_spdx_ids_mixed_text():
    text = "Licensed under MIT or GPL-3.0-only at user's option."
    ids = find_spdx_ids(text)
    lowered = [i.lower() for i in ids]
    assert "mit" in lowered and "gpl-3.0-only" in lowered


