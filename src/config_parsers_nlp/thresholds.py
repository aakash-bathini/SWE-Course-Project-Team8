'''
This file sets the whitelist, blacklist, and any ambiguities for the license metric.
'''

LICENSE_WHITELIST = {
    # Permissive
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Unlicense",
    "Zlib",
    "Artistic-2.0",
    "CC0-1.0",
    "WTFPL",

    # Weak copyleft (generally LGPL-compatible)
    "MPL-2.0",
    "CDDL-1.0",
    "EPL-2.0",

    # LGPL family (explicitly required by ACME)
    "LGPL-2.1-only",
    "LGPL-2.1-or-later",
    "LGPL-3.0-only",
    "LGPL-3.0-or-later",
}


LICENSE_BLACKLIST = {
    # GPL family (strong copyleft, not LGPL-2.1 compatible for redistribution)
    "GPL-2.0-only",
    "GPL-2.0-or-later",
    "GPL-3.0-only",
    "GPL-3.0-or-later",

    # AGPL family (network copyleft)
    "AGPL-3.0-only",
    "AGPL-3.0-or-later",

    # Restrictive / non-free
    "Proprietary",
    "Commercial",
    "Non-Commercial",   # heuristic (see aliases)
    "Shareware",
    "Freeware",
}

LICENSE_AMBIGUOUS_03 = {
    "gpl", "gnu general public license",   # too broad: could be GPLv3
    "open source",
    "free software license",
    "public license",
    "see license",
    "custom license",
    "own license",
    "other license",
}

LICENSE_AMBIGUOUS_07 = {
    "lgpl", "gnu lesser general public license",
    "apache", "apache license",
    "bsd", "bsd license",
    "mpl", "mozilla public license",
    "epl", "eclipse public license",
}



# Aliases to map non-standard license mentions to SPDX identifiers or custom tokens
# (all lowercase for case-insensitive matching)
LICENSE_ALIASES = {
    # Common typos / informal mentions → SPDX
    "mit license": "MIT",
    "mit": "MIT",
    "apache license 2.0": "Apache-2.0",
    "apache 2": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "bsd 2-clause": "BSD-2-Clause",
    "bsd 3-clause": "BSD-3-Clause",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "isc license": "ISC",
    "isc": "ISC",
    "mozilla public license 2.0": "MPL-2.0",
    "mpl-2.0": "MPL-2.0",
    "common development and distribution license": "CDDL-1.0",
    "cddl-1.0": "CDDL-1.0",
    "eclipse public license 2.0": "EPL-2.0",
    "epl-2.0": "EPL-2.0",
    "gnu lgpl v2.1": "LGPL-2.1-only",
    "gnu lesser general public license v2.1": "LGPL-2.1-only",
    "lgpl-2.1": "LGPL-2.1-only",
    "gnu lgpl v3": "LGPL-3.0-only",
    "gnu lesser general public license v3": "LGPL-3.0-only",
    "lgpl-3.0": "LGPL-3.0-only",
    "the unlicense": "Unlicense",
    "unlicense": "Unlicense",
    "zlib license": "Zlib",
    "zlib": "Zlib",
    "artistic license 2.0": "Artistic-2.0",
    "artistic-2.0": "Artistic-2.0",
    "creative commons zero": "CC0-1.0",
    "cc0-1.0": "CC0-1.0",
    "wtfpl license": "WTFPL",
    "wtfpl": "WTFPL",

    # Restrictive phrases → custom blacklist tokens
    "noncommercial": "Non-Commercial",
    "non-commercial": "Non-Commercial",
    "for non commercial use only": "Non-Commercial",
    "proprietary license": "Proprietary",
    "commercial use only": "Commercial",
}

