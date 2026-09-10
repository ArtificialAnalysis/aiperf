# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Fallback tool vocabulary for the inline `toolname{` text format.

The preferred vocabulary is the dataset's own per-record `tools` field (see
`reference.tool_vocabulary`); this OpenCode default set applies when no dataset
is at hand (e.g. `--levels 0` without a reference).
"""

TOOLS = [
    "question",
    "bash",
    "read",
    "glob",
    "grep",
    "edit",
    "write",
    "task",
    "webfetch",
    "todowrite",
    "todoread",
    "skill",
]
