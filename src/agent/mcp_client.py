"""MCP transport config for the jobtrack server.

Phase 1: launch jobtrack as a stdio subprocess via `uv run`. The Claude Agent
SDK spawns the process, pipes stdin/stdout, and speaks MCP over those pipes.

Phase 4 will swap this for HTTP transport when jobtrack lives on a separate
host. Only this module should need to change; the agent loop itself doesn't
care which transport is in use.

Env vars:
    JOBTRACK_PROJECT_PATH: path to the jobtrack repo. Defaults to
        ../application-tracker relative to this repo's parent dir.
"""
from __future__ import annotations

import os
from pathlib import Path

JOBTRACK_SERVER_NAME = "jobtrack"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_JOBTRACK_PATH = _REPO_ROOT.parent / "application-tracker"


def jobtrack_mcp_config() -> dict:
    project = os.environ.get("JOBTRACK_PROJECT_PATH", str(_DEFAULT_JOBTRACK_PATH))
    return {
        "command": "uv",
        "args": ["run", "--project", project, "jobtrack-mcp"],
    }
