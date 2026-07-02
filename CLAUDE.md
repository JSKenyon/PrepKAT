# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

PrepKAT is a small collection of routines for preparing MeerKAT radio telescope data (measurement sets) for processing. It is packaged as a `uv`/hatchling project exposing CLI scripts (built with `typer`) backed by `python-casacore` for measurement set I/O.

## Setup and commands

The project uses `uv` for environment and dependency management, with `hatchling` as the build backend (`[build-system]` in [pyproject.toml](pyproject.toml)).

- Install dependencies: `uv sync`
- Run the CLI directly: `uv run katflip --help`
- After install, the `katflip` entry point (defined in `[project.scripts]`) maps to `prepkat.main:feed_flip`

There are no test, lint, or build scripts configured yet — no test suite exists in the repo.

## Architecture

Each unit of functionality lives in its own subpackage under `prepkat/`, with a thin dispatch wrapper in [prepkat/main.py](prepkat/main.py) that wires a `typer` CLI command to the subpackage's implementation function. `pyproject.toml`'s `[project.scripts]` then exposes that dispatch function as an installable console script (currently only `katflip`).

When adding new functionality, follow this pattern: create a new subpackage under `prepkat/`, implement the core logic there, then add a corresponding thin wrapper function in `main.py` and a script entry in `pyproject.toml`.

### Feed flip ([prepkat/feed_flip/__init__.py](prepkat/feed_flip/__init__.py))

Flips the feeds (swaps XX/XY/YX/YY correlations to YY/YX/XY/XX) of a measurement set in place, and zeroes the `RECEPTOR_ANGLE` column in the `FEED` subtable to match.

Key implementation details:
- Operates on `casacore` tables via `pyrap.tables` (from `python-casacore`), opened read-write directly on disk — this mutates the measurement set in place, there is no dry-run mode.
- Processes each target column in fixed-size row chunks (`chunk_n_row = 10000`) to bound memory usage on large measurement sets, using `getcolnp`/`putcol` for in-place chunked read/write.
- The numba-jitted `apply_flip` does the actual correlation swap and expects the last axis to have exactly 4 correlations; it raises `ValueError` otherwise.
- Idempotency is tracked via a custom table keyword, `PREPKAT_FEED_FLIP`, set on each flipped column (and on `RECEPTOR_ANGLE`) after a successful flip. Re-running on an already-flipped column is a no-op (skipped with a warning) rather than double-flipping.
- Progress is reported per-column via `rich.progress.track`.
