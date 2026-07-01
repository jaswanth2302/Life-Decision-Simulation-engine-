# Generative Agent Simulation Engine

A multi-phase simulation engine for constructing high-fidelity **digital twins** from dynamic AI interviews. The engine synthesizes rich agent profiles by orchestrating specialized AI expert personas — a Virtual Psychologist, Behavioral Economist, Political Scientist, and Demographer — to achieve an **85% normalized simulation accuracy** target.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                  Dynamic AI Interview                │
│           (Multi-Domain Transcript Capture)          │
└──────────────────────┬───────────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │     PII Sanitizer Layer   │
         │  (Differential Privacy)   │
         └─────────────┬─────────────┘
                       │
    ┌──────────────────▼──────────────────┐
    │        Expert Analysis Pipeline     │
    │  ┌────────────┐  ┌───────────────┐  │
    │  │ Psychologist│  │ Economist     │  │
    │  └────────────┘  └───────────────┘  │
    │  ┌────────────┐  ┌───────────────┐  │
    │  │ Political   │  │ Demographer   │  │
    │  │ Scientist   │  │ /Sociologist  │  │
    │  └────────────┘  └───────────────┘  │
    └──────────────────┬──────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │      AgentProfile         │
         │  (Immutable Aggregate     │
         │   Root / Identity Matrix) │
         └───────────────────────────┘
```

## Phase 1 — Core Data Contracts

This phase establishes the foundational schema layer:

- **Immutable Pydantic v2 Models** — Strongly-typed domain models with frozen configs, field validation, and explicit constraints.
- **PII Sanitizer** — Regex-based differential privacy layer for transcript scrubbing.
- **Structured Logging** — Pipeline-tick tracking via standard library logging.
- **Test Suite** — Comprehensive pytest coverage for validation, serialization round-trips, and boundary enforcement.

## Project Structure

```
.
├── config/
│   ├── __init__.py
│   └── logging_config.py          # Structured console logging
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── schema.py              # Immutable Pydantic v2 domain models
│   └── utils/
│       ├── __init__.py
│       └── sanitizer.py           # PII scrubbing utilities
├── tests/
│   ├── __init__.py
│   └── test_schema.py             # Validation & serialization tests
├── .env.example
├── pyproject.toml
└── README.md
```

## Quickstart

```bash
# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Key Data Models

| Model                    | Domain Expert         | Purpose                                       |
|--------------------------|-----------------------|-----------------------------------------------|
| `TranscriptTurn`         | Interview System      | Single interview turn with semantic density   |
| `PsychologicalProfile`   | Virtual Psychologist  | Core values, coping mechanisms, anchors       |
| `EconomicDecisionMatrix` | Behavioral Economist  | Risk appetite, loss aversion, heuristics      |
| `SocioPoliticalWorldview`| Political Scientist   | Institutional trust, ideology, authority bias |
| `DemographicBaseline`    | Demographer/Sociologist| Structural constraints, geo context          |
| `MemoryStreamItem`       | Memory System         | Episodic memory with three-factor decay       |
| `AgentMetadata`          | System                | Agent tracking & provenance                   |
| **`AgentProfile`**       | **Aggregate Root**    | **Master immutable identity matrix**          |

## License

MIT
