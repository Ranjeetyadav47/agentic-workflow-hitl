"""agentflow: a multi-agent invoice-approval workflow with human-in-the-loop review.

Design in one paragraph: LLMs do perception (reading messy invoice text) and
explanation (writing the note a human reviewer sees). Deterministic code makes
every decision. Every step is logged to SQLite with its inputs, outputs,
latency and attempt count, so a run can be audited or replayed after the fact.
"""

__version__ = "0.1.0"
