"""Search-R1 style interleaved search on the on-call runbook set."""

__all__ = ["main"]


def main() -> None:
    from agentic_grpo.search.trainer import main as run

    run()
