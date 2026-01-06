"""Helpers for structured logging section headings."""

from __future__ import annotations

import logging


def format_boxed_heading(label: str, title: str) -> str:
    """Return a unicode box heading with multi-level numeric label."""
    label = label.strip()
    title = title.strip()
    content = f"{label} {title}".strip()
    line = "═" * (len(content) + 4)
    body = f"╔{line}╗\n║  {content}  ║\n╚{line}╝"
    return f"\n{body}"


def log_boxed_heading(logger: logging.Logger, label: str, title: str, level: int = logging.INFO) -> None:
    """Emit a boxed heading for a pipeline stage."""
    heading = format_boxed_heading(label, title)
    logger.log(level, heading)


__all__ = ["format_boxed_heading", "log_boxed_heading"]
