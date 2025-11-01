"""
Formatting utilities for Pyroclast - bytes, latency, numbers, etc.
"""


def format_number(n):
    """Formats a number with commas."""
    return f"{n:,}"


def format_bytes(b):
    """Formats bytes into KB, MB, GB, etc."""
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b/1024:.2f} KB"
    elif b < 1024**3:
        return f"{b/1024**2:.2f} MB"
    else:
        return f"{b/1024**3:.2f} GB"


def format_latency(ms):
    """Formats latency in milliseconds."""
    if ms < 1000:
        return f"{ms:.2f}ms"
    else:
        return f"{ms/1000:.2f}s"
