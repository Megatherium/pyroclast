"""
Utility functions for the Executorch toolkit.
"""

import sys

class Colors:
    """ANSI colors"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

def print_header(text, width=80):
    """Prints a centered header with a border."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*width}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(width)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*width}{Colors.END}\n")

def print_subheader(text):
    """Prints a subheader with a separator."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}▶ {text}{Colors.END}")
    print(f"{Colors.CYAN}{'─'*70}{Colors.END}")

def print_success(text):
    """Prints a success message."""
    print(f"{Colors.GREEN}✓{Colors.END} {text}")

def print_warning(text):
    """Prints a warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")

def print_error(text):
    """Prints an error message."""
    print(f"{Colors.RED}✗{Colors.END} {text}", file=sys.stderr)

def print_info(key, value, unit=""):
    """Prints a key-value pair."""
    print(f"  {Colors.BLUE}{key}:{Colors.END} {Colors.BOLD}{value}{Colors.END} {unit}")

def print_dim(text):
    """Prints a dim message."""
    print(f"{Colors.DIM}{text}{Colors.END}")

def print_step(text):
    """Prints a step message."""
    print(f"{Colors.BLUE}▶{Colors.END} {text}")

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
