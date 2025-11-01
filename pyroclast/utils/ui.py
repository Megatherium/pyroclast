"""
UI utilities for Pyroclast - colors, banners, and formatted output.
"""

import sys


class Colors:
    """ANSI color codes for terminal output"""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"
    RESET = "\033[0m"  # Alias for END


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


def print_banner():
    """Print Pyroclast ASCII banner"""
    banner = f"""{Colors.BOLD}{Colors.CYAN}
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   ██████╗ ██╗   ██╗██████╗  ██████╗  ██████╗██╗      █████╗ ███████╗████████╗ ║
║   ██╔══██╗╚██╗ ██╔╝██╔══██╗██╔═══██╗██╔════╝██║     ██╔══██╗██╔════╝╚══██╔══╝ ║
║   ██████╔╝ ╚████╔╝ ██████╔╝██║   ██║██║     ██║     ███████║███████╗   ██║    ║
║   ██╔═══╝   ╚██╔╝  ██╔══██╗██║   ██║██║     ██║     ██╔══██║╚════██║   ██║    ║
║   ██║        ██║   ██║  ██║╚██████╔╝╚██████╗███████╗██║  ██║███████║   ██║    ║
║   ╚═╝        ╚═╝   ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚══════╝╚═╝  ╚═╝╚══════╝   ╚═╝    ║
║                                                                          ║
║                       Executorch Toolkit v2.0                            ║
║              Convert • Run • Analyze • Optimize • Deploy                 ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
{Colors.END}"""
    print(banner)
