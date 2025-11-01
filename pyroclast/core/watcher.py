#!/usr/bin/env python3
"""
File Watcher for Pyroclast

Watch model directory for changes and auto-reconvert.
Perfect for iterative development workflows.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Set, Optional
import subprocess

from pyroclast.utils import Colors, print_success, print_error, print_step, print_warning


class ModelWatcher:
    """Watch model directory and auto-reconvert on changes"""

    def __init__(
        self,
        model_path: str,
        output_dir: str = "./outputs",
        backend: str = "xnnpack",
        interval: float = 2.0,
        verbose: bool = False,
    ):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.backend = backend
        self.interval = interval
        self.verbose = verbose

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # Track file modification times
        self.file_mtimes: Dict[Path, float] = {}
        self.last_conversion_time: float = 0

    def get_model_files(self) -> Set[Path]:
        """Get all relevant model files"""
        files = set()

        if self.model_path.is_file():
            # Single file - watch parent directory
            files.add(self.model_path)
        else:
            # Directory - watch common model files
            patterns = [
                "*.py",  # Python model definitions
                "*.pt",  # PyTorch checkpoints
                "*.pth",  # PyTorch weights
                "*.bin",  # Binary weights
                "*.safetensors",  # SafeTensors format
                "config.json",  # Model config
                "*.json",  # Config files
            ]

            for pattern in patterns:
                files.update(self.model_path.glob(pattern))
                files.update(self.model_path.glob(f"**/{pattern}"))  # Recursive

        return files

    def check_for_changes(self) -> bool:
        """Check if any files have changed"""
        current_files = self.get_model_files()
        changed = False

        # Check for new or modified files
        for file in current_files:
            try:
                mtime = file.stat().st_mtime

                if file not in self.file_mtimes:
                    # New file
                    self.file_mtimes[file] = mtime
                    changed = True
                    if self.verbose:
                        print(
                            f"  New file: {file.relative_to(self.model_path.parent if self.model_path.is_file() else self.model_path)}"
                        )
                elif mtime > self.file_mtimes[file]:
                    # Modified file
                    self.file_mtimes[file] = mtime
                    changed = True
                    if self.verbose:
                        print(
                            f"  Modified: {file.relative_to(self.model_path.parent if self.model_path.is_file() else self.model_path)}"
                        )

            except OSError:
                # File might have been deleted
                pass

        # Check for deleted files
        deleted = set(self.file_mtimes.keys()) - current_files
        if deleted:
            for file in deleted:
                del self.file_mtimes[file]
                changed = True
                if self.verbose:
                    print(f"  Deleted: {file.name}")

        return changed

    def convert_model(self):
        """Run model conversion"""
        print(f"\n{Colors.CYAN}{'─'*70}{Colors.RESET}")
        print_step(f"Converting model with {self.backend} backend...")

        try:
            # Build conversion command
            cmd = [
                sys.executable,
                "pyroclast.py",
                "convert",
                str(self.model_path),
                "-o",
                str(self.output_dir),
                "-b",
                self.backend,
            ]

            if self.verbose:
                cmd.append("-v")

            # Run conversion
            result = subprocess.run(cmd, check=False, capture_output=not self.verbose)

            if result.returncode == 0:
                print_success("Conversion completed successfully")
                self.last_conversion_time = time.time()
            else:
                print_error("Conversion failed")
                if not self.verbose and result.stderr:
                    print(f"{Colors.DIM}{result.stderr.decode()[:500]}{Colors.RESET}")

        except Exception as e:
            print_error(f"Conversion error: {e}")

    def watch(self):
        """Start watching for changes"""
        print(f"{Colors.BOLD}{Colors.CYAN}Pyroclast Watch Mode{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")

        print(f"  Watching:  {self.model_path}")
        print(f"  Output:    {self.output_dir}")
        print(f"  Backend:   {self.backend}")
        print(f"  Interval:  {self.interval}s")
        print(f"\n{Colors.DIM}Press Ctrl+C to stop{Colors.RESET}\n")

        # Initial scan
        print_step("Initial scan...")
        self.get_model_files()  # Populate file_mtimes
        file_count = len(self.file_mtimes)
        print(f"  Watching {file_count} files\n")

        # Initial conversion
        print_step("Initial conversion...")
        self.convert_model()

        print(f"\n{Colors.GREEN}✓ Watching for changes...{Colors.RESET}\n")

        try:
            while True:
                time.sleep(self.interval)

                if self.check_for_changes():
                    print(f"\n{Colors.YELLOW}⚡ Changes detected!{Colors.RESET}")
                    time.sleep(0.5)  # Debounce - wait for file writes to complete
                    self.convert_model()
                    print(f"\n{Colors.GREEN}✓ Watching for changes...{Colors.RESET}")

        except KeyboardInterrupt:
            print(f"\n\n{Colors.CYAN}Watch mode stopped{Colors.RESET}")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Watch model directory and auto-reconvert on changes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Watch model directory
  python3 pyroclast.py watch models/my_model

  # With specific backend
  python3 pyroclast.py watch models/my_model -b xnnpack

  # Custom output and faster polling
  python3 pyroclast.py watch models/my_model -o ./outputs --interval 1.0

  # Verbose mode
  python3 pyroclast.py watch models/my_model -v
        """,
    )

    parser.add_argument("model_path", help="Path to model file or directory")

    parser.add_argument("-o", "--output-dir", default="./outputs", help="Output directory")

    parser.add_argument(
        "-b",
        "--backend",
        choices=["xnnpack", "vulkan", "portable"],
        default="xnnpack",
        help="Conversion backend",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    try:
        watcher = ModelWatcher(
            model_path=args.model_path,
            output_dir=args.output_dir,
            backend=args.backend,
            interval=args.interval,
            verbose=args.verbose,
        )

        watcher.watch()

    except KeyboardInterrupt:
        print(f"\n\n{Colors.CYAN}Watch mode stopped{Colors.RESET}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Watch failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
