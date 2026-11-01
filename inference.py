"""Entry point kept for compatibility: `python inference.py [--replay] [--no-camera]`."""
from firenode.node import main

if __name__ == "__main__":
    raise SystemExit(main())
