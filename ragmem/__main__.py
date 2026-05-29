"""Enable ``python -m ragmem`` as an alias for the ragmem CLI."""

from ragmem.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
