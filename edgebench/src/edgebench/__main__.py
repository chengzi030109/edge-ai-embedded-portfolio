"""Module entry point for ``python -m edgebench``."""

from .cli import main


if __name__ == "__main__":
    # Delegate to the same CLI entry point used by installed console scripts.
    main()
