"""Lambda Biolab lab copilot gateway."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lab-copilot-gateway")
except PackageNotFoundError:
    __version__ = "0.0.0"
