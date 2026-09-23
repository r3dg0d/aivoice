"""Safe RVC v1/v2 package inspection (no arbitrary code execution)."""

from .detect import RvcPackageInfo, inspect_path, inspect_rvc_package

__all__ = ["RvcPackageInfo", "inspect_path", "inspect_rvc_package"]
