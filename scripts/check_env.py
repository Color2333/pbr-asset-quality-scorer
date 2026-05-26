from __future__ import annotations

import importlib
import sys


REQUIRED_MODULES = [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "sklearn",
    "PIL",
    "cv2",
    "trimesh",
    "yaml",
    "tqdm",
]


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    failed = []
    for module in REQUIRED_MODULES:
        try:
            imported = importlib.import_module(module)
        except Exception as exc:
            failed.append((module, str(exc)))
            print(f"[missing] {module}: {exc}")
            continue
        version = getattr(imported, "__version__", "unknown")
        print(f"[ok] {module}: {version}")

    if failed:
        print("\nEnvironment check failed.")
        return 1

    try:
        import torch

        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device_count: {torch.cuda.device_count()}")
            for idx in range(torch.cuda.device_count()):
                print(f"cuda:{idx}: {torch.cuda.get_device_name(idx)}")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
