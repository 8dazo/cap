#!/usr/bin/env python3
import platform

import torch


def main() -> None:
    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()
    cuda_available = torch.cuda.is_available()

    if mps_available:
        selected = "mps"
    elif cuda_available:
        selected = "cuda"
    else:
        selected = "cpu"

    print(f"python_platform: {platform.platform()}")
    print(f"torch_version: {torch.__version__}")
    print(f"mps_available: {mps_available}")
    print(f"mps_built: {mps_built}")
    print(f"cuda_available: {cuda_available}")
    print(f"selected_device: {selected}")


if __name__ == "__main__":
    main()
