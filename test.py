"""Small CUDA diagnostic kept for manual local checks."""

import torch


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    print(torch.cuda.get_device_name(0))
    print("Device count", torch.cuda.device_count())


if __name__ == "__main__":
    main()
