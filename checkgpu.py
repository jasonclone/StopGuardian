# checkgpu.py
import torch
import platform
import subprocess
import sys

# Try DirectML
try:
    import torch_directml
    dml_available = True
except ImportError:
    dml_available = False


def get_gpu_info():
    """Return GPU name(s) using system tools."""
    gpus = []

    # Windows: use wmic
    if platform.system() == "Windows":
        try:
            output = subprocess.check_output(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                stderr=subprocess.STDOUT,
                text=True
            )
            for line in output.split("\n"):
                line = line.strip()
                if line and "Name" not in line:
                    gpus.append(line)
        except Exception:
            gpus.append("Could not query GPU names on Windows")

    return gpus


def check_cuda():
    print("=== CUDA CHECK ===")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA GPU Name:", torch.cuda.get_device_name(0))
    else:
        print("No CUDA GPU detected (NVIDIA only).")
    print()


def check_directml():
    print("=== DirectML CHECK ===")
    print("torch_directml installed:", dml_available)
    if dml_available:
        try:
            dml = torch_directml.device()
            print("DirectML device created successfully.")
            print("DirectML is usable on this system.")
        except Exception as e:
            print("DirectML installed but failed to initialize:", e)
    else:
        print("DirectML not installed. Install with: pip install torch-directml")
    print()


def check_rocm():
    print("=== ROCm CHECK ===")
    # ROCm is Linux-only and only for specific AMD GPUs
    if platform.system() != "Linux":
        print("ROCm not supported on this OS (Windows cannot use ROCm).")
        return

    # If Linux, check for ROCm
    try:
        import torch
        print("torch.version.hip:", torch.version.hip)
        print("ROCm available:", torch.version.hip is not None)
    except Exception as e:
        print("ROCm check failed:", e)
    print()


def main():
    print("====================================")
    print(" GPU DIAGNOSTIC TOOL (PyTorch + DML)")
    print("====================================\n")

    print("Python:", sys.version)
    print("OS:", platform.system(), platform.release())
    print()

    print("=== SYSTEM GPU(s) DETECTED ===")
    gpus = get_gpu_info()
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")
    else:
        print("No GPUs detected.")
    print()

    check_cuda()
    check_directml()
    check_rocm()

    print("=== SUMMARY ===")
    if torch.cuda.is_available():
        print("CUDA GPU detected — full PyTorch CUDA support available.")
    elif dml_available:
        print("DirectML available — PyTorch can use your AMD GPU through DirectML.")
        print("  (torch.cuda.is_available() will still be False — this is normal.)")
    else:
        print("No GPU backend available — CPU-only mode.")
    print()


if __name__ == "__main__":
    main()
