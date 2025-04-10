# Add this to test_gpu.py
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU device name:", torch.cuda.get_device_name(0))