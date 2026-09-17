@echo off
REM Install PyTorch CUDA + diffusers ecosystem for Phase 0H depth experiments
REM Models and HuggingFace cache → D:\hf_cache to avoid filling C:

set PYTHON=C:\Users\berka\AppData\Local\Programs\Python\Python311\python.exe
set HF_HOME=D:\hf_cache
set TORCH_HOME=D:\torch_cache

echo === Installing PyTorch 2.3 + CUDA 12.1 ===
"%PYTHON%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet

echo === Installing diffusers ecosystem ===
"%PYTHON%" -m pip install diffusers transformers accelerate safetensors --quiet

echo === Verifying CUDA ===
"%PYTHON%" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo === Done ===
