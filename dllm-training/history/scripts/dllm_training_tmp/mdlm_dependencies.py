import importlib.util
import torch
print(torch.__version__, flush=True)
print({m: importlib.util.find_spec(m) is not None for m in ['flash_attn', 'omegaconf', 'einops', 'huggingface_hub']}, flush=True)
