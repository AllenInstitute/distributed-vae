import time
import threading
import torch

def current_gpu():
    return torch.cuda.current_device()

def string_to_dtype(s):
  if s == 'fp16':
    return torch.float16
  elif s == 'bf16':
    return torch.bfloat16
  elif s == 'fp32':
    return torch.float32
  else:
    raise ValueError(f"Unknown dtype: {s}")
  
def is_tensor(x):
    return isinstance(x, torch.Tensor)
  
def flatten(x):
  if is_number(x):
      return x
  elif is_tensor(x):
      return x.flatten()
  else:
      raise ValueError('Input must be a tensor')
    
def bin(x):
  return torch.where(x > 0.1, 1.0, 0.0)

def count_gpus():
  return torch.cuda.device_count()