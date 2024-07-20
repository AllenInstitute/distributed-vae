def truncate(lst, n):
  ret = []
  for i in range(min(n, len(lst))):
    ret.append(lst[(len(lst) // n) * i])
  return ret


def MB(bytes):
  return bytes / 1024**2

def cuda(n):
  try:
    import torch
  except ImportError:
    print("warning: torch not imported")
    return None
  return torch.device(f'cuda:{n}')

def count_params(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)