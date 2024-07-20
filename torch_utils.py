import time
import threading
import torch

from my_utils import dprint, avg, bold, convert

def print_available_torch():
  print(f"cpu: {torch.cpu.is_available()}")
  print(f"cuda: {torch.cuda.is_available()}")
  if torch.cuda.is_available():
    print(f"cuda device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
      print(f"cuda {i}: {torch.cuda.get_device_name(i)}")
  print(f"mps: {torch.backends.mps.is_available()}")

class ResourceLogger:
  def __init__(self, world_size, interval=0.005, run=None, rank=None):
    self.memory_allocated = []
    self.max_memory_allocated = []
    self.running = False
    self.interval = interval
    self.world_size = world_size
    self.run = run
    self.rank= rank 

  def log_memory(self):
    torch.cuda.set_device(self.rank)
    while self.running:
      self.memory_allocated.append(convert(torch.cuda.memory_allocated(self.rank), 'B', 'MB'))
      self.max_memory_allocated.append(convert(torch.cuda.max_memory_allocated(self.rank), 'B', 'MB'))
      if self.run is not None:
        self.run.log({f'rank {self.rank} memalloc': self.memory_allocated[-1],
                      f'rank {self.rank} max memalloc': self.max_memory_allocated[-1]})
      time.sleep(self.interval)
    mem_avg = avg(self.memory_allocated)
    dprint(f"{bold(self.rank)}: average memory allocated: {mem_avg}MB")
    if self.run is not None:
      self.run.log({f'rank {self.rank} logger avg memalloc': mem_avg})
    
  def start(self):
    dprint(f"{bold(self.rank)}: starting memory logger")
    self.running = True
    self.thread = threading.Thread(target=self.log_memory)
    self.thread.start()

  def stop(self):
    dprint(f"{bold(self.rank)}: stopping memory logger")
    self.running = False
    self.thread.join()

  def get(self):
    assert not self.running
    return self.memory_allocated
  
def count(model, quantity):
  match quantity:
    case 'params':
      return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def take_percent_data(dataset, percent):
  return torch.utils.data.Subset(dataset, list(range(int(len(dataset) * percent))))

def check_not_a100(rank, backend):
  assert not ("A100" in torch.cuda.get_device_name(rank) and backend == 'nccl')