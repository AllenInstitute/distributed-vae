import torch
import torch.distributed as dist
import time
import os
import torch.multiprocessing as mp


def measure_latency(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  dist.init_process_group('nccl', rank=rank, world_size=world_size)
  tensor = torch.ones(1).cuda()

  if rank == 0:
    start_time = time.time()
    dist.send(tensor, dst=1)
    dist.recv(tensor, src=1)
    end_time = time.time()
    print("Latency: ", end_time - start_time)
  else:
    dist.recv(tensor, src=0)
    dist.send(tensor, dst=0)
  
  dist.destroy_process_group()
  
if __name__ == "__main__":
  world_size = 2
  rank = int(input("Enter rank: "))
  mp.spawn(measure_latency, args=(world_size,), nprocs=world_size, join=True)