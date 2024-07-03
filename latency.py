import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def measure_allreduce_time(tensor, num_iters=10):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warm up
    for _ in range(5):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # Measure
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(num_iters):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / num_iters  # in milliseconds
    return elapsed_time

def main(rank, world_size):
    master_rank = rank == 0

    setup(rank, world_size)
    tensor = torch.randn(1000000).cuda()
    avg_time = measure_allreduce_time(tensor, num_iters=100)
    if master_rank:
        print(f'Average allreduce time: {avg_time:.3f} ms')
    cleanup()

    

if __name__ == '__main__':
  world_size = torch.cuda.device_count()
  mp.spawn(main, nprocs=world_size, args=(world_size,), join=True)
