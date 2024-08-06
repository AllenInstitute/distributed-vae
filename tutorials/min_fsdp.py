
import argparse
import os
import datetime
from contextlib import contextmanager
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

@contextmanager
def dist_init(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  # print(dist.is_nccl_available())
  dist.init_process_group('nccl', rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=60))
  try:
    yield
  finally:
    dist.destroy_process_group()


def main(rank, world_size, args):
  with dist_init(rank, world_size):
    torch.cuda.set_device(rank)
    assert rank == torch.cuda.current_device()
    assert rank == dist.get_rank()

    _t = torch.tensor([1, 2, 3], device=rank)
    print("before reduce:", _t)
    dist.all_reduce(_t, op=dist.ReduceOp.SUM)
    print("after reduce:", _t)



if __name__ == '__main__':
  # os.environ['TORCH_CPP_LOG_LEVEL'] = 'INFO'
  # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

  parser = argparse.ArgumentParser()
  args = parser.parse_args()
  WORLD_SIZE = torch.cuda.device_count()
  # print("reached")
  mp.spawn(main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
  # main(0, WORLD_SIZE, args)
