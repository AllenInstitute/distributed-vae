import builtins
from functools import partial
import os
import signal
from datetime import timedelta
import socket

import torch
import torch.distributed as dist


def init_dist_env(rank, world_size, addr=None, port=None):
    _init_dist_flags(addr, port)
    _init_gpu_flags()
    init_pg(rank, world_size)
    torch.cuda.set_device(rank)
    set_print(rank)


def destroy_dist_env():
    destroy_pg()


def _init_dist_flags(addr, port):
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = port


def _init_gpu_flags():
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    if "a100" in torch.cuda.get_device_name().lower():
        os.environ["NCCL_P2P_LEVEL"] = "NVL"  # new! use only for a100's
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        print("warning: changing matmul precision")
        print("warning: changing NCCL_P2P_LEVEL to NVL")


def init_pg(rank, world_size):
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=300)
    )
    signal.signal(signal.SIGINT, lambda _, __: destroy_pg())


def destroy_pg():
    dist.destroy_process_group()


def set_print(rank):
    builtins.print = partial(print, f"[R{rank}]")


def find_addr():
    return socket.gethostbyname_ex(socket.gethostname())[2][0]


def find_port(addr):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((addr, 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port
