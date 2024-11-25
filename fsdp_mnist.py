# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import argparse
import builtins
import os
import signal
import socket
import time
from datetime import timedelta
from functools import partial, reduce

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch._dynamo import OptimizedModule
from torch.distributed.fsdp import BackwardPrefetch, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (MixedPrecision, ShardingStrategy,
                                    StateDictType)
from torch.distributed.fsdp.wrap import (enable_wrap,
                                         size_based_auto_wrap_policy, wrap)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm

import wandb
from mmidas._dist_utils import (destroy_dist_env, find_addr, find_port,
                                init_dist_env, set_print)


def is_imported(m):
    return m in globals()


def ct_dir(pth):
    return sum(os.path.isdir(os.path.join(pth, x)) for x in os.listdir(pth))


def make_reduce_op(op: str):
    match op:
        case "sum":
            return dist.ReduceOp.SUM
        case "product":
            return dist.ReduceOp.PRODUCT
        case "min":
            return dist.ReduceOp.MIN
        case "max":
            return dist.ReduceOp.MAX
        case _:
            raise ValueError(f"Unknown reduce op: {op}")


def print_train_loss(train_loss, epoch, r):
    if is_master(r):
        print(
            "Train Epoch: {} \tLoss: {:.6f}".format(
                epoch, train_loss[0] / train_loss[1]
            )
        )


def print_test_loss(test_loss, r):
    if is_master(r):
        avg_loss = test_loss[0] / test_loss[2]
        accuracy = test_loss[1] / test_loss[2]
        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                avg_loss, int(test_loss[1]), int(test_loss[2]), accuracy
            )
        )


def print_epoch(epoch, time, r):
    if is_master(r):
        print(f"Epoch {epoch} took {time}sec")


def print_cuda_time(time):
    print(f"CUDA event elapsed time: {time}sec")


def print_summary(time, model, r):
    if is_master(r):
        print_cuda_time(time)
        print(model)


class Net(nn.Module):
    def __init__(self, use_batchnorm=False):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        if not use_batchnorm:
            print(f"warning: not using batchnorm")
        self.use_batchnorm = use_batchnorm

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class DeepNet(nn.Module):
    def __init__(self, use_batchnorm=False):
        super(DeepNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 9000)
        self.fc1a = nn.Linear(9000, 1000)
        self.fc1b = nn.Linear(1000, 1000)
        self.fc1c = nn.Linear(1000, 1000)
        self.fc1d = nn.Linear(1000, 128)
        self.fc2 = nn.Linear(128, 10)
        if not use_batchnorm:
            print(f"warning: not using batchnorm")
        else:
            self.bn1a = nn.BatchNorm1d(1000)
            self.bn1b = nn.BatchNorm1d(1000)
            self.bn1c = nn.BatchNorm1d(1000)
        self.use_batchnorm = use_batchnorm

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc1a(x)
        if self.use_batchnorm:
            x = self.bn1a(x)
        x = self.fc1b(x)
        if self.use_batchnorm:
            x = self.bn1b(x)
        x = self.fc1c(x)
        if self.use_batchnorm:
            x = self.bn1c(x)
        x = self.fc1d(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def make_model(s, r, **kw):
    if s == "shallow":
        model = Net(use_batchnorm=kw.get("use_batchnorm", False))
    elif s == "deep":
        model = DeepNet(use_batchnorm=kw.get("use_batchnorm", False))
    else:
        raise ValueError(f"Unknown model: {s}")
    return model.to(r)


def make_optimizer(model, lr):
    if is_mnist_net(model):
        return 
    else:
        raise ValueError(f"Unknown model: {model}")


def is_fp16(p):
    return p == "fp16"


def is_bf16(p):
    return p == "bf16"


def is_fp32(p):
    return p == "fp32"


def str2dt(s):
    if s == "fp16":
        return torch.float16
    elif s == "bf16":
        return torch.bfloat16
    elif s == "fp32":
        return torch.float32
    elif s == "none":
        return None
    else:
        raise ValueError(f"Unknown dtype: {s}")


def string_to_sharding_strategy(s):
    if s == "full":
        print(f"using sharding strategy: full")
        return ShardingStrategy.FULL_SHARD
    elif s == "grad-op":
        return ShardingStrategy.SHARD_GRAD_OP
    elif s == "no":
        return ShardingStrategy.NO_SHARD
    elif s == "hybrid":
        return ShardingStrategy.HYBRID_SHARD
    elif s == "hybrid-zero2":
        return ShardingStrategy._HYBRID_SHARD_ZERO2
    else:
        raise ValueError(f"Unknown sharding strategy: {s}")


def make_mixed_precision(p):
    dtype = str2dt(p)
    if dtype is None:
        return None
    else:
        return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


def make_wrap_policy(params):
    return partial(size_based_auto_wrap_policy, min_num_params=params)


def is_master(r):
    return r == 0


def transform_loader(loader, *funs):
    return reduce(lambda acc, f: f(acc), funs, loader)


def make_pbar(loader, r, *funs):
    total = len(loader)
    loader = transform_loader(loader, *funs)
    if is_master(r):
        return tqdm(loader, total=total, unit_scale=True)
    else:
        return loader


def is_mnist_net(model):
    return isinstance(module(model), (Net, DeepNet))


def make_test_loss(model, r):
    if is_mnist_net(model):
        return torch.zeros(3, device=r)
    else:
        raise ValueError(f"Unknown model: {model}")


def is_a100(name):
    return "a100" in name.lower()


def using_a100():
    return is_a100(torch.cuda.get_device_name())
        


def convert(x, from_dtype, to_dtype):
    match from_dtype, to_dtype:
        case "B", "MB":
            return x / 1024 / 1024
        case _:
            raise ValueError(f"Unknown dtype: {from_dtype}")


def train(
    model,
    r,
    train_loader,
    optimizer,
    epoch,
    sampler=None,
    losses=None,
    epoch_times=None,
    ws=None,
    mem=None,
):
    model.train()
    ddp_loss = torch.zeros(2, device=r)
    if sampler:
        sampler.set_epoch(epoch)

    for batch_idx, (data, target) in make_pbar(train_loader, r, enumerate):
        data, target = data.to(r), target.to(r)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction="sum")
        mem.append(convert(torch.cuda.memory_allocated(), "B", "MB"))
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] = train_loader.batch_size

    ddp_loss[0] = ddp_loss[0] / (batch_idx + 1)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    losses[epoch] = ddp_loss[0] / ddp_loss[1]
    print_train_loss(ddp_loss, epoch, r)


def test(model, r, test_loader, val_losses, epoch):
    model.eval()
    ddp_loss = make_test_loss(model, r)
    with torch.no_grad():
        for batch_indx, (data, target) in make_pbar(test_loader, r, enumerate):
            data, target = data.to(r), target.to(r)
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] = test_loader.batch_size

    ddp_loss[0] = ddp_loss[0] / (batch_indx + 1)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    val_losses[epoch] = ddp_loss[0] / ddp_loss[2]
    print_test_loss(ddp_loss, r)


def count_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    else:
        return os.cpu_count()


def count_num_workers(args):
    if args.num_workers == -1:
        return count_cpus() // 2
    else:
        return args.num_workers


def main(r, ws, args):
    print(f"starting...")
    init_dist_env(r, ws, args.addr, args.port)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_data = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST("../data", train=False, transform=transform)

    if args.use_dist_sampler:
        train_sampler = DistributedSampler(
            train_data, rank=r, num_replicas=ws, shuffle=False
        )
        test_sampler = DistributedSampler(
            train_data, rank=r, num_replicas=ws, shuffle=False
        )
    else:
        if is_master(r):
            print("warning: not using distributed sampler")
        train_sampler = None
        test_sampler = None

    cuda_kw = {
        "num_workers": count_num_workers(args),
        "pin_memory": True,
        "shuffle": False,
        "drop_last": True,
        "persistent_workers": count_num_workers(args) > 0,
        "prefetch_factor": 2,
    }
    train_loader_config = {
        "batch_size": args.batch_size,
        "sampler": train_sampler,
        **cuda_kw,
    }
    test_loader_config = {
        "batch_size": args.test_batch_size,
        "sampler": test_sampler,
        **cuda_kw,
    }
    train_loader = DataLoader(train_data, **train_loader_config)
    test_loader = DataLoader(test_data, **test_loader_config)
    torch.cuda.set_device(r)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    model = make_model(args.model, r, use_batchnorm=args.use_batchnorm)
    strat = args.sharding
    if ws > 1:
        if strat == "ddp":
            if is_master(r):
                print("using ddp")
            model = DDP(model, device_ids=[r], output_device=r)
        else:
            if is_master(r):
                print("using fsdp")
            model = FSDP(
                model,
                auto_wrap_policy=make_wrap_policy(20000),
                sharding_strategy=string_to_sharding_strategy(strat),
                use_orig_params=args.use_orig_params or args.compile,
                sync_module_states=args.sync,
                mixed_precision=make_mixed_precision(args.mixed),
            )
    if args.compile:
        model = torch.compile(model)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    losses = torch.empty(args.epochs, device=r)
    epoch_times = torch.empty(args.epochs, device=r)
    mem = []

    val_losses = torch.empty(args.epochs, device=r)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    start_event.record()
    for epoch in range(args.epochs):
        tic = time.time()
        train(
            model,
            r,
            train_loader,
            optimizer,
            epoch,
            sampler=train_sampler,
            losses=losses,
            epoch_times=epoch_times,
            ws=ws,
            mem=mem,
        )
        test(model, r, test_loader, val_losses, epoch)
        scheduler.step()
        toc = time.time() - tic
        print_epoch(epoch, toc, r)
        epoch_times[epoch] = toc
    end_event.record()

    dist.barrier()
    if is_master(r):
        # if False:
        print(f"losses: {losses}")
        print(f"val_losses: {val_losses}")
        print(f"mem: {max(mem)}")
        print(f"epoch times: {epoch_times[1:]}")

        _dir = "toy-runs"
        d = f"{_dir}/r{ct_dir(_dir)}"
        os.makedirs(d, exist_ok=True)

        plt.plot(losses[1:].cpu().numpy(), label="train")
        plt.plot(val_losses[1:].cpu().numpy(), label="test")
        plt.legend()
        plt.savefig(f'{d}/{make_filename("LossesPlot")}.png')
        plt.close()

        plt.plot(epoch_times[1:].cpu().numpy(), label="epoch_times")
        plt.legend()
        plt.savefig(f'{d}/{make_filename("EpochTimesPlot")}.png')
        plt.close()

        torch.save(
            {
                "losses": losses,
                "val_losses": val_losses,
                "epoch_times": epoch_times,
                "ws": ws,
                "args": args,
                "mem": mem,
                "mem_summary": max(mem),
                "str(model)": str(model),
                "sharding_strat": strat,
                "prec": args.mixed,
                "orig": args.use_orig_params,
                "sync": args.sync,
                "compile": args.compile,
                # 'gpu': torch.cuda.get_device_name(),
            },
            f'{d}/{make_filename("Run", model=args.model, b=args.batch_size, work=args.num_workers, E=args.epochs, g=args.gpus, dsamp=args.use_dist_sampler, shard=strat, prec=args.mixed, sync=args.sync, comp=args.compile, orig=args.use_orig_params)}.pt',
        )
        print(f"saved to {d}")

    torch.cuda.synchronize()
    print_summary(start_event.elapsed_time(end_event) / 1000, model, r)

    if args.save_model:
        dist.barrier()
        if is_master(r):
            torch.save(model.state_dict(), "mnist_cnn.pt")

    destroy_dist_env()


def str_filter(s, cs):
    return reduce(lambda acc, x: acc.replace(x, ""), cs, s)


def make_filename(nm, **kw) -> str:
    return reduce(
        lambda acc, x: f"{acc}_{str_filter(str(x), ('-', '.')).upper()}{kw[x]}", kw, nm
    )


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="shallow",
        metavar="M",
        help="Model to use: shallow, deep",
    )
    parser.add_argument(
        "--compile", action="store_true", default=False, help="Compile the model"
    )
    parser.add_argument(
        "--mixed",
        type=str,
        default="none",
        metavar="M",
        help="Mixed precision: fp16, bf16, fp32",
    )
    parser.add_argument(
        "--sharding",
        type=str,
        default="full",
        metavar="M",
        help="Sharding strategy: full, grad-op, no, hybrid, hybrid-zero2",
    )
    parser.add_argument("--gpus", type=int, default=-1)
    parser.add_argument("--use_orig_params", default=False, action="store_true")
    parser.add_argument("--sync", default=False, action="store_true")
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--prefetch_factor", type=int, default=-1)
    parser.add_argument("--use_batchnorm", default=False, action="store_true")
    parser.add_argument("--use_dist_sampler", default=False, action="store_true")
    parser.add_argument("--wandb", default=False, action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.gpus == -1:
        ws = torch.cuda.device_count()
    else:
        ws = args.gpus
    print(f"ws: {ws}")
    args.addr = find_addr()
    args.port = find_port(args.addr)
    args.num_workers = count_num_workers(args)
    args.gpus = ws
    args.prefetch_factor = 2
    print(args)
    mp.spawn(main, args=(ws, args), nprocs=ws, join=True)
