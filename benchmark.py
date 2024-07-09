import torch
import torch.distributed as dist
import time

def benchmark_communication():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    sizes = [2**i for i in range(10, 25)]  # From 1KB to 16MB
    
    def measure_bandwidth(size):
        tensor = torch.randn(size).to('cuda')
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        end = time.perf_counter()
        return size * 4 * 10 / (end - start) / 1e9  # GB/s

    def measure_latency():
        tensor = torch.zeros(1).to('cuda')
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        end = time.perf_counter()
        return (end - start) / 100 * 1e6  # microseconds

    if rank == 0:
        print("Size (KB)\tBandwidth (GB/s)")
        for size in sizes:
            bw = measure_bandwidth(size)
            print(f"{size*4/1024:.2f}\t\t{bw:.2f}")
        
        latency = measure_latency()
        print(f"\nLatency: {latency:.2f} microseconds")

if __name__ == "__main__":
    benchmark_communication()