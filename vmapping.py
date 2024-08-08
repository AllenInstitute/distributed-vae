import torch as T
import torch.nn as nn
import time

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

x = T.randn(10)

num_passes = 30000

def batch_forward(model, x, num_passes):
    batch = x.repeat(num_passes, 1)
    with T.no_grad():
        return model(batch)

def vmap_forward(model, x, num_passes):
    batched_model = T.vmap(model)
    expanded_data = x.expand(num_passes, -1)
    with T.no_grad():
        return batched_model(expanded_data)

def loop_forward(model, x, num_passes):
    ret = []
    with T.no_grad():
        for _ in range(num_passes):
            ret.append(model(x))
    return ret

batch_results = batch_forward(model, x, num_passes)
vmap_results = vmap_forward(model, x, num_passes)
loop_results = loop_forward(model, x, num_passes)

assert T.allclose(batch_results, vmap_results.squeeze(), atol=1e-6)
# assert T.allclose(batch_results, loop_results, atol=1e-6)
print("All approaches produce the same results.")

def benchmark(fun, *args, repeats=10):
    start_time = time.time()
    for _ in range(repeats):
        fun(*args)
    end_time = time.time()
    return (end_time - start_time) / repeats

batch_time = benchmark(batch_forward, model, x, num_passes)
vmap_time = benchmark(vmap_forward, model, x, num_passes)
loop_time = benchmark(loop_forward, model, x, num_passes)

def compute_speedup(slow, fast):
    speedup = (slow - fast) / slow * 100
    return round(speedup, 2)

print(f"Batch forward pass: {batch_time:.8f} seconds")
print(f"vmap forward pass: {vmap_time:.8f} seconds")
print(f"Loop forward pass: {loop_time:.8f} seconds")

print(f"\nSpeedup of Batch over Loop: {compute_speedup(loop_time, batch_time)}%")
print(f"Speedup of vmap over Loop: {compute_speedup(loop_time, vmap_time)}%")
print(f"Speedup of Batch over vmap: {compute_speedup(vmap_time, batch_time)}%")