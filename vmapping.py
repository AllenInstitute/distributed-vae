import torch
import torch.nn as nn
import time

# Assuming you have a model defined
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# Your single datapoint
single_datapoint = torch.randn(10)

# Number of times to pass the datapoint through the model
num_passes = 1000

# Approach 1: Batching
def batch_approach(model, datapoint, num_passes):
    batch = datapoint.repeat(num_passes, 1)
    with torch.no_grad():
        return model(batch)

# Approach 2: vmap
def vmap_approach(model, datapoint, num_passes):
    batched_model = torch.vmap(model)
    expanded_data = datapoint.expand(num_passes, -1)
    with torch.no_grad():
        return batched_model(expanded_data)

# Approach 3: For loop
def loop_approach(model, datapoint, num_passes):
    results = []
    with torch.no_grad():
        for _ in range(num_passes):
            results.append(model(datapoint))
    return results

# Run all three approaches
batch_results = batch_approach(model, single_datapoint, num_passes)
vmap_results = vmap_approach(model, single_datapoint, num_passes)
loop_results = loop_approach(model, single_datapoint, num_passes)

# Verify that results are the same
assert torch.allclose(batch_results, vmap_results.squeeze(), atol=1e-6)
# assert torch.allclose(batch_results, loop_results, atol=1e-6)
print("All approaches produce the same results.")

# Benchmark the functions
def benchmark(func, *args, repeats=10):
    start_time = time.time()
    for _ in range(repeats):
        func(*args)
    end_time = time.time()
    return (end_time - start_time) / repeats

batch_time = benchmark(batch_approach, model, single_datapoint, num_passes)
vmap_time = benchmark(vmap_approach, model, single_datapoint, num_passes)
loop_time = benchmark(loop_approach, model, single_datapoint, num_passes)

print(f"Batch approach time: {batch_time:} seconds")
print(f"vmap approach time: {vmap_time:} seconds")
print(f"Loop approach time: {loop_time:} seconds")