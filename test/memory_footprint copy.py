"""
https://nadavb.com/Memory-Footprint-of-Neural-Net/
- Nadav Benedek
"""

import torch
import torch.nn as nn

def test_memory(in_size=100, out_size=10, num_layers=200, freeze_start=0, freeze_end=0,
                hidden_size=1000, optimizer_type=torch.optim. Adam, batch_size=1,
                device=0, add_relu=True):

  factor = 1
  sample_input = torch.randn(batch_size, in_size)

  layers = [nn.Linear(in_size, hidden_size)]
  for layer_index in range(num_layers):
    layers_to_append = [nn.Linear(hidden_size, hidden_size, bias=False)]
    if add_relu:
      layers_to_append.append(nn.ReLU())

    # Selectively freeze some layers
    if freeze_start <= layer_index < freeze_end:
      for layer in layers_to_append:
        for param in layer.parameters():
          param.requires_grad = False

    layers.extend(layers_to_append)

  layers.append(nn.Linear(hidden_size, out_size))
  print(f"number of layers: {len(layers)}")
  model = nn.Sequential(*layers)

  optimizer = optimizer_type(model.parameters(), lr=.001)
  start = torch.cuda.memory_allocated (device)
  print("Starting at 0 memory usage as baseline.")
  model.to(device)
  after_model =  (torch.cuda.memory_allocated (device) - start) / factor
  print(f"1: After model to device: {after_model:,}")
  print("")
  for i in range(3):
    print("Iteration", i)

    a = (torch.cuda.memory_allocated(device)  - start) / factor
    # Running the forward pass. Here all activations will be saved, per every sample in batch
    out = model(sample_input.to(device)).sum()
    b = (torch.cuda.memory_allocated(device) - start) / factor
    print(f"2: Memory consumed after forward pass (activations stored, depends on batch size): {b:,} change: ", f'{b - a:,}' )  # batch * num layers * hidden_size * 4 bytes per float

    # Backward step: Here we allocate (unless already allocated) and store the gradient of each non-frozen parameter,
    # and we release/discard the activations which are descendants in the DAG as we go.
    # So at the end the change in memory = +non-frozen parameters (if was unallocated) - non-degenerate activations
    # gradients are accumulated in place in the .grad attribute of the tensors for which gradients are being computed. Each GPU core works on a different
    # part of the .grad tensor, so they can all work in parallel
    out.backward()
    c = (torch.cuda.memory_allocated(device) - start) / factor
    print(f"3: After backward pass (activations released, grad stored) {c:,} change: {c-b:,}")

    # Running the optimizer, at the first time, will store 2 moments for each non-frozen parameter (if using Adam), which will be kept throughout the training
    # So change in memory, in the first time = 2 * non-frozen parameters
    # optimizer changes the model parameters in place
    optimizer.step()
    d = (torch.cuda.memory_allocated(device)  - start) / factor
    print(f"4: After optimizer step (moments stored at first time): {d:,} change: {d-c:,} " )

    # zero_grad = Reset and release gradients tensors created in .backward()
    model.zero_grad()
    e = (torch.cuda.memory_allocated(device)  - start) / factor
    print(f"5: After zero_grad step (grads released): {e:,} change: {e-d:,} " )
    print("")

test_memory(optimizer_type=torch.optim.Adam, batch_size=64, freeze_start=0, freeze_end=0, add_relu=False)
