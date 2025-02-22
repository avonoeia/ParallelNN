import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from big_model import create_model

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(128, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def setup(rank, world_size):
    """
    IP address are dynamically allocated in the research lab. Might be helpful if we can get 
    them to permanently assign these ip addresses to us. Firewall disabled on all nodes so ports
    are accessible without hindrence. 
    """
    os.environ['MASTER_ADDR'] = '172.18.156.32'  
    os.environ['MASTER_PORT'] = '12355'    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    print(f"Rank {rank}, Local Rank {local_rank}, Using GPU {local_rank}")

    setup(rank, world_size)

    in_size = 128
    hidden_size = 500
    num_layers = 200
    out_size = 10

    model = create_model(in_size=in_size, hidden_size=hidden_size, num_layers=num_layers, out_size=out_size, add_relu=False).to(local_rank)
    # model wrapped with FSDP
    model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)

    # Create dataset and dataloader
    inputs = torch.rand(10000, 128)
    targets = torch.randint(0, 10, (10000,))
    dataset = TensorDataset(inputs, targets)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Create optimizer
    optimizer = Adam(model.parameters(), lr=0.001)

    # Training loop
    """
    Logs only appear in master node. Sometimes the program does not exit gracefully on the 
    worker nodes. Probably because the model is too small so it only executes on the master node.
    """
    for epoch in range(1):
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(local_rank), target.to(local_rank)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0 and rank == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')

    cleanup()

if __name__ == "__main__":
    print("World size from current node:", os.environ['WORLD_SIZE'])
    print("Number of GPUs in current node:", torch.cuda.device_count())
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    train(rank, world_size)

