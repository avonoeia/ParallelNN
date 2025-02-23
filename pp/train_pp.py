import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, ScheduleGPipe
from model import Transformer, ModelArgs

global rank, device, pp_group, stage_index, num_stages

def init_distributed():
    global rank, device, pp_group, stage_index, num_stages
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    dist.init_process_group(backend="nccl")

    # This group can be a sub-group in the N-D parallel case
    pp_group = dist.new_group(backend="nccl")
    stage_index = int(os.environ["RANK"])
    num_stages = world_size

# def manual_model_split(model, input_microbatch) -> PipelineStage:
#     if stage_index == 0:
#         # prepare the first stage model
#         for i in range(4, 8):
#             del model.layers[str(i)]
#         model.norm = None
#         model.output = None

#     elif stage_index == 1:
#         # prepare the second stage model
#         for i in range(4):
#             del model.layers[str(i)]
#         model.tok_embeddings = None

#     stage = PipelineStage(
#         model,
#         stage_index,
#         num_stages,
#         device,
#         input_args=(input_microbatch,),
#     )
#     return stage

if __name__ == "__main__":
    init_distributed()
    num_microbatches = 4
    model_args = ModelArgs()
    model = Transformer(model_args)

    # Dummy data
    x = torch.ones(32, 512, dtype=torch.long)
    y = torch.randint(0, model_args.vocab_size, (32, 512), dtype=torch.long)
    example_input_microbatch = x.chunk(num_microbatches)[0]

    # Option 1: Manual model splitting
    # stage = manual_model_split(model, example_input_microbatch)

    # Option 2: Tracer model splitting
    # stage = tracer_model_split(model, example_input_microbatch)

    pipe = pipeline(
        module=model,
        mb_args=(example_input_microbatch,),
        split_spec={
            # "layers.2": SplitPoint.BEGINNING,
            f'layers.{i * 1}': SplitPoint.BEGINNING
            for i in range(1, num_stages)
        }
    )

    model.to(device)
    x = x.to(device)
    y = y.to(device)

    def tokenwise_loss_fn(outputs, targets):
        loss_fn = nn.CrossEntropyLoss()
        outputs = outputs.reshape(-1, model_args.vocab_size)
        targets = targets.reshape(-1)
        return loss_fn(outputs, targets)

    stage = pipe.build_stage(stage_index=stage_index, device=rank)
    print(f"Pipe on node {stage_index}: {pipe.info()}")
    
    schedule = ScheduleGPipe(stage, n_microbatches=num_microbatches, loss_fn=tokenwise_loss_fn)
    
    if stage_index == 0:
        schedule.step(x)
    elif stage_index == 3:
        losses = []
        output = schedule.step(target=y, losses=losses)
        print(f"losses: {losses}")
    else:
        schedule.step()
    
    dist.barrier()
    dist.destroy_process_group()
    print(f"Rank {stage_index} completes")