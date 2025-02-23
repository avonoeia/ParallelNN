#!/bin/bash

# Define the list of nodes with their IP addresses
declare -A nodes=(
    ["yanPC1"]="172.18.156.32"  # Master node (where the script is executed)
    ["yanPC2"]="172.18.156.204"
    ["yanPC3"]="172.18.156.77"
    ["yanPC4"]="172.18.156.251"
)

# Define the master address and port
MASTER_ADDR="172.18.156.32"  # IP of the master node
MASTER_PORT="12355"

# Define the path to the training script on the master node
TRAIN_ROOT_PATH="experiments/gpt2"
TRAIN_SCRIPT_PATH="$TRAIN_ROOT_PATH/pippy_gpt2.py"

# conda environment name
CONDA_ENV_NAME="gpt2"

# Define the command template
COMMAND_TEMPLATE="source ~/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV_NAME && export MASTER_ADDR=$MASTER_ADDR && export MASTER_PORT=$MASTER_PORT && torchrun --nproc_per_node=1 --nnodes=${#nodes[@]} --node_rank=%d --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT ~/$TRAIN_SCRIPT_PATH"




# Propagate the training script to all worker nodes
for node_name in "${!nodes[@]}"; do
    node_ip="${nodes[$node_name]}"  # Format: ip

    # Skip the master node (no need to copy the file to itself)
    if [[ "$node_ip" != *"$MASTER_ADDR"* ]]; then
        echo "Propagating training script to $node_name ($node_ip)..."

        # Create the directory structure on the worker node if it doesn't exist
        ssh "$node_name@$node_ip" "mkdir -p ~/$TRAIN_ROOT_PATH && exit"

        # Copy the training script to the worker node
        scp "$HOME/$TRAIN_SCRIPT_PATH" "$node_name@$node_ip:~/$TRAIN_ROOT_PATH/"
        scp "$HOME/$TRAIN_ROOT_PATH/hf_utils.py" "$node_name@$node_ip:~/$TRAIN_ROOT_PATH/"
    fi
done






# Start the master node process first
echo "Starting process on master node (yanPC1) with rank 0..."
eval "$(printf "$COMMAND_TEMPLATE" "0")" | tee "logs_yanPC1.txt" &

# Wait for the master node process to start
echo "Waiting for master node to start..."
sleep 5  # Adjust the sleep duration as needed

# Start the worker node processes concurrently
echo "Starting process on yanPC2 with rank 1..."
ssh "yanPC2@172.18.156.204" "$(printf "$COMMAND_TEMPLATE" "1")" > "logs_yanPC2.txt" 2>&1 &

echo "Starting process on yanPC3 with rank 2..."
ssh "yanPC3@172.18.156.77" "$(printf "$COMMAND_TEMPLATE" "2")" > "logs_yanPC3.txt" 2>&1 &

echo "Starting process on yanPC4 with rank 3..."
ssh "yanPC4@172.18.156.251" "$(printf "$COMMAND_TEMPLATE" "3")" > "logs_yanPC4.txt" 2>&1 &

# Wait for all background processes to finish
echo "Waiting for all nodes to complete..."
wait

echo "All nodes have completed the training."

# Display logs from each node
for node_name in "${!nodes[@]}"; do
    echo "===== Logs from $node_name ====="
    cat "logs_$node_name.txt"
    echo "================================"
done

# Ask if the user wants to keep the logs
read -p "Do you want to keep the logs? (y/n) [default: n]: " keep_logs
keep_logs=${keep_logs:-n}  # Set default to 'n' if no input is provided

if [[ "${keep_logs,,}" == "y" ]]; then
    # Ask for a folder name
    read -p "Enter a folder name to save the logs: " folder_name

    # Create the folder in the directory specified in the torchrun command
    log_dir="$HOME/$TRAIN_ROOT_PATH/logs/$folder_name"
    mkdir -p "$log_dir"

    # Move the logs to the folder
    for node_name in "${!nodes[@]}"; do
        mv "logs_$node_name.txt" "$log_dir/"
    done

    echo "Logs have been moved to $log_dir."
else
    # Delete the log files
    for node_name in "${!nodes[@]}"; do
        rm -f "logs_$node_name.txt"
    done

    echo "Logs have been deleted."
fi
