#!/bin/bash

# Check if dataset argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset_name>"
    echo "Example: $0 mnist"
    echo "         $0 boston"
    echo "         $0 cifar10"
    exit 1
fi

DATASET=$1
BASE_DIR="./results_${DATASET}"
mkdir -p $BASE_DIR

# Log file to keep track of all runs
LOG_FILE="${BASE_DIR}/experiment_log.txt"

echo "Starting BNN experiments at $(date)" | tee -a $LOG_FILE
echo "Base directory: ${BASE_DIR}" | tee -a $LOG_FILE
echo "---------------------------------------------" | tee -a $LOG_FILE

# Function to run an experiment with given parameters
run_experiment() {
    local method=$1
    local lr=$2
    local epochs=$3
    local fc_dims=${4:-256}  # Default value is 256
    local n_components=${5:-5}  # Default value is 5
    local device=$6
    
    # Create a descriptive name for this experiment
    exp_name="${method}_lr${lr}_e${epochs}_n${n_components}"
    
    echo "Starting experiment: ${exp_name}" | tee -a $LOG_FILE
    echo "  Method: ${method}" | tee -a $LOG_FILE
    echo "  Learning rate: ${lr}" | tee -a $LOG_FILE
    echo "  Epochs: ${epochs}" | tee -a $LOG_FILE
    echo "  Hidden dimension: ${fc_dims}" | tee -a $LOG_FILE
    echo "  Number of components: ${n_components}" | tee -a $LOG_FILE
    echo "  Started at: $(date)" | tee -a $LOG_FILE
    
    # Run the Python script with the specified parameters
    # Use the base directory as save_dir to organize all runs
    python bnn_torch.py \
        --method ${method} \
        --lr ${lr} \
        --epochs ${epochs} \
        --fc_dims ${fc_dims} \
        --n_components ${n_components} \
        --save_dir ${BASE_DIR} \
        --save_interval 1 \
        --bs 128 \
        --compile 0 \
        --warmup_epochs 1 \
        --kl_start 1 \
        --kl_end 1 \
        --device ${device} \
        --dataset ${DATASET} \
        --optimizer sgd \
        --dropout 0 \
        --model mlp \
    
    echo "  Finished at: $(date)" | tee -a $LOG_FILE
    echo "---------------------------------------------" | tee -a $LOG_FILE
}

# Create a combined results directory
mkdir -p "${BASE_DIR}/combined_results"

# Run all combinations as specified
# Methods: ibw, md, lin
# Learning rates: 1e-3 (100 epochs)
# n_components: 5, 1 (for ibw and md only)

echo "Running all experiments sequentially..." | tee -a $LOG_FILE

# Method: ibw
echo "Running IBW experiments..." | tee -a $LOG_FILE
run_experiment "ibw" "1e-1" 250 256 1 "cpu"
