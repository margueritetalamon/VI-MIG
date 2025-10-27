#!/bin/bash

# Function to display usage
show_usage() {
    echo "Usage: $0 <dataset_name> [options]"
    echo ""
    echo "Required:"
    echo "  dataset_name          Dataset to use (mnist)"
    echo ""
    echo "Options:"
    echo "  --method METHOD       Method to use: ibw, md, laplace (default: all)"
    echo "  --n_components N      Number of components: 1,5 or specific value )"
    echo "  --laplace_type TYPE   Laplace type: diag, kfac, or both "
    echo "  --lr LR              Learning rate"
    echo "  --epochs EPOCHS      Number of epochs "
    echo "  --device DEVICE      Device to use: cpu, gpu"
    echo ""
    echo "Examples:"
    echo "  $0 mnist                                    # Run all methods with default settings"
    echo "  $0 mnist --method ibw                       # Run only IBW with n_components=5,1"
    echo "  $0 mnist --method ibw --n_components 5      # Run only IBW with n_components=5"
    echo "  $0 mnist --method laplace --laplace_type diag  # Run only Laplace diagonal"
    echo "  $0 mnist --method md --n_components 10      # Run MD with n_components=10"
    exit 1
}

# Check if dataset argument is provided
if [ $# -eq 0 ]; then
    show_usage
fi

DATASET=$1
shift  # Remove first argument (dataset name)

# Default values
METHOD="all"
N_COMPONENTS="default"
LAPLACE_TYPE="both"
LR=""
EPOCHS=""
DEVICE=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --n_components)
            N_COMPONENTS="$2"
            shift 2
            ;;
        --laplace_type)
            LAPLACE_TYPE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --prior_var)
            PV="$2"
            shift 2
            ;;
        --bs)
            bs="$2"
            shift 2
            ;;
        --help|-h)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

BASE_DIR="./results_${DATASET}"
mkdir -p $BASE_DIR

# Log file to keep track of all runs
LOG_FILE="${BASE_DIR}/experiment_log.txt"

echo "Starting BNN experiments at $(date)" | tee -a $LOG_FILE
echo "Base directory: ${BASE_DIR}" | tee -a $LOG_FILE
echo "Method: ${METHOD}" | tee -a $LOG_FILE
echo "Number of components: ${N_COMPONENTS}" | tee -a $LOG_FILE
echo "Prior var: ${PV}" | tee -a $LOG_FILE
echo "Laplace type: ${LAPLACE_TYPE}" | tee -a $LOG_FILE
echo "---------------------------------------------" | tee -a $LOG_FILE

# Function to run an experiment with given parameters
run_experiment() {
    local method=$1
    local lr=$2
    local epochs=$3
    local fc_dims=${4:-256}
    local n_components=${5:-5}
    local device=$6
    local pv=$7
    local bs=$8
    
    # Create a descriptive name for this experiment
    exp_name="${method}_lr${lr}_e${epochs}_n${n_components}"
    
    echo "Starting experiment: ${exp_name}" | tee -a $LOG_FILE
    echo "  Method: ${method}" | tee -a $LOG_FILE
    echo "  Learning rate: ${lr}" | tee -a $LOG_FILE
    echo "  Epochs: ${epochs}" | tee -a $LOG_FILE
    echo "  Hidden dimension: ${fc_dims}" | tee -a $LOG_FILE
    echo "  Number of components: ${n_components}" | tee -a $LOG_FILE
    echo "  Device: ${device}" | tee -a $LOG_FILE
    echo "  Started at: $(date)" | tee -a $LOG_FILE
    
    # Run the Python script with the specified parameters
    python train.py \
        --method ${method} \
        --lr ${lr} \
        --epochs ${epochs} \
        --fc_dims ${fc_dims} \
        --n_components ${n_components} \
        --prior_var ${pv} \
        --save_dir ${BASE_DIR} \
        --bs ${bs} \
        --device ${device} \
        --dataset ${DATASET} \
        --optimizer sgd \
        --dropout 0 \
        
    echo "  Finished at: $(date)" | tee -a $LOG_FILE
    echo "---------------------------------------------" | tee -a $LOG_FILE
}

# Create a combined results directory
mkdir -p "${BASE_DIR}/combined_results"

echo "Running experiments..." | tee -a $LOG_FILE

# Determine which methods to run
run_laplace=false
run_ibw=false
run_md=false

if [ "$METHOD" == "all" ]; then
    run_laplace=true
    run_ibw=true
    run_md=true
elif [ "$METHOD" == "laplace" ]; then
    run_laplace=true
elif [ "$METHOD" == "ibw" ]; then
    run_ibw=true
elif [ "$METHOD" == "md" ]; then
    run_md=true
else
    echo "Error: Unknown method '$METHOD'. Valid options: ibw, md, laplace, all"
    exit 1
fi

# Determine n_components to use
if [ "$N_COMPONENTS" == "default" ]; then
    n_comp_list=(5 1)
else
    # User specified specific value(s), could be comma-separated
    IFS=',' read -ra n_comp_list <<< "$N_COMPONENTS"
fi

# Run Laplace experiments
if [ "$run_laplace" == true ]; then
    echo "Running Laplace experiments..." | tee -a $LOG_FILE
    
    laplace_lr=${LR:-5e-4}
    laplace_epochs=${EPOCHS:-250}
    laplace_device=${DEVICE:-gpu}
    laplace_pv=${PV:-10.0} 
    laplace_bs=${bs:-128} 
    
    if [ "$LAPLACE_TYPE" == "both" ] || [ "$LAPLACE_TYPE" == "diag" ]; then
        run_experiment "laplace_diag" "$laplace_lr" "$laplace_epochs" 256 1 "$laplace_device" "$laplace_pv" "$laplace_bs" 
    fi
    
    if [ "$LAPLACE_TYPE" == "both" ] || [ "$LAPLACE_TYPE" == "kfac" ]; then
        run_experiment "laplace_kfac" "$laplace_lr" "$laplace_epochs" 256 1 "$laplace_device" "$laplace_pv" "$laplace_bs" 
    fi
fi

# Run IBW experiments
if [ "$run_ibw" == true ]; then
    echo "Running IBW experiments..." | tee -a $LOG_FILE
    
    ibw_lr=${LR:-5e-3}
    ibw_epochs=${EPOCHS:-1000}
    ibw_device=${DEVICE:-gpu}
    ibw_pv=${PV:-10.0} 
    ibw_bs=${bs:-128} 
    
    for n_comp in "${n_comp_list[@]}"; do
        run_experiment "ibw" "$ibw_lr" "$ibw_epochs" 256 "$n_comp" "$ibw_device" "$ibw_pv" "$ibw_bs"
    done
fi

# Run MD experiments
if [ "$run_md" == true ]; then
    echo "Running MD experiments..." | tee -a $LOG_FILE
    
    md_lr=${LR:-5e-3}
    md_epochs=${EPOCHS:-1000}
    md_device=${DEVICE:-gpu}
    md_pv=${PV:-0.1} 
    
    for n_comp in "${n_comp_list[@]}"; do
        run_experiment "md" "$md_lr" "$md_epochs" 256 "$n_comp" "$md_device" "$md_pv"
    done
fi

# Summary
echo "All experiments completed at $(date)" | tee -a $LOG_FILE
echo "Results saved in ${BASE_DIR}" | tee -a $LOG_FILE

# Create summary of best results
echo "Creating summary of results..." | tee -a $LOG_FILE
SUMMARY_FILE="${BASE_DIR}/combined_results/best_results_summary.txt"
echo "Best Results Summary" > $SUMMARY_FILE
echo "=====================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

echo "Method | Learning Rate | Epochs | N_Comp | Best Test Accuracy | Best ELBO | Final Test NLL | Final KL Div" >> $SUMMARY_FILE
echo "-------|---------------|--------|--------|-------------------|-----------|----------------|------------" >> $SUMMARY_FILE

# Get all training_summary.txt files
find $BASE_DIR -name "training_summary.txt" | while read summary_file; do
    method=$(grep "method:" $summary_file | head -1 | awk '{print $2}')
    lr=$(grep "learning_rate:" $summary_file | awk '{print $2}')
    epochs=$(grep "epochs:" $summary_file | awk '{print $2}')
    n_comp=$(grep "n_components:" $summary_file | awk '{print $2}')
    
    best_acc=$(grep "Best Test Accuracy:" $summary_file | awk '{print $4}')
    best_elbo=$(grep "Best ELBO:" $summary_file | awk '{print $3}')
    
    test_nll=$(grep "test_nll:" $summary_file | awk '{print $2}')
    test_kl=$(grep "test_kl_div:" $summary_file | awk '{print $2}')
    
    echo "$method | $lr | $epochs | $n_comp | $best_acc | $best_elbo | $test_nll | $test_kl" >> $SUMMARY_FILE
done

echo "Experiment batch completed successfully!" | tee -a $LOG_FILE
echo "Check summary at: ${BASE_DIR}/combined_results/best_results_summary.txt" | tee -a $LOG_FILE