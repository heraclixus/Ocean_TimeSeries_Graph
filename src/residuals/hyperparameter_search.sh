#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p results

# Create a log directory for this specific grid search
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/grid_search_${TIMESTAMP}"
mkdir -p $LOG_DIR

# Log file for tracking best model
BEST_MODEL_LOG="${LOG_DIR}/best_model.txt"
echo "Hyperparameter Grid Search - $(date)" > $BEST_MODEL_LOG
echo "===================================" >> $BEST_MODEL_LOG

# Initialize best validation loss to infinity
BEST_VAL_LOSS=9999999
BEST_MODEL_PATH=""
BEST_HYPERPARAMS=""

# Log the hyperparameter grid
GRID_LOG="${LOG_DIR}/grid_config.txt"
echo "Hyperparameter Grid Search Configuration" > $GRID_LOG
echo "=======================================" >> $GRID_LOG

# Define hyperparameter grids
HIDDEN_DIMS=(32 64 128)
NUM_HEADS=(2 4 8)
NUM_LAYERS=(1 2 3)
DROPOUTS=(0.1 0.3 0.5)
CRITICAL_FACTORS=(5.0 8.0 10.0)
LEARNING_RATES=(0.00005 0.0001 0.0005)

# Log configuration to grid_config.txt
echo "HIDDEN_DIMS: ${HIDDEN_DIMS[@]}" >> $GRID_LOG
echo "NUM_HEADS: ${NUM_HEADS[@]}" >> $GRID_LOG
echo "NUM_LAYERS: ${NUM_LAYERS[@]}" >> $GRID_LOG
echo "DROPOUTS: ${DROPOUTS[@]}" >> $GRID_LOG
echo "CRITICAL_FACTORS: ${CRITICAL_FACTORS[@]}" >> $GRID_LOG
echo "LEARNING_RATES: ${LEARNING_RATES[@]}" >> $GRID_LOG
echo "" >> $GRID_LOG
echo "Total combinations: $((${#HIDDEN_DIMS[@]} * ${#NUM_HEADS[@]} * ${#NUM_LAYERS[@]} * ${#DROPOUTS[@]} * ${#CRITICAL_FACTORS[@]} * ${#LEARNING_RATES[@]}))" >> $GRID_LOG
echo "Started at: $(date)" >> $GRID_LOG

# Common parameters for all runs
# Use --hyperparameter_search flag to explicitly disable figure saving during search
COMMON_PARAMS="--model transformer --normalize --importance_factor 2.0 --early_stopping 15 --epochs 100 --seed 42 --hyperparameter_search"

# Counter for tracking progress
TOTAL_COMBINATIONS=$((${#HIDDEN_DIMS[@]} * ${#NUM_HEADS[@]} * ${#NUM_LAYERS[@]} * ${#DROPOUTS[@]} * ${#CRITICAL_FACTORS[@]} * ${#LEARNING_RATES[@]}))
CURRENT=0

# Nested loops to iterate through all hyperparameter combinations
for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do
  for NUM_HEAD in "${NUM_HEADS[@]}"; do
    for NUM_LAYER in "${NUM_LAYERS[@]}"; do
      for DROPOUT in "${DROPOUTS[@]}"; do
        for CRITICAL_FACTOR in "${CRITICAL_FACTORS[@]}"; do
          for LR in "${LEARNING_RATES[@]}"; do
            # Increment counter
            CURRENT=$((CURRENT + 1))
            
            # Create a name for this combination
            COMBO_NAME="h${HIDDEN_DIM}_head${NUM_HEAD}_l${NUM_LAYER}_d${DROPOUT}_lr${LR}_cf${CRITICAL_FACTOR}"
            LOG_FILE="${LOG_DIR}/${COMBO_NAME}.log"
            
            echo "[$CURRENT/$TOTAL_COMBINATIONS] Running with $COMBO_NAME"
            
            # Build the full command with current hyperparameters
            CMD="python run_residual_training.py $COMMON_PARAMS --hidden_dim $HIDDEN_DIM --num_heads $NUM_HEAD --num_layers $NUM_LAYER --dropout $DROPOUT --critical_importance_factor $CRITICAL_FACTOR --lr $LR"
            
            # Run the command and capture output to log file
            echo "Command: $CMD" > $LOG_FILE
            echo "Started at: $(date)" >> $LOG_FILE
            echo "-------------------------------------------" >> $LOG_FILE
            
            # Run the command and capture output
            eval "$CMD" >> $LOG_FILE 2>&1
            
            # Extract validation loss from log file
            VAL_LOSS=$(grep "Test Loss (Weighted MSE)" $LOG_FILE | tail -1 | awk '{print $5}')
            MODEL_PATH=$(grep "Model saved to" $LOG_FILE | awk '{print $4}')
            
            echo "Completed at: $(date)" >> $LOG_FILE
            echo "Validation Loss: $VAL_LOSS" >> $LOG_FILE
            echo "Model Path: $MODEL_PATH" >> $LOG_FILE
            
            # Check if this is the best model so far
            if (( $(echo "$VAL_LOSS < $BEST_VAL_LOSS" | bc -l) )); then
              BEST_VAL_LOSS=$VAL_LOSS
              BEST_MODEL_PATH=$MODEL_PATH
              BEST_HYPERPARAMS=$COMBO_NAME
              
              # Update the best model log
              echo "New best model found!" >> $BEST_MODEL_LOG
              echo "Time: $(date)" >> $BEST_MODEL_LOG
              echo "Hyperparameters: $COMBO_NAME" >> $BEST_MODEL_LOG
              echo "Validation Loss: $BEST_VAL_LOSS" >> $BEST_MODEL_LOG
              echo "Model Path: $BEST_MODEL_PATH" >> $BEST_MODEL_LOG
              echo "-------------------------------------------" >> $BEST_MODEL_LOG
              
              # Copy the best model to a specific location
              cp $BEST_MODEL_PATH "${LOG_DIR}/best_model.pt"
            fi
            
            echo "[$CURRENT/$TOTAL_COMBINATIONS] Completed $COMBO_NAME (Loss: $VAL_LOSS)"
            echo ""
          done
        done
      done
    done
  done
done

# After all runs, regenerate the best model with figures
echo "Re-running best configuration to generate figures..."

# Extract hyperparameters from the best model name
BEST_HIDDEN_DIM=$(echo $BEST_HYPERPARAMS | sed -n 's/^h\([0-9]*\)_.*/\1/p')
BEST_NUM_HEAD=$(echo $BEST_HYPERPARAMS | sed -n 's/.*head\([0-9]*\)_.*/\1/p')
BEST_NUM_LAYER=$(echo $BEST_HYPERPARAMS | sed -n 's/.*l\([0-9]*\)_.*/\1/p')
BEST_DROPOUT=$(echo $BEST_HYPERPARAMS | sed -n 's/.*d\([0-9.]*\)_.*/\1/p')
BEST_LR=$(echo $BEST_HYPERPARAMS | sed -n 's/.*lr\([0-9.]*\)_.*/\1/p')
BEST_CF=$(echo $BEST_HYPERPARAMS | sed -n 's/.*cf\([0-9.]*\).*/\1/p')

# Remove --hyperparameter_search flag and add --force_save_figs to generate plots
BEST_CMD="python run_residual_training.py --model transformer --normalize --importance_factor 2.0 --early_stopping 15 --epochs 100 --seed 42 --hidden_dim $BEST_HIDDEN_DIM --num_heads $BEST_NUM_HEAD --num_layers $BEST_NUM_LAYER --dropout $BEST_DROPOUT --critical_importance_factor $BEST_CF --lr $BEST_LR --force_save_figs"

echo "Running: $BEST_CMD"
eval "$BEST_CMD" > "${LOG_DIR}/best_model_with_figures.log" 2>&1

# Finalize the logs
echo "" >> $GRID_LOG
echo "Completed at: $(date)" >> $GRID_LOG
echo "Total runs: $CURRENT" >> $GRID_LOG
echo "" >> $GRID_LOG
echo "Best model:" >> $GRID_LOG
echo "Hyperparameters: $BEST_HYPERPARAMS" >> $GRID_LOG
echo "Validation Loss: $BEST_VAL_LOSS" >> $GRID_LOG
echo "Model Path: $BEST_MODEL_PATH" >> $GRID_LOG

echo "" >> $BEST_MODEL_LOG
echo "Grid search completed at: $(date)" >> $BEST_MODEL_LOG
echo "Final best model:" >> $BEST_MODEL_LOG
echo "Hyperparameters: $BEST_HYPERPARAMS" >> $BEST_MODEL_LOG
echo "Validation Loss: $BEST_VAL_LOSS" >> $BEST_MODEL_LOG
echo "Model Path: $BEST_MODEL_PATH" >> $BEST_MODEL_LOG
echo "Figures have been generated for this model configuration." >> $BEST_MODEL_LOG

echo "Grid search completed! Best model: $BEST_MODEL_PATH with validation loss: $BEST_VAL_LOSS"
echo "Best hyperparameters: $BEST_HYPERPARAMS"
echo "Figures have been generated for the best model configuration."
echo "See ${LOG_DIR} for complete logs"

# Make script executable 
chmod +x "$0" 