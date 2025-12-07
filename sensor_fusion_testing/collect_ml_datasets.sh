#!/bin/bash
################################################################################
# ML Dataset Collection Automation Script (Linux/Mac)
################################################################################
# This script automates the collection of GPS/IMU data for training ML models
# to detect GPS spoofing attacks. It provides menu-driven options for
# different collection scenarios.
################################################################################

# Color codes for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored text
print_header() {
    echo -e "${CYAN}============================================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}============================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

print_info() {
    echo -e "${YELLOW}$1${NC}"
}

# Function to display menu
show_menu() {
    clear
    print_header "ML DATASET COLLECTION AUTOMATION"
    echo ""
    echo "Please select collection mode:"
    echo ""
    echo "  1. Quick Test (5 runs x 60s) - Testing and validation"
    echo "  2. One-Class Training (25 runs x 120s) - Standard training dataset"
    echo "  3. One-Class Validation (5 runs x 180s, random) - Validation set"
    echo "  4. Supervised Training (20 runs x 120s) - Balanced supervised learning"
    echo "  5. Supervised Validation (10 runs x 150s, random) - Supervised validation"
    echo "  6. Custom Parameters - Specify your own settings"
    echo "  7. Exit"
    echo ""
    print_header ""
    echo ""
}

# Function to wait for user confirmation
confirm() {
    read -p "Continue? (Y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return 1
    fi
    return 0
}

# Function to run collection with error handling
run_collection() {
    local run_num=$1
    local total_runs=$2
    local command=$3
    
    echo ""
    print_info "[${run_num}/${total_runs}] Collecting run ${run_num}..."
    echo "Time: $(date +%H:%M:%S)"
    
    if eval $command; then
        print_success "Run ${run_num} completed successfully."
        echo ""
        sleep 2
        return 0
    else
        print_error "ERROR: Collection failed on run ${run_num}"
        read -p "Press Enter to return to menu..."
        return 1
    fi
}

################################################################################
# Mode 1: Quick Test
################################################################################
quick_test() {
    clear
    print_header "QUICK TEST - 5 runs x 60 seconds"
    echo ""
    echo "This will collect 5 test runs with 60-second duration each."
    echo "Attack starts after 10 seconds (clean baseline)."
    echo "Output directory: data/test"
    echo ""
    
    confirm || return
    
    mkdir -p data/test
    
    echo ""
    echo "Starting data collection..."
    echo ""
    
    for i in {1..5}; do
        run_collection $i 5 "python ml_data_collector.py --duration 60 --attack-delay 10 --warmup 5 --output-dir data/test --label test_run${i}" || return
    done
    
    echo ""
    print_header "COLLECTION COMPLETE"
    echo "Total runs: 5"
    echo "Estimated samples: ~3,000"
    echo "Location: data/test/"
    echo ""
    read -p "Press Enter to continue..."
}

################################################################################
# Mode 2: One-Class Training
################################################################################
one_class_training() {
    clear
    print_header "ONE-CLASS TRAINING - 25 runs x 120 seconds"
    echo ""
    echo "This will collect 25 training runs with 120-second duration each."
    echo "Attack starts after 30 seconds (clean baseline for training)."
    echo "Output directory: data/training"
    echo ""
    print_info "Estimated time: ~1 hour"
    echo ""
    
    confirm || return
    
    mkdir -p data/training
    
    echo ""
    echo "Starting data collection..."
    echo "This may take approximately 1 hour. Please be patient."
    echo ""
    
    for i in {1..25}; do
        run_collection $i 25 "python ml_data_collector.py --duration 120 --attack-delay 30 --warmup 5 --output-dir data/training --label train_run${i}" || return
    done
    
    echo ""
    print_header "COLLECTION COMPLETE"
    echo "Total runs: 25"
    echo "Estimated samples: ~30,000"
    echo "Location: data/training/"
    echo ""
    read -p "Press Enter to continue..."
}

################################################################################
# Mode 3: One-Class Validation
################################################################################
one_class_validation() {
    clear
    print_header "ONE-CLASS VALIDATION - 5 runs x 180 seconds (Random Attacks)"
    echo ""
    echo "This will collect 5 validation runs with 180-second duration each."
    echo "Random attack mode with unpredictable timing."
    echo "Output directory: data/validation"
    echo ""
    
    confirm || return
    
    mkdir -p data/validation
    
    echo ""
    echo "Starting data collection..."
    echo ""
    
    for i in {1..5}; do
        run_collection $i 5 "python ml_data_collector.py --random-attacks --duration 180 --warmup 5 --min-attack-duration 5 --max-attack-duration 20 --min-clean-duration 5 --max-clean-duration 15 --output-dir data/validation --label val_run${i}" || return
    done
    
    echo ""
    print_header "COLLECTION COMPLETE"
    echo "Total runs: 5"
    echo "Estimated samples: ~9,000"
    echo "Location: data/validation/"
    echo ""
    read -p "Press Enter to continue..."
}

################################################################################
# Mode 4: Supervised Training
################################################################################
supervised_training() {
    clear
    print_header "SUPERVISED TRAINING - 20 runs x 120 seconds"
    echo ""
    echo "This will collect 20 training runs with 120-second duration each."
    echo "Attack starts immediately (attack_delay=0) for maximum labeled pairs."
    echo "Output directory: data/supervised_training"
    echo ""
    print_info "Estimated time: ~45 minutes"
    echo ""
    
    confirm || return
    
    mkdir -p data/supervised_training
    
    echo ""
    echo "Starting data collection..."
    echo "This may take approximately 45 minutes."
    echo ""
    
    for i in {1..20}; do
        run_collection $i 20 "python ml_data_collector.py --attack-delay 0 --duration 120 --warmup 5 --output-dir data/supervised_training --label sup_train_run${i}" || return
    done
    
    echo ""
    print_header "COLLECTION COMPLETE"
    echo "Total runs: 20"
    echo "Estimated samples: ~24,000"
    echo "Location: data/supervised_training/"
    echo ""
    read -p "Press Enter to continue..."
}

################################################################################
# Mode 5: Supervised Validation
################################################################################
supervised_validation() {
    clear
    print_header "SUPERVISED VALIDATION - 10 runs x 150 seconds (Random Attacks)"
    echo ""
    echo "This will collect 10 validation runs with 150-second duration each."
    echo "Random attack mode with immediate start for timing diversity."
    echo "Output directory: data/supervised_validation"
    echo ""
    
    confirm || return
    
    mkdir -p data/supervised_validation
    
    echo ""
    echo "Starting data collection..."
    echo ""
    
    for i in {1..10}; do
        run_collection $i 10 "python ml_data_collector.py --random-attacks --attack-delay 0 --duration 150 --warmup 5 --min-attack-duration 8 --max-attack-duration 25 --min-clean-duration 8 --max-clean-duration 25 --output-dir data/supervised_validation --label sup_val_run${i}" || return
    done
    
    echo ""
    print_header "COLLECTION COMPLETE"
    echo "Total runs: 10"
    echo "Estimated samples: ~15,000"
    echo "Location: data/supervised_validation/"
    echo ""
    read -p "Press Enter to continue..."
}

################################################################################
# Mode 6: Custom Parameters
################################################################################
custom_parameters() {
    clear
    print_header "CUSTOM PARAMETERS"
    echo ""
    echo "Enter your custom collection parameters:"
    echo ""
    
    read -p "Number of runs: " num_runs
    read -p "Duration per run (seconds): " duration
    read -p "Attack delay (seconds, 0 for immediate): " attack_delay
    read -p "Output directory (e.g., data/custom): " output_dir
    read -p "Label prefix (e.g., custom_run): " label_prefix
    read -p "Use random attacks? (Y/N): " use_random
    
    mkdir -p "$output_dir"
    
    echo ""
    print_header "SUMMARY"
    echo "Runs: $num_runs"
    echo "Duration: ${duration}s"
    echo "Attack delay: ${attack_delay}s"
    echo "Random attacks: $use_random"
    echo "Output: $output_dir"
    echo "Label: $label_prefix"
    print_header ""
    echo ""
    
    confirm || return
    
    echo ""
    echo "Starting data collection..."
    echo ""
    
    for ((i=1; i<=$num_runs; i++)); do
        if [[ $use_random =~ ^[Yy]$ ]]; then
            run_collection $i $num_runs "python ml_data_collector.py --random-attacks --duration $duration --attack-delay $attack_delay --warmup 5 --output-dir $output_dir --label ${label_prefix}${i}" || return
        else
            run_collection $i $num_runs "python ml_data_collector.py --duration $duration --attack-delay $attack_delay --warmup 5 --output-dir $output_dir --label ${label_prefix}${i}" || return
        fi
    done
    
    echo ""
    print_header "COLLECTION COMPLETE"
    echo "Total runs: $num_runs"
    echo "Location: $output_dir/"
    echo ""
    read -p "Press Enter to continue..."
}

################################################################################
# Main Loop
################################################################################
main() {
    while true; do
        show_menu
        read -p "Enter your choice (1-7): " choice
        
        case $choice in
            1) quick_test ;;
            2) one_class_training ;;
            3) one_class_validation ;;
            4) supervised_training ;;
            5) supervised_validation ;;
            6) custom_parameters ;;
            7) 
                echo ""
                echo "Exiting..."
                echo ""
                exit 0
                ;;
            *) 
                print_error "Invalid choice. Please try again."
                sleep 2
                ;;
        esac
    done
}

# Run main
main

