#!/bin/bash
set -uo pipefail

#
# This software is free for non-commercial, research and evaluation use under the terms of the
# LICENSE.md file.
#
# This shell script is a conversion of the Python script provided in the original 3dgs repository to
# facilitate easier integration with COLMAP workflows for 3dgs.
#

# Function to display script usage
show_usage() {
    echo "Usage: $0 -s <source_path> [options]"
    echo ""
    echo "Options:"
    echo "  -s, --source_path <path>   (Required) Path to the data source."
    echo "  --no_gpu                   Disable GPU usage for SIFT extraction/matching."
    echo "  --skip_matching            Skip feature extraction and matching."
    echo "  --camera <model>           Camera model (default: OPENCV)."
    echo "  --colmap_executable <path> Path to the COLMAP executable."
    echo "  --resize                   Enable resizing of images (50%, 25%, 12.5%)."
    echo "  --magick_executable <path> Path to the ImageMagick 'magick' executable (or empty string for v6)."
    echo "  -h, --help                 Show this help message."
}

# Function to check the exit code of the last command
check_exit_code() {
    # $1 = command name, $2 = exit code
    if [ "$2" -ne 0 ]; then
        # Print a newline to clear the progress bar before showing the error
        printf "\n"
        echo "Error: $1 failed with exit code $2. Exiting." >&2
        # Ensure all background jobs are killed on error
        pkill -P $$
        exit 1
    fi
}

# Argument Parsing

# Set default values
NO_GPU=0
SKIP_MATCHING=0
SOURCE_PATH=""
CAMERA="OPENCV"
COLMAP_EXECUTABLE=""
RESIZE=0
# Set default "magick" for Magick v7. Set to "" for v6
MAGICK_CMD="magick"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--source_path)
            SOURCE_PATH="$2"
            shift # past argument
            shift # past value
            ;;
        --no_gpu)
            NO_GPU=1
            shift # past argument
            ;;
        --skip_matching)
            SKIP_MATCHING=1
            shift # past argument
            ;;
        --camera)
            CAMERA="$2"
            shift # past argument
            shift # past value
            ;;
        --colmap_executable)
            COLMAP_EXECUTABLE="$2"
            shift # past argument
            shift # past value
            ;;
        --resize)
            RESIZE=1
            shift # past argument
            ;;
        --magick_executable)
            MAGICK_CMD="$2" # Can be set to "" for ImageMagick 6
            shift # past argument
            shift # past value
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            show_usage
            exit 1
            ;;
    esac
done

# Check if required argument is provided
if [ -z "$SOURCE_PATH" ]; then
    echo "Error: --source_path (-s) is required." >&2
    show_usage
    exit 1
fi

# Set command variables, quoting the path if provided (for use with eval)
COLMAP_CMD="colmap"
[ -n "$COLMAP_EXECUTABLE" ] && COLMAP_CMD="\"$COLMAP_EXECUTABLE\""

# Set GPU flag (1 for use_gpu, 0 for no_gpu)
USE_GPU=1
[ "$NO_GPU" -eq 1 ] && USE_GPU=0

# Main Script Logic

if [ "$SKIP_MATCHING" -eq 0 ]; then
    echo "Running feature extraction and matching..."
    mkdir -p "$SOURCE_PATH/distorted/sparse"
    check_exit_code "mkdir distorted/sparse" $?

    # Feature extraction
    # We use 'eval' to correctly handle the quoted command path if it was provided
    eval $COLMAP_CMD feature_extractor \
        --database_path "\"$SOURCE_PATH/distorted/database.db\"" \
        --image_path "\"$SOURCE_PATH/input\"" \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model "$CAMERA" \
        --SiftExtraction.max_num_features 65536
    check_exit_code "Feature extraction" $?

    #Feature matching
    eval $COLMAP_CMD exhaustive_matcher \
        --database_path "\"$SOURCE_PATH/distorted/database.db\""
    check_exit_code "Feature matching" $?

    # Bundle adjustment
    eval $COLMAP_CMD mapper \
        --database_path "\"$SOURCE_PATH/distorted/database.db\"" \
        --image_path "\"$SOURCE_PATH/input\"" \
        --output_path "\"$SOURCE_PATH/distorted/sparse\"" \
        --Mapper.ba_global_function_tolerance=0.000001
    check_exit_code "Mapper" $?
else
    echo "Skipping feature extraction and matching."
fi

# Image undistortion
echo "Running image undistortion..."
eval $COLMAP_CMD image_undistorter \
    --image_path "\"$SOURCE_PATH/input\"" \
    --input_path "\"$SOURCE_PATH/distorted/sparse/0\"" \
    --output_path "\"$SOURCE_PATH\"" \
    --output_type COLMAP
check_exit_code "Image undistorter" $?

# Organize sparse directory
echo "Organizing sparse directory..."
mkdir -p "$SOURCE_PATH/sparse/0"
check_exit_code "mkdir sparse/0" $?

# Move all files from sparse/ into sparse/0/
# This loop skips the '0' directory itself
for f in "$SOURCE_PATH/sparse"/*; do
    filename=$(basename "$f")
    if [ "$filename" != "0" ]; then
        mv "$f" "$SOURCE_PATH/sparse/0/"
        check_exit_code "Moving $filename to sparse/0" $?
    fi
done

# Image Resizing
if [ "$RESIZE" -eq 1 ]; then
    echo "Copying and resizing images..."

    # Create destination directories
    mkdir -p "$SOURCE_PATH/images_2"
    mkdir -p "$SOURCE_PATH/images_4"
    mkdir -p "$SOURCE_PATH/images_8"
    check_exit_code "Creating resize directories" $?

    # START: Parallel Setup
    # Get number of available CPU cores, default to 4 if nproc not found
    MAX_JOBS=$(nproc 2>/dev/null || echo 4)
    echo "Using up to $MAX_JOBS parallel jobs."
    
    # This function runs in a background subshell for each file
    process_image() {
        local source_file="$1"
        local file=$(basename "$source_file")

        # 50% resize (images_2)
        dest_file_2="$SOURCE_PATH/images_2/$file"
        cp "$source_file" "$dest_file_2" || { echo "Failed copy to images_2: $file"; return 1; }
        eval $MAGICK_CMD mogrify -resize 50% "\"$dest_file_2\"" || { echo "Failed 50% resize: $file"; return 1; }

        # 25% resize (images_4)
        dest_file_4="$SOURCE_PATH/images_4/$file"
        cp "$source_file" "$dest_file_4" || { echo "Failed copy to images_4: $file"; return 1; }
        eval $MAGICK_CMD mogrify -resize 25% "\"$dest_file_4\"" || { echo "Failed 25% resize: $file"; return 1; }

        # 12.5% resize (images_8)
        dest_file_8="$SOURCE_PATH/images_8/$file"
        cp "$source_file" "$dest_file_8" || { echo "Failed copy to images_8: $file"; return 1; }
        eval $MAGICK_CMD mogrify -resize 12.5% "\"$dest_file_8\"" || { echo "Failed 12.5% resize: $file"; return 1; }
    }
    
    # Export function and variables needed by subshells
    export -f process_image
    export SOURCE_PATH
    export MAGICK_CMD
    # END: Parallel Setup

    # START: Progress Bar & Loop
    # Collect all files into an array, handling spaces etc.
    file_list=()
    while IFS= read -r -d '' file; do
        file_list+=("$file")
    done < <(find "$SOURCE_PATH/images" -maxdepth 1 -type f -print0)
    
    total_files=${#file_list[@]}
    processed_count=0
    echo "Found $total_files images to resize."
    
    job_pids=() # Array to store PIDs of background jobs
    
    # Function to update progress bar
    update_progress() {
        processed_count=$((processed_count + 1))
        local percentage=$(( (processed_count * 100) / total_files ))
        local bar_length=40
        local filled_length=$(( (processed_count * bar_length) / total_files ))
        
        local bar=$(printf "%0.s#" $(seq 1 $filled_length))
        local empty=$(printf "%0.s-" $(seq 1 $((bar_length - filled_length))))

        # Print the progress bar, overwriting the line with `\r`
        printf "  Progress: [%s%s] %d%% (%d/%d) \r" "$bar" "$empty" "$percentage" "$processed_count" "$total_files"
    }

    # Loop through all files and process them
    for source_file in "${file_list[@]}"; do
        # Launch the process in the background
        process_image "$source_file" &
        job_pids+=($!) # Store the PID of the new job

        # If we've hit the job limit, wait for the oldest job to finish
        if ((${#job_pids[@]} >= MAX_JOBS)); then
            wait "${job_pids[0]}"
            check_exit_code "Resize job" $?
            job_pids=("${job_pids[@]:1}") # Remove PID from list
            update_progress # Update progress as a job *finishes*
        fi
    done
    
    # Wait for all remaining jobs to finish
    for pid in "${job_pids[@]}"; do
        wait "$pid"
        check_exit_code "Resize job" $?
        update_progress
    done
    
    # END: Progress Bar & Loop
    
    # Print a final newline to move off the progress bar line
    printf "\n"
    echo "Image resizing complete."
fi

echo "Done."
