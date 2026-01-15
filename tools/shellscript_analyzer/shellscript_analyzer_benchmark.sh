#!/usr/bin/env bash
set -euo pipefail

# Change to project root directory (two levels up from tools/shellscript_analyzer/)
cd "$(dirname "$0")/../.."

# Array of models to test (ordered from smallest to largest)
models=(
    "qwen2.5-7b-instruct-mlx"
    "deepseek/deepseek-r1-0528-qwen3-8b"
    "nvidia/nemotron-3-nano"
    "mistralai/devstral-small-2-2512"
    "mistral-small-3.2-24b-instruct-2506-mlx@4bit"
    "google/gemma-3-27b"
    "mistral-small-3.2-24b-instruct-2506-mlx@6bit"
    "qwen/qwen3-coder-30b"
    "qwen3-30b-a3b-thinking-2507-mlx@6bit"
    "glm-4-32b-0414"
)

# Function to parse log and create CSV
parse_log_to_csv() {
    local log_file="$1"
    local csv_file="$2"

    # CSV header
    echo "Script,Expected,Predicted,Match,Error" > "$csv_file"

    # Parse the log file for test results
    grep "Analyzing \[" "$log_file" | while IFS= read -r line; do
        # Extract filepath and result
        # Format: "Analyzing [1/100]: path/to/script.sh ... ✅ PASS" or "❌ FAIL"

        if [[ "$line" =~ Analyzing\ \[[0-9]+/[0-9]+\]:\ (.+)\ \.\.\.\ ([✅❌])\ (PASS|FAIL|ERROR) ]]; then
            filepath="${BASH_REMATCH[1]}"
            predicted="${BASH_REMATCH[3]}"

            # Get script name (basename)
            script_name=$(basename "$filepath")

            # Determine expected result from filename prefix
            if [[ "$script_name" == bad_* ]]; then
                expected="FAIL"
            elif [[ "$script_name" == good_* ]]; then
                expected="PASS"
            else
                expected="UNKNOWN"
            fi

            # Determine match
            if [[ "$expected" == "$predicted" ]]; then
                match="Y"
            else
                match="N"
            fi

            # Error column (empty for now, could be populated from ERROR status)
            error=""
            if [[ "$predicted" == "ERROR" ]]; then
                error="Analysis error"
            fi

            # Output CSV row
            echo "$filepath,$expected,$predicted,$match,$error"
        fi
    done >> "$csv_file"
}

# Create reports directory if it doesn't exist
mkdir -p reports

# Total models
total_models=${#models[@]}
current=0

echo "======================================"
echo "Shell Script Detector Model Benchmark"
echo "======================================"
echo "Total models to test: $total_models"
echo "Test scripts: 100"
echo ""

# Loop through each model
for model in "${models[@]}"; do
    current=$((current + 1))

    # Convert model name to filename (replace / with _)
    log_filename=$(echo "$model" | sed 's/\//_/g')
    log_path="reports/${log_filename}.log"
    csv_path="reports/${log_filename}.csv"

    # Skip if CSV already exists
    if [ -f "$csv_path" ]; then
        echo ""
        echo "======================================"
        echo "[$current/$total_models] Model: $model"
        echo "======================================"
        echo "  ⏭️  SKIPPED - CSV file already exists: $csv_path"
        echo ""
        continue
    fi

    echo ""
    echo "======================================"
    echo "[$current/$total_models] Model: $model"
    echo "======================================"
    echo "  Log file: $log_path"
    echo "  CSV file: $csv_path"
    echo ""
    echo "Please load model '$model' in LM Studio, then press ENTER to start the tests..."
    read -r

    # Record start time
    start_time=$(date +%s)

    # Run the detector and capture output to BOTH log file AND stdout using tee
    {
        echo "======================================"
        echo "Model Benchmark Test"
        echo "======================================"
        echo "Model: $model"
        echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Test scripts: 100"
        echo ""

        # Run the detector (capture exit code but don't let it stop the script)
        set +e  # Temporarily disable exit on error
        uv run python tools/shellscript_analyzer/shellscript_analyzer.py \
            --root-path tools/shellscript_analyzer/example_tests/ \
            --model "$model" \
            --no-cache 2>&1
        exit_code=$?
        set -e  # Re-enable exit on error

        # Record end time
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))

        echo ""
        echo "======================================"
        echo "Benchmark Results"
        echo "======================================"
        echo "Model: $model"
        echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Total elapsed time: ${elapsed} seconds"
        echo "Exit code: $exit_code"
        echo "======================================"
    } 2>&1 | tee "$log_path"

    # Get exit code from the pipeline
    exit_code=${PIPESTATUS[0]}

    # Calculate elapsed time
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    # Check if the test succeeded
    if [ $? -eq 0 ]; then
        echo "  ✅ Completed in ${elapsed}s"
    else
        echo "  ⚠️  Completed with errors in ${elapsed}s (see log for details)"
    fi

    # Parse log to create CSV
    echo -n "  📊 Generating CSV... "
    if parse_log_to_csv "$log_path" "$csv_path"; then
        echo "✓"
    else
        echo "✗ (failed to parse)"
    fi

    echo ""
done

echo "======================================"
echo "All model tests completed!"
echo "======================================"
echo ""
echo "Results saved to reports/*.log and reports/*.csv"
