#!/bin/bash
# List existing weatherbench .nc result files

OUTPUT_BASE="${RESULTS_DIR:-./weatherbench_results}"

echo "Weatherbench Results in: $OUTPUT_BASE"
echo "============================================"

if [ ! -d "$OUTPUT_BASE" ]; then
    echo "Directory does not exist!"
    exit 1
fi

# Find all .nc files and extract grid/lead_time info
files=$(find "$OUTPUT_BASE" -name "*.nc" -type f 2>/dev/null | sort)

if [ -z "$files" ]; then
    echo "No .nc files found."
    exit 0
fi

echo ""
printf "%-12s %-10s %s\n" "GRID" "LEAD_TIME" "FILE"
printf "%-12s %-10s %s\n" "----" "---------" "----"

while IFS= read -r file; do
    filename=$(basename "$file")
    # Extract grid and lead_time from pattern: TEST_${resolution}_${lead_time}.nc or similar
    if [[ $filename =~ ^TEST_([A-Z0-9]+)_([0-9a-z]+)\.nc$ ]]; then
        grid="${BASH_REMATCH[1]}"
        lead_time="${BASH_REMATCH[2]}"
        printf "%-12s %-10s %s\n" "$grid" "$lead_time" "$filename"
    else
        # Print other .nc files too
        printf "%-12s %-10s %s\n" "?" "?" "$filename"
    fi
done <<< "$files"

echo ""
echo "Summary:"
echo "--------"
count=$(echo "$files" | wc -l)
echo "Total files: $count"

