#!/bin/bash

# Default values
LATEST_N_DAYS=1
OUTPUT_DIR="data/katago"

# Function to show usage
usage() {
    echo "Usage: $0 [--latest-n-days N] [--output-dir DIR]"
    echo "  --latest-n-days N  Number of latest days to download (default: 1)"
    echo "  --output-dir DIR   Directory to save data (default: data/katago)"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --latest-n-days) LATEST_N_DAYS="$2"; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

echo "Downloading $LATEST_N_DAYS latest days of KataGo training data to $OUTPUT_DIR..."

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Fetch index and extract all npzs.tgz links
INDEX_URL="https://katagoarchive.org/kata1/trainingdata/index.html"
BASE_URL="https://katagoarchive.org/kata1/trainingdata/"

# Get the list of archives, sort them, and take the last N
# Pattern: YYYY-MM-DDnpzs.tgz
ARCHIVES=$(curl -s "$INDEX_URL" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}npzs\.tgz' | sort -r | head -n "$LATEST_N_DAYS")

if [ -z "$ARCHIVES" ]; then
    echo "No archives found matching the pattern."
    exit 1
fi

for ARCHIVE in $ARCHIVES; do
    URL="${BASE_URL}${ARCHIVE}"
    echo "Downloading $ARCHIVE..."
    curl -L "$URL" -o "${OUTPUT_DIR}/${ARCHIVE}"
    
    echo "Extracting $ARCHIVE..."
    tar -xzf "${OUTPUT_DIR}/${ARCHIVE}" -C "$OUTPUT_DIR"
    
    # Optional: remove the tgz after extraction to save space, but maybe keep for now
    # rm "${OUTPUT_DIR}/${ARCHIVE}"
done

echo "Done."
