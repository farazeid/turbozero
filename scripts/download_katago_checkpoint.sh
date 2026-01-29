#!/bin/bash
set -e

# Model URL
MODEL_bin_URL="https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-s12192929536-d5655876072.bin.gz"
OUTPUT_DIR="./checkpoints/katago_frozen"

# Create directory
mkdir -p "$OUTPUT_DIR"

# Download
echo "Downloading KataGo checkpoint..."
wget -q --show-progress -O "$OUTPUT_DIR/model.bin.gz" "$MODEL_bin_URL"

# Unzip

# Unzipping not required for bin.gz (or handled by loader)

echo "Download complete. Model saved to $OUTPUT_DIR"
ls -l "$OUTPUT_DIR"
