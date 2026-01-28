#!/bin/bash
set -e

# URL for AlphaGo Game 2
# Note: The user provided URL was an HTML page. The actual SGF link is usually embedded.
# Based on typical hosting, let's try to download from a reliable source or the one found.
# Link found: https://www.eurogofed.org/newsp/userfiles/AlphaGo%20games/Game%202.sgf
SGF_URL="https://www.eurogofed.org/newsp/userfiles/AlphaGo%20games/Game%202.sgf"
DATA_DIR="data"
SGF_FILE="$DATA_DIR/alphago_lee_sedol_game2.sgf"
OUTPUT_FILE="$DATA_DIR/alphago_game2.npz"

mkdir -p "$DATA_DIR"

echo "Downloading AlphaGo vs Lee Sedol Game 2..."
uv run python -c "import urllib.request; opener = urllib.request.build_opener(); opener.addheaders = [('User-agent', 'Mozilla/5.0')]; urllib.request.install_opener(opener); urllib.request.urlretrieve('$SGF_URL', '$SGF_FILE')"

echo "Converting SGF to Trajectory..."
uv run python scripts/sgf_to_trajectory.py --sgf "$SGF_FILE" --output "$OUTPUT_FILE"

echo "Done. Dataset saved to $OUTPUT_FILE"
