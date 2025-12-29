#!/usr/bin/env bash
set -euo pipefail

# Download files
curl -fSL "https://cdn.intra.42.fr/pdf/pdf/184750/en.subject.pdf" -o "en.subject.pdf"
curl -fSL "https://cdn.intra.42.fr/document/document/42036/leaves.zip" -o "leaves.zip"

# Extract zip
unzip -q leaves.zip

# Clean up zip file
rm leaves.zip

# Move Apple and Grape folders
mv images/ dataset/

# mv images/Apple_* dataset/
# mv images/Grape_* dataset/

# Create organized folders
mkdir -p dataset/Apples dataset/Grapes
mv dataset/Apple_* dataset/Apples/
mv dataset/Grape_* dataset/Grapes/