#!/bin/bash
# Example usage for square geometry data generation
python3 square.py \
  --data-per-boundary 500 \
  --Nc 20000 \
  --output-path ./output/ \
  --plot \
  --bc-values 50.0 90.0 0.0 0.0
