#!/bin/bash
# Example usage for gear geometry data generation.
# Set VARIANT to "sym" or "asym" (default: sym)
VARIANT=${1:-sym}

if [ "$VARIANT" = "asym" ]; then
  python3 asym_gear.py \
    --output-path ./output_asym \
    --plot
else
  python3 sym_gear.py \
    --output-path ./output_sym \
    --plot
fi
