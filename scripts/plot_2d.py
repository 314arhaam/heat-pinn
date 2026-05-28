import matplotlib.pyplot as plt
import pandas as pd
import sys

# Load the data
data = pd.read_parquet(sys.argv[1])

# Create the figure
plt.figure(figsize=(10, 7))

# Plot with magma colormap
# 'c' determines the color values, 'cmap' sets the palette
sc = plt.scatter(data["x"], data["y"], c=data["val"], cmap="magma", s=10)

# Add a color bar to show the scale of 'val'
cbar = plt.colorbar(sc)
cbar.set_label('Value') # Optional: label your color bar

# Aesthetics
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Inference Result Visualization")
plt.tight_layout()
plt.savefig(sys.argv[2])
plt.show()