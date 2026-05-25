import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_parquet("data/res1.parquet")

plt.scatter(data["x"], data["y"], c=data["val"])
plt.show()