# PINN and simple NN performance comparison
## Introduction
In this work, 2 neural networks with the same architectures, one with pysics-informed loss functions and the other using simple MSE loss were compared. The model properties for the simple neural network (SNN) and physics-informed neural network (PINN) are presented in the following table:
|Model|Loss functions|Data|
|-|-|-|
|**PINN**|Physics-Informed & MSE|(x, y, T) for BCs<br />(x, y) for Colloc.|
|**SNN**|MSE|(x, y, T)|
## Performance
Performance comparison between PINN and a simple NN (a neural network without MSE loss) are presented in the following table:
|Method|Training time|Loss|Data Size|
|-|-|-|-|
|**PINN**|66.35 (s)|0.019|BC points: (100, 3)<br />Colloc. points: (10000, 2)|
|**SNN**|21.0 (min.)|0.020 (noisy)|Training data: (7000, 3)<br />Test data: (1800, 3)<br />Val. data: (1200, 3)|

## Conclusion
1. SNN takes a relatively long time to be trained with comparable loss. This makes SNN unfeasible compared to both of the FDM solver and the PINN.
2. For a 100*100 grid, 7000 data points was used and the computed loss value for the test data, was 0.25 which means poor performance for the unseen data. Also, 7000 data points simply means the grid is almost solved by the FDM method. With fewer training data, the performance becomes worse.
3. Unlike the data required for the SNN, there is no need to have the value of dependent variable (T) in collocation points used in PINNs. It could be said that these points indicate the domain in which the equation is solved and the non-homogenous terms.

In summary, not only it is not feasible to use SNNs instead of PINNs for this example, it is not reasonable as a huge amount of data is needed for the SNN to be trained.