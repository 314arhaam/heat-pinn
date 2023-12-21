<p align="center">
  <img src="https://github.com/314arhaam/heat-pinn/blob/main/gallery/2023-11-08-19-09-09_EDIT.org.png" width="360" title="Heat-PINN; made by EDIT.org">
</p>  

# 🔥 $\textbf{Heat-PINN}$ 🔥

<p> A Physics-Informed Neural Network to solve 2D steady-state heat equation. Based on the methodology introduced in: <a href="https://arxiv.org/abs/1711.10561">Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations.</a></p>  

## **ToC**
 - [Introduction](#intro)
 - [Results](#res)


## Introduction <a name="intro"></a>
In this project, a PINN is trained to solve a 2D heat equation and the final results is compared to a solution based on FDM method. 
For more detailts about the project read [this](https://github.com/314arhaam/burger-pinn).
### Problem
The governing equation:  
  
### $\nabla^2{T} = (\partial_{xx}+\partial_{yy})T=0$  
in the following domain:  
  

### $D = \\{ (x, y)|-1\le x \le +1 \land -1\le y \le +1 \\}$
With the following boundary conditions:
  

$$
\begin{equation}
  \begin{cases}
    T(-1, y) = 75.0 \degree{C}\\
    T(+1, y) = 0.0 \degree{C}\\
    T(x, -1) = 50.0 \degree{C}\\  
    T(x, +1) = 0.0 \degree{C}\\
  \end{cases}
\end{equation}
$$

## Results <a name="res"></a>
  
### Square geometry 
<p align="center">
  <img src="https://github.com/314arhaam/heat-pinn/blob/main/gallery/results_compare.png" title="pinn-vs-fdm">
</p> 
Temperature profiles:  
<p align="center">
  <img src="https://github.com/314arhaam/heat-pinn/blob/main/gallery/profiles.png" title="profiles">
</p>

### Doughnut geometry
<p align="center">
  <img src="https://github.com/314arhaam/heat-pinn/blob/main/gallery/heat_pinn_doughnotts.png" title="doughnotts">
</p>


## Performance Comparison
Results obtained from a [9 layered DNN](https://github.com/314arhaam/heat-pinn/blob/main/gallery/model_plot.png) (1000 epochs) and FDM code on a 100×100 grid. The FDM code is written in Python, a C++ based solver could perform much better.
|**Method**|**Computation time (s)**|
|-|-|
|PINN|66.35|
|FDM|77.60|


## Note
This implementation is based on [Tensorflow 2.0](https://www.tensorflow.org/guide/effective_tf2) package and made possible by [Google Colabratory](https://colab.research.google.com) GPU.
