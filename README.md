<p align="center">
  <img src="https://github.com/314arhaam/heat-pinn/blob/main/gallery/2023-11-08-19-09-09_EDIT.org.png" width="360" title="Heat-PINN; made by EDIT.org">
</p>
<h1>Heat-PINN 🔥</h1>
<p> A Physics-Informed Neural Network to solve 2D steady-state heat equation. Based on the methodology introduced in: <a href="https://arxiv.org/abs/1711.10561">Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations</a></p>

## **ToC**
 - [Introduction](#intro)
 - [Results](#res)


## Introduction <a name="intro"></a>
In this project, a PINN is trained to solve a 2D heat equation and the final results is compared to a solution based on FDM method. 
For more detailts about the project read [this](https://github.com/314arhaam/burger-pinn).
### Problem
The governing equation:  
  
![](https://latex.codecogs.com/svg.image?%5Cbg_white%20%5C%5C%5Cfrac%7B%5Cpartial%5E2%7BT%7D%7D%7B%5Cpartial%7Bx%5E2%7D%7D&plus;%5Cfrac%7B%5Cpartial%5E2%7BT%7D%7D%7B%5Cpartial%7By%5E2%7D%7D=0)

in the following domain:  
  
<img src="https://latex.codecogs.com/svg.image?\bg_white&space;\\D&space;=&space;\{(x,&space;y)|-1\leq{x}\leq{&plus;1},-1\leq{y}\leq{&plus;1}\}" title="\bg_white \\D = \{(x, y)|-1\leq{x}\leq{+1},-1\leq{y}\leq{+1}\}" />
  
With following boundary conditions:
  
<img src="https://latex.codecogs.com/svg.image?\bg_white&space;\\T(-1,&space;y)&space;=&space;75.0&space;^\circ{C}\\T(&plus;1,&space;y)&space;=&space;0.0&space;^\circ{C}\\T(x,&space;-1)&space;=&space;50.0&space;^\circ{C}\\T(x,&space;&plus;1)&space;=&space;0.0&space;^\circ{C}\\" title="\bg_white \\T(-1, y) = 75.0 ^\circ{C}\\T(+1, y) = 0.0 ^\circ{C}\\T(x, -1) = 50.0 ^\circ{C}\\T(x, +1) = 0.0 ^\circ{C}\\" />
  

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
