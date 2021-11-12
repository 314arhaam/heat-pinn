# heat-pinn
A Physics-Informed Neural Network to solve 2D steady-state heat equation

Based on the methodology introduced in: [Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10561)

## Introduction
In this project, a PINN is trained to solve a 2D heat equation and the final results is compared to a solution based on FDM method. For more detailts about the project please read [this](https://github.com/314arhaam/burger-pinn).
### Problem details
The governing equation:  
  
![](https://latex.codecogs.com/svg.image?%5Cbg_white%20%5C%5C%5Cfrac%7B%5Cpartial%5E2%7BT%7D%7D%7B%5Cpartial%7Bx%5E2%7D%7D&plus;%5Cfrac%7B%5Cpartial%5E2%7BT%7D%7D%7B%5Cpartial%7By%5E2%7D%7D=0)

in the following domain:  
  
<img src="https://latex.codecogs.com/svg.image?\bg_white&space;\\D&space;=&space;\{(x,&space;y)|-1\leq{x}\leq{&plus;1},-1\leq{y}\leq{&plus;1}\}" title="\bg_white \\D = \{(x, y)|-1\leq{x}\leq{+1},-1\leq{y}\leq{+1}\}" />
  
With following boundary conditions:
  
<img src="https://latex.codecogs.com/svg.image?\bg_white&space;\\T(-1,&space;y)&space;=&space;75.0&space;^\circ{C}\\T(&plus;1,&space;y)&space;=&space;0.0&space;^\circ{C}\\T(x,&space;-1)&space;=&space;50.0&space;^\circ{C}\\T(x,&space;&plus;1)&space;=&space;0.0&space;^\circ{C}\\" title="\bg_white \\T(-1, y) = 75.0 ^\circ{C}\\T(+1, y) = 0.0 ^\circ{C}\\T(x, -1) = 50.0 ^\circ{C}\\T(x, +1) = 0.0 ^\circ{C}\\" />
  
Model architecture:  
![]()
## Results
Comparing PINN to FDM:  
![]()
Temperature profiles:  
![]()
