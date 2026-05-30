<p align="center">
  <img src="https://github.com/314arhaam/heat-pinn/blob/main/assets/2023-11-08-19-09-09_EDIT.org.png" width="360" title="Heat-PINN; made by EDIT.org">
</p>  

[![CI](https://github.com/314arhaam/heat-pinn/actions/workflows/ci.yml/badge.svg)](https://github.com/314arhaam/heat-pinn/actions/workflows/ci.yml)

[![Validation-CPU](https://github.com/314arhaam/heat-pinn/actions/workflows/validation.yml/badge.svg)](https://github.com/314arhaam/heat-pinn/actions/workflows/validation.yml)

# Heat-PINN

<p> A Physics-Informed Neural Network, to solve 2D steady-state heat equation based on the methodology, introduced in: <a href="https://arxiv.org/abs/1711.10561">Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations. </a></p>  

## **Table of Contents**
 - [Introduction](#intro)
 - [Results](#res)


## Introduction <a name="intro"></a>
In this project, a PINN is trained to solve a 2D heat equation and the final results is compared to a solution based on FDM method. 
For more detailts about the project read [this](https://github.com/314arhaam/burger-pinn).
### Problem
The governing equation:  

$$
\Theta = \frac{T - T_{\textbf{min}}}{T_{\textbf{max}}-T_{\textbf{min}}}
$$   

$$ 
\nabla^2{\Theta} = (\partial_{xx}+\partial_{yy})\Theta=0
$$  

in the following domain:  
  

$$  
D = \\{ (x, y)|-1\le x \le +1 \land -1\le y \le +1 \\}
$$  

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
  
When normalized:  

$$
\begin{equation}
  \begin{cases}
    \Theta(-1, y) = 1\\
    \Theta(+1, y) = 0\\
    \Theta(x, -1) = \frac{2}{3}\\  
    \Theta(x, +1) = 0\\
  \end{cases}
\end{equation}
$$

## Heat-PINN CLI

### Installation

### Build

```
.-----------------------------------------------------------------------.
|.##.....#.#######....###...#######........########.###.##....#.##....##|
|.##.....#.##........##.##.....##..........##.....#..##.###...#.###...##|
|.##.....#.##.......##...##....##..........##.....#..##.####..#.####..##|
|.########.######..##.....#....##...######.########..##.##.##.#.##.##.##|
|.##.....#.##......########....##..........##........##.##..###.##..####|
|.##.....#.##......##.....#....##..........##........##.##...##.##...###|
|.##.....#.#######.##.....#....##..........##.......###.##....#.##....##|
'-----------------------------------------------------------------------'

usage: heat build [-h] [--in-shape IN_SHAPE] [--out-shape OUT_SHAPE]
                  [--n-hidden-layers N_HIDDEN_LAYERS]
                  [--neuron-per-layer NEURON_PER_LAYER] [--actfun ACTFUN]
                  [--name NAME]

options:
  -h, --help            show this help message and exit
  --in-shape IN_SHAPE   Shape of the input tensor to feed into NN. Equal to
                        the number of independent variables of PDE.
  --out-shape OUT_SHAPE
                        Shape of the output tensor of NN. For heat transfer
                        it's T (equal to 1)
  --n-hidden-layers N_HIDDEN_LAYERS
                        Number of hidden layers in the NN
  --neuron-per-layer NEURON_PER_LAYER
                        Number of neurons in each hidden layer
  --actfun ACTFUN       Activation function
  --name NAME           Name of the model.
```
### Train
```
.-----------------------------------------------------------------------.
|.##.....#.#######....###...#######........########.###.##....#.##....##|
|.##.....#.##........##.##.....##..........##.....#..##.###...#.###...##|
|.##.....#.##.......##...##....##..........##.....#..##.####..#.####..##|
|.########.######..##.....#....##...######.########..##.##.##.#.##.##.##|
|.##.....#.##......########....##..........##........##.##..###.##..####|
|.##.....#.##......##.....#....##..........##........##.##...##.##...###|
|.##.....#.#######.##.....#....##..........##.......###.##....#.##....##|
'-----------------------------------------------------------------------'

usage: heat train [-h] [--domain DOMAIN] [--boundary BOUNDARY] [--model MODEL]
                  [-l LR] [--epochs EPOCHS] [--every EVERY]

options:
  -h, --help            show this help message and exit
  --domain DOMAIN       Path of domain data file
  --boundary BOUNDARY   Path of boundary data file
  --model MODEL         Path of model file
  -l LR, --lr LR, --learning-rate LR
                        Learning rate for the optimizer
  --epochs EPOCHS       Number of training epochs
  --every EVERY         Print result for every n epochs
```
### Inference
```
.-----------------------------------------------------------------------.
|.##.....#.#######....###...#######........########.###.##....#.##....##|
|.##.....#.##........##.##.....##..........##.....#..##.###...#.###...##|
|.##.....#.##.......##...##....##..........##.....#..##.####..#.####..##|
|.########.######..##.....#....##...######.########..##.##.##.#.##.##.##|
|.##.....#.##......########....##..........##........##.##..###.##..####|
|.##.....#.##......##.....#....##..........##........##.##...##.##...###|
|.##.....#.#######.##.....#....##..........##.......###.##....#.##....##|
'-----------------------------------------------------------------------'

usage: heat infer [-h] [--data DATA] [--model MODEL] [--output OUTPUT]

options:
  -h, --help       show this help message and exit
  --data DATA      Path of data file to perform inference
  --model MODEL    Path of model file
  --output OUTPUT  Path of output data file
```

## Validation <a name="res"></a>
  
### Square geometry 

<p align="center">
  <img src="https://github.com/314arhaam/heat-pinn/blob/main/assets/results_compare.png" title="pinn-vs-fdm">
</p> 

Temperature profiles:  
<p align="center">
  <img src="https://github.com/314arhaam/heat-pinn/blob/main/assets/profiles.png" title="profiles">
</p>

## Results <a name="res"></a>

### Doughnut geometry

<p align="center">
  <img src="https://github.com/314arhaam/heat-pinn/blob/main/assets/heat_pinn_doughnotts.png" title="doughnotts">
</p>

### Screw 

<p align="center">
  <img src="https://github.com/314arhaam/heat-pinn/blob/main/assets/screw.png" title="screw">
</p>

### Connecting Rod

<p align="center">
  <img src="https://github.com/314arhaam/heat-pinn/blob/main/assets/rod.png" title="conn-rod">
</p>

### Gear geometry

<p align="center">
  <img src="https://github.com/314arhaam/heat-pinn/blob/main/assets/symgear.png" title="symgear">
</p>

<p align="center">
  <img src="https://github.com/314arhaam/heat-pinn/blob/main/assets/asymgear.png" title="asymgear">
</p>


## Performance Comparison
Results obtained from a [9 layered DNN](https://github.com/314arhaam/heat-pinn/blob/main/assets/model_plot.png) (1000 epochs) and FDM code on a 100×100 grid. The FDM code is written in Python.
|**Method**|**Computation time (s)**|
|-|-|
|PINN|66.35|
|FDM|77.60|


## Note
This implementation is based on [Tensorflow 2.0](https://www.tensorflow.org/guide/effective_tf2) package and made possible by [Google Colabratory](https://colab.research.google.com) GPU.
