# Deep-reinforcement-learning Controlled Fluid-Rigid Simulation

A python-based ball-shooting game of fluid-rigid simulation and autonomous control via Deep Reinforcement Learning. 

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
<!-- [Contributing](#contributing) -->
<!-- [License](#license) -->
<!-- [Acknowledgements](#acknowledgements) -->

## Features

- MLS-MPM method for fluid simulation
- Newton-Eulerian system for rigid body movement
- Lagrangian sampling for fluid-rigid coupling
- Robustness and Vram optimization for GPU parallelization
- Soft Actor-Critic deep reinforcement learning framework for tube control
- Meta Learning for problem generalization
- CNN Autoencoder pretraining for correlated field properties 


## Installation

### Requirements

* python 3.10

* jittor 1.3.6

* taichi 1.4

<!-- * tensorboardX 2.5.1 -->

* gym 0.26.2

* pytorch 2.1.0

### Platform
 Good Nvidia GPU (cuda) to run simulation, rendering and learning 


## Usage

### How to play with shooting tube
- A(&leftarrow;) & D(&rightarrow;): control the tube to move leftwards and rightwards
- W(&uparrow;) & S(&downarrow;): increase or decrease the ejecting speed
- right & left mouse click: control the tube to rotate clockwise and counter-clockwise
- R: reset the ball and tube

### Quick Start

#### Train

```
python train.py ./configs/train.json
```

#### Eval

```
python eval.py ./configs/eval.json
```
 -provided models in model_trial1

 
## Demo
![Scene_demo 1](https://github.com/Furkath/GL-Rendering/tree/master/demos/16-41-53.png)
![Scene_demo 2](https://github.com/Furkath/GL-Rendering/tree/master/demos/15-24-34.png)
