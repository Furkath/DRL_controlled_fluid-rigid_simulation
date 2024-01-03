# Fluid directed rigid body control in 2-3D spaces

## Requirements

* python 3.10

* jittor 1.3.6

* taichi 1.4

<!-- * tensorboardX 2.5.1 -->

* gym 0.26.2

* pytorch 2.1.0

## Platform
 Good GPU to run simulation, rendering and learning 

## How to play with shooting tube
- W & S: control the tube to move leftwards and rightwards
- right & left mouse click: control the tube to rotate clockwise and counter-clockwise
- R: reset the ball and tube

## Quick Start

### Train

```
python train.py ./configs/train.json
```

### Eval

```
python eval.py ./configs/eval.json
```
 -provided models in model_trial1
