# Connect4 Deep Q-Learning project using PyTorch

## A fully functional Q-Learning project written in python documenting the stages of development

This project was created as part of my AQA A-level Computer Science NEA with the purpose of creating an adaptive algorithm targeted towards a user's skill in connect4. 

## Project Objectives

* Predefined model/memory written to file
* Must allow a user to select game preference
* Must successfully display a connect 4 game board
* Model should assess board position and determine whether it should play more aggressively or not
* After a fully finished game, the win rate of the agent/player should be displayed hyperparameters
* Allow user to play the game multiple times after 

## Overview

* Initially I created a simple connect4 game using numpy arrays then i built upon that using an RL-algorithm called Q-learning
* Q-learning (tabular) was inefficient since it only found discrete values for each possible state when there are 4 trillion
* With the help of pytorch, I was able to create a CNN using 3 layers with 3x3 kernels and used LeakyReLU between layers
* The features from the Convolutional Layers are sent to the fully connected layers to find a weighting
