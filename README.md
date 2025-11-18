# Connect4 Deep Q-Learning project using PyTorch

## A fully functional Q-Learning project written in python documenting the stages of development

This project was created as part of my AQA A-level Computer Science NEA with the purpose of creating an adaptive algorithm targeted towards a user's skill in connect4. 

## Project Objectives

* Create an adaptive Connect 4 AI using Q-Learning/Deep Q-Networks
* Allow the agent to match the player's skill level by training on their moves
* Build a full GUI with game board rendering, player selection, win-screen and stats, resettable AI model
* Model should assess board position and determine whether it should play more aggressively or not
* After a fully finished game, the win rate of the agent/player should be displayed hyperparameters
* Save/load trained models so progress is not lost

## How the AI works

### Q-Learning Foundation
* The game state is encoded as a flattened 7x6 board
* Actions represent dropping a piece in one of seven columns
* Rewards are assigned for: +1 win, -1 loss, 0 ongoing game
* Exploration using the epsilon-greedy strategy that gradually reduces randomness as training progresses

### Deep Neural Network (DQN)
* Initially I created a simple connect4 game using numpy arrays then i built upon that using an RL-algorithm called Q-learning
* Q-learning (tabular) was inefficient since it only found discrete values for each possible state when there are 4 trillion
* With the help of pytorch, I was able to create a CNN using 3 layers with 3x3 kernels and used LeakyReLU between layers
* The features from the Convolutional Layers are sent to the fully connected layers to find a weighting

## Performance Tracking
### During training
* Win rate is logged every 20 episodes
* Average move count is tracked
* Moving averages plotted using Matplotlib

### Results
* The DQN shows strong improvement over time
* Agent can set up and execute dual threats


## GPU acceleration for Network (using CUDA)
* Used Pytorch to uttilize my NVIDIA GPU for processing rather than my CPU
* x6 faster using CUDA than without
