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
* I plotted a graph for the agent against a random bot and a simple connect 4 heuristic algorithm that blocked winning moves
* Against the heuristic algorithm

<img width="936" height="398" alt="image" src="https://github.com/user-attachments/assets/fbd5f63e-3bef-4da7-8354-8aed37a1dbe6" />

* Against the random bot

<img width="940" height="445" alt="image" src="https://github.com/user-attachments/assets/760a011f-55a9-4887-8a8f-df07f58a3c1f" />

## GPU acceleration for Network (using CUDA)
* Used Pytorch to uttilize my NVIDIA GPU for processing rather than my CPU
* x6 faster using CUDA than without

<img width="272" height="33" alt="image" src="https://github.com/user-attachments/assets/b75e45bb-76e9-4dce-97e2-563ac4792d50" />

### Using GPU
<img width="1028" height="153" alt="image" src="https://github.com/user-attachments/assets/eeb6e22d-5d40-4022-9f72-8e476f1ffdbf" />

### Using CPU
<img width="1014" height="69" alt="image" src="https://github.com/user-attachments/assets/23517c29-7696-4250-aeac-e88501b489ff" />

