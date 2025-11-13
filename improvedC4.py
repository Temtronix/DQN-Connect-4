import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import math
import matplotlib.pyplot as plt
import os
import time

#Setting device (cpu/gpu) for training the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========================= connect4 logic =========================
#Objective 3C, program should identify game win states/draw states
#Objective 6AII, gameboard window should show an empty grid when play again button pressed
class Connect4:
    def __init__(self): #Board is already intiialized to zeroes meaning we wont need to reset the board each time
        self.rows = 6
        self.columns = 7
        self.board = np.zeros((self.rows, self.columns), dtype=int) #Numpy 7 by 6 board consisting of 1,2 or 0s
        self.player_turn = 1  

    def drop_piece(self, col):
        #Checks each column from the bottom up to see if there is an empty space and places the piece there
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.player_turn
                return row
        return False

    def is_winning_move(self, player):#Function to check for every 4-in-a-row in the board
        for r in range(self.rows):
            for c in range(self.columns - 3):
                if all(self.board[r, c + i] == player for i in range(4)):
                    return True
        for c in range(self.columns):
            for r in range(self.rows - 3):
                if all(self.board[r + i, c] == player for i in range(4)):
                    return True
        for r in range(self.rows - 3):
            for c in range(self.columns - 3):
                if all(self.board[r + i, c + i] == player for i in range(4)):
                    return True
        for r in range(3, self.rows):
            for c in range(self.columns - 3):
                if all(self.board[r - i, c + i] == player for i in range(4)):
                    return True
        return False

    def get_state(self): 
        #returns flattened board state
        return self.board.flatten()

    def legal_actions(self):
        #It validates a column based on whether its topmost row is empty or not
        return [c for c in range(self.columns) if self.board[0, c] == 0]

    def switch_player(self):
        self.player_turn = 3 - self.player_turn  # toggles between 1 and 2

# ========================= DQN Implementation =========================
class DQN(nn.Module):
    def __init__(self, outputs=7):
        super().__init__()
        # Convolutional layers with 3x3 kernels since so we can identify complex patterns up to 3-in-a-row
        # Input is a 7x6 board which is reshaped into 1x7x6 for the convolutional layers of the network
        # Padding is used so we can retain the board dimensions after convolution and edges are accounted for more often
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  #32 filters that detect across the board for regular piece patterns
            nn.LeakyReLU(), #Changed from ReLU to leaky relu so we dont have 'dead' neurons 
            nn.Conv2d(32, 64, kernel_size=3, padding=1), #Feature maps of the 32 filters are passed to 64 filters to detect even more patterns
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  
            nn.LeakyReLU()
        )
        # The fully connected layers which process the features extracted from the convolutional layers
        # Input size is 6x7x64 and flattened to match the output of the last CNN layer with 64 feature maps of 7x6 boards
        linear_input_size = 6 * 7 * 64
        self.fc_layers = nn.Sequential(
            nn.Linear(linear_input_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, outputs)
        )

    def forward(self, x): # Forward pass through the network
        # Reshape the input to match the expected input shape for the convolutional layers
        x = x.view(-1, 1, 6, 7).to(device) #Converts the input to a 4D tensor with shape (batch_size, channels, height, width)
        # Pass through the convolutional layers and then flatten the output
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) 
        return self.fc_layers(x) #Returns the q-values for each action in the game


class DQNAgent:
    def __init__( #parameters for the q-learning agent and the bellman equation
        self,
        learning_rate = 0.001, #This heavily influences how much the weights are updated when training. A small value is ideal since we want a stable growth
        gamma = 0.95, #otherwise known as discount factor which determines how much future rewards have on the agent
        epsilon = 1.0, #used for epsilon greedy exploration/ determines the randomness of our agent
        epsilon_decay = 2000,
        min_epsilon = 0.05,
        batch_size = 64 #Used for sampling the memory queue and is important for passing the training data to the network due to its dimensions
    ):
        self.q_network = DQN().to(device) 
        self.target_network = DQN().to(device) 
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.memory: deque = deque(maxlen=25000) #Queue that stores the last 250000 experiences of the model
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.lossfunction = nn.MSELoss() #Mean square error loss function which is used to calculate the difference between the predicted and target Q-values
        self.gamma = gamma #Discount factor which determines importance of immediate rewards over future rewards
        #Epislon greedy exploration parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size

    def choose_action(self, state, legal_actions, training = True) :
        #Epsilon greedy strategy used to determine the action agent takes. If not training the model will always choose the highest q-value option
        if training and random.random() < self.epsilon and legal_actions:
            return random.choice(legal_actions)
        else:
            if legal_actions:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                # Only uses Q-values that have a legal action pair
                action_values = {action: q_values[0, action].item() for action in legal_actions}
                #print(action_values)
                return max(action_values, key=action_values.get)
            else:
                raise ValueError("No legal actions available.")

    def store_experience(self, state, action, reward, next_state, done):
        #stores the experiences in the memory queue which is later saved and sampled to train the model
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        #Once agent has enough experiences, we fetch a random sample of 64 experiences to train against
        if len(self.memory) < self.batch_size:
            return 

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device) #convert the training data into tensors for the neural network and run using GPU
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device) #Dones is used to determine whether we have reached a terminal state

        # Current Q-values for the taken actions
        current_q_values = self.q_network(states).gather(1, actions).squeeze()

        # Compute target Q-values using the target network (unaffected by the gradients)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values #bellman equation

        loss = self.lossfunction(current_q_values, target_q_values) #applies MSE to each currentq and targetq pair
        self.optimizer.zero_grad()
        loss.backward() #computes derivatives of the loss function with respect to the weights (dL/dw) for every parameter x
        self.optimizer.step() #updates the weights x = x - lr * dL/dw

    def update_target_network(self):
        #updates target weights with the weights of the qnetwork
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        # I have changed the method of decay_epsilon to use exponents as it produces a much more stable decay over a longer eps count
        self.epsilon = max(self.min_epsilon, self.min_epsilon + (self.epsilon - self.min_epsilon) * math.exp(-1. * (1 / self.epsilon_decay)))

#Seperate winning board function for the improved agent to check for winning moves a step ahead of the current game
def is_winning_board(board, player, rows= 6, columns= 7) :
    for r in range(rows):
        for c in range(columns - 3):
            if all(board[r, c + i] == player for i in range(4)):
                return True
    for c in range(columns):
        for r in range(rows - 3):
            if all(board[r + i, c] == player for i in range(4)):
                return True
    for r in range(rows - 3):
        for c in range(columns - 3):
            if all(board[r + i, c + i] == player for i in range(4)):
                return True
    for r in range(3, rows):
        for c in range(columns - 3):
            if all(board[r - i, c + i] == player for i in range(4)):
                return True

    return False

def improved_agent(game): #Heuristic algorithm to force the model to play more aggressively and identify winning/blocking moves
    legal_moves = game.legal_actions()
    current_player = game.player_turn
    opponent = 3 - current_player
    for move in legal_moves: 
        board_copy = game.board.copy() #Copy of the current board is created we can simulate a move 1 turn ahead and check for wins
        for row in range(game.rows - 1, -1, -1):
            if board_copy[row, move] == 0:
                board_copy[row, move] = current_player
                break
        if is_winning_board(board_copy, current_player, game.rows, game.columns) and random.random() < 0.5:
            return move
        
    for move in legal_moves: #Similarly we also check 1 move ahead however it is for the opponent being able to win so we block it
        board_copy = game.board.copy()
        for row in range(game.rows - 1, -1, -1):
            if board_copy[row, move] == 0:
                board_copy[row, move] = opponent
                break
        if is_winning_board(board_copy, opponent, game.rows, game.columns):
            return move

    return random.choice(legal_moves)



# ========================= Training and Evaluation =========================
agent = DQNAgent() #Instantiation of the agent class
def win_rate_test(num_games= 100):
    #Evaluating the agent's win rate by playing 100 games against a random opponent
    #avg_win_moves is the average number of moves the agent itself has taken that led to a victory
    #set the epsilon value to zero by making training = False (testing the agent as its peak performance)
    win_moves_taken_list = [] #I used a list to store number of moves played for it to win then used np.mean to calculate its average
    wins = 0
    for _ in range(num_games):
        env = Connect4()
        moves_taken = 0

        while True:
            state = env.get_state()
            legal = env.legal_actions()
            action = agent.choose_action(state, legal, training=False)
            env.drop_piece(action)
            moves_taken += 1
            if env.is_winning_move(1):
                wins += 1
                win_moves_taken_list.append(moves_taken)
                break
            if not env.legal_actions():
                break
            env.switch_player()
            #opp_action = improved_agent(env) #heuristic algorithm
            opp_action = random.choice(env.legal_actions()) #playing against random bot
            env.drop_piece(opp_action)
            env.switch_player()

            if env.is_winning_move(2) or not env.legal_actions():
                break

    avg_moves = np.mean(win_moves_taken_list) if win_moves_taken_list else 0
    return wins / num_games, avg_moves

def train_dqn(episodes= 24000) :
    performance_history = []  # Each element: [episode, win_rate, avg_moves] will be later used to plot a graph

    if os.path.exists('AgentDict.pth') and os.path.exists('AgentTarget.pth'): 
        #checks if file path exists and loads both the target and predicted dictionaries for the model
        agent.q_network.load_state_dict(torch.load('AgentDict.pth'))
        agent.target_network.load_state_dict(torch.load('AgentTarget.pth'))
        print("Sucessfully loaded agent model")

    if os.path.exists('AgentMemory.pth'): #checks if the memory (queue) file exists and loads it
        torch.serialization.add_safe_globals([deque])
        agent.memory = torch.load('AgentMemory.pth', weights_only=False)
        print("sucessfully loaded memory")

    start_time = time.time()
    for episode in range(episodes):
        game = Connect4()
        state = game.get_state()
        done = False

        # Call win rate test every 20 episodes which are not included in the agent's training
        if episode % 20 == 19:
            win_rate, avg_moves = win_rate_test(num_games=100)
            performance_history.append([episode + 1, win_rate, avg_moves])
            if episode % 200 == 199:
                #Outputting the epsilon, average move count, win rate every 200 episodes to see the agent's progress
                print(f"Episode {episode+1} | Win rate: {win_rate:.2f} | Avg moves (wins): {avg_moves:.2f} | Epsilon: {agent.epsilon:.3f} | Time: {(time.time() - start_time):.2f}s")
                start_time = time.time()

        while not done:
            legal = game.legal_actions()
            try:
                action = agent.choose_action(state, legal, training=True)
            except ValueError:
                break  # No legal moves
            game.drop_piece(action)

            # Check if agent wins or draw occurs
            done = (not game.legal_actions()) or game.is_winning_move(1) or game.is_winning_move(2)
            next_state = game.get_state()
            if done:
                reward = 1.0 if game.is_winning_move(1) else 0.5
                agent.store_experience(state, action, reward, next_state, done)
                break

            # Opponent move
            game.switch_player()
            #opp_action = improved_agent(game)
            opp_action = random.choice(game.legal_actions())
            game.drop_piece(opp_action)
            game.switch_player()
            next_state = game.get_state()

            done = (not game.legal_actions()) or game.is_winning_move(1) or game.is_winning_move(2)
            if done:
                # If opponent wins, assign negative reward to agent's last move
                reward = -1.0 if game.is_winning_move(2) else 0.5
                agent.store_experience(state, action, reward, next_state, done)
                break

            # Small negative reward to encourage faster wins for the agent
            reward = -0.05
            agent.store_experience(state, action, reward, next_state, done)

            state = next_state
            agent.train()

        agent.decay_epsilon()
        if episode % 10 == 0:
            agent.update_target_network() #Target is only updated for every 10 episodes to ensure the model is stable and not overfitting to the training data
    #Objective 1: Saving the predefined model and memory queue to a file so we can load it later for the GUI file
    torch.save(agent.q_network.state_dict(), 'AgentDict.pth') #Saves the model dictionary
    torch.save(agent.target_network.state_dict(), 'AgentTarget.pth') #Saves the target model dictionary
    torch.save(agent.memory,'AgentMemory.pth') #Saves the memory of the agent
    print("Saved model and memory (training)")


    #Plot training history (plotting moving averages against the bot) part of the objectives but helps to understand our models growth
    plot_win_history = np.array(performance_history) #An array of win rates and moves taken

    if plot_win_history.size:
        plt.figure(figsize=(12, 5)) #Creating a figure with 2 subplots to plot the win rate and average moves taken against episode count
        plt.subplot(1, 2, 1)
        #Cyan line for every win rate calculated and blue for the line of the moving averages
        plt.plot(plot_win_history[:, 0], plot_win_history[:, 1], 'c-', label='Win Rate') 
        moving_avg = np.convolve(plot_win_history[:, 1], np.ones(20)/20, mode='valid') #Finds average of every 20 episodes
        plt.plot(plot_win_history[19:, 0], moving_avg, 'b-', label='Moving Average') 
        plt.xlabel("Episode")
        plt.ylabel("Win Rate")
        plt.legend()
        plt.title("Agent Win Rate vs Episodes")

        plt.subplot(1, 2, 2) 
        plt.plot(plot_win_history[:, 0], plot_win_history[:, 2], 'c-', label='Avg Moves (Wins)')
        moving_avg_moves = np.convolve(plot_win_history[:, 2], np.ones(20)/20, mode='valid')
        plt.plot(plot_win_history[19:, 0], moving_avg_moves, 'b-', label='Moving Average')
        plt.xlabel("Episode")
        plt.ylabel("Avg Moves Taken")
        plt.legend()
        plt.title("Agent Win Efficiency vs Episodes")

        plt.tight_layout() #Adjusts the subplots to fit into the figure area.
        plt.show()


# ========================= testing against human =========================
#Code here is not part of the objectives but is used as a simple way for me to measure the agents performance
def play_human():
    #Loading the model and memory queue from training phase
    agent.q_network.load_state_dict(torch.load('AgentDict.pth'))
    torch.serialization.add_safe_globals([deque])
    agent.memory = torch.load('AgentMemory.pth', weights_only=False)
    print("Loaded agent model and memory")

    game = Connect4()
    print("Welcome to Connect4!")
    while True:
        print("\nCurrent board:")
        print(game.board)
        legal = game.legal_actions()
        state = game.get_state()
        agent_action = agent.choose_action(state, game.legal_actions(), training=False)
        print(f"Agent chooses column {agent_action}")
        game.drop_piece(agent_action)
        if game.is_winning_move(1):
            print("\nFinal board:")
            print(game.board)
            print("Agent wins")
            break
        if not game.legal_actions():
            print("Draw")
            break
        print(game.board)
        game.switch_player()
        action = None
        while action not in legal:
            try:
                action = int(input(f"Your move (choose column from {legal}): "))
                if action not in legal:
                    print("Illegal move. Please choose from:", legal)
            except ValueError:
                print("Invalid input. Please enter an integer.")
        game.drop_piece(action)
        if game.is_winning_move(2):
            print("\nFinal board:")
            print(game.board)
            print("Human Wins")
            break
        if not game.legal_actions():
            print("Draw")
            break
        game.switch_player()

if __name__ == "__main__":
    mode = input("Enter 'train' to train the agent or 'play' to play against the agent: ").lower()
    if mode == "train":
        train_dqn(episodes=24000)
        play_human()
    elif mode == "play":
        play_human()
    else:
        print("Invalid mode selected.")
