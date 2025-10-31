import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class Connect4:
    def __init__(self):
        self.rows = 6
        self.columns = 7
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.player_turn = 1  

    def drop_piece(self, col):
        """ Drops a piece in the selected column if valid. """
        for row in range(5, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.player_turn
                return True
        return False

    def is_winning_move(self,player):
        """ Checks if the last move resulted in a win. """
        # Horizontal
        for c in range(self.columns - 3):
            for r in range(self.rows):
                if all(self.board[r, c + i] == player for i in range(4)):
                    return True

        # Vertical
        for c in range(self.columns):
            for r in range(self.rows - 3):
                if all(self.board[r + i, c] == player for i in range(4)):
                    return True

        # Positive diagonal
        for c in range(self.columns - 3):
            for r in range(self.rows - 3):
                if all(self.board[r + i, c + i] == player for i in range(4)):
                    return True

        # Negative diagonal
        for c in range(self.columns - 3):
            for r in range(3, self.rows):
                if all(self.board[r - i, c + i] == player for i in range(4)):
                    return True

        return False

    def get_state(self):
        return self.board.flatten()

    def legal_actions(self):
        """ Returns a list of valid columns for the next move. """
        return [c for c in range(self.columns) if self.board[0, c] == 0]

    def switch_player(self):
        """ Switches the turn to the other player. """
        self.player_turn = 3 - self.player_turn

# ========================= DQN Implementation =========================

class DQN(nn.Module):
    """ Neural Network to Approximate Q-values """

    def __init__(self, input_dim=42, output_dim=7): 
        super(DQN, self).__init__()
        self.first_layer = nn.Linear(input_dim, 128)  
        self.second_layer = nn.Linear(128, 128)  
        self.output_layer = nn.Linear(128, output_dim)  #

    def forward(self, x): 
        x = torch.relu(self.first_layer(x))
        x = torch.relu(self.second_layer(x))
        return self.output_layer(x) 

class DQNAgent:
    def __init__(self, learning_rate=0.01):
        self.q_network = DQN()
        self.target_network = DQN() 
        self.target_network.load_state_dict(self.q_network.state_dict()) # copy the weights from the qnetwork to target network to ensure they start identically
        self.target_network.eval()
        
        self.memory = deque(maxlen=5000)  #5000 experiences stored in memory for agent to learn from/Plan to store them in a file
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate) # Controls how much weights are adjusted during backpropagation
        self.lossfunction = nn.MSELoss() #Another parameter to control how much weights are adjusted during backpropagation
        self.discount_factor = 0.95  # Discount factor
        self.epsilon_greedychance = 1 
        self.epsilon_decay = 999769768  
        self.min_epsilon = 0.01 
        self.batch_size = 32

    def choose_action(self, state, legal_actions):
        """ Chooses an action using an epsilon-greedy exploration. """
        if random.random() < self.epsilon_greedychance:
            return random.choice(legal_actions) 
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0) #Flattens the board into a 1D array
            q_values = self.q_network(state_tensor)  # Predict Q-values/ foward function is called here
            print(q_values)
            action_values = {action: q_values[0, action].item() for action in legal_actions} 
            #debugging to see what moves the agent is most likely to take
            print(action_values)
            return max(action_values, key=action_values.get)  

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """ Trains the DQN using experiences from the experience buffer. """
        if len(self.memory) < self.batch_size: #Wont train till enough experiences
            return  

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch) 

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones) # we convert the boolean values into 1(True) and 0(False) for the bellmans equation

        current_q_values = self.q_network(states).gather(1, actions).squeeze() 
        with torch.no_grad(): #Doesnt compute the gradients for the target q values
            next_q_values = self.target_network(next_states).max(dim=1)[0] 
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values #Bellman equation to calculate the target q values
        print(actions, current_q_values, target_q_values)
        loss = self.lossfunction(current_q_values, target_q_values)
        print(loss) 
        self.optimizer.zero_grad() 
        loss.backward() #computes derivatives of the loss function with respect to the weights (dL/dw) for every parameter x
        self.optimizer.step() #updates the weights x = x - lr * dL/dw

    def update_target_network(self):
        """ Updates target network with new weights from Q-network. """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon_greedychance = max(self.min_epsilon, self.epsilon_greedychance * self.epsilon_decay)


# ========================= Training Loop =========================
agent = DQNAgent()
#Should create a file to store memory into a file after training
def train_dqn(episodes=10):
    for episode in range(episodes):
        game = Connect4()
        state = game.get_state()
        done = False
        states = {1: [], 2: []}

        while not done:
            legal_actions = game.legal_actions()
            action = agent.choose_action(state, legal_actions)
            game.drop_piece(action)
            next_state = game.get_state() #Current state after the agent has taken the action
            if game.is_winning_move(game.player_turn):
                reward = 1 
            else:
                reward = 0
            done = game.is_winning_move(game.player_turn) or len(game.legal_actions()) == 0
            states[game.player_turn].append((state, action, reward, next_state, done))
            if done:
                #updates q-values for self play as both player 1 and 2
                for player in [1,2]:
                    for s, a, r, ns, d in states[player]:
                        agent.store_experience(s, a, r, ns, d)
                        agent.train()
            else:
                state = next_state
                game.switch_player()

        agent.decay_epsilon()
        if episode % 100 == 0:
            agent.update_target_network()
            print(f"Episode {episode} - Epsilon: {agent.epsilon_greedychance:.3f}") 

# ========================= Agent vs player =========================
def playhuman():
    game = Connect4()
    while True:
        print(game.board)
        action = int(input("Enter column between 0-6: "))
        if action not in game.legal_actions():
            print("Invalid move. Try again.")
            continue
        game.drop_piece(action)
        state_before_agent = game.get_state()
        if game.is_winning_move(1):
            print("Human wins!")
            break
        if np.all(game.board != 0):
            print("It's a draw!")
            break
        game.switch_player()
        legal_actions = game.legal_actions()
        action = agent.choose_action(state_before_agent, legal_actions)
        game.drop_piece(action)
        if game.is_winning_move(2):
            print("Agent wins!")
            print("Agent played in column: ", action)
            break

        game.switch_player()
train_dqn()
playhuman()
