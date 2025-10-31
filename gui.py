import tkinter as tk
from tkinter import messagebox
from improvedC4 import *
from PIL import Image, ImageTk

class Connect4GUI:
    human_wins = 0
    agent_wins = 0
    #Objective 1A, the trained file from the improvedC4 code is used as the original state dict for the agent
    #However if the new file exists, we load that file instead of our original but wont modify the original
    if os.path.exists('AgentDict.pth'):
        agent.q_network.load_state_dict(torch.load('AgentDictNewFile.pth'))
        agent.target_network.load_state_dict(torch.load('AgentTargetNewFile.pth'))
        torch.serialization.add_safe_globals([deque])
        agent.memory = torch.load('AgentMemoryNewFile.pth', weights_only=False)
        print("New File exists, loading new file")
    else:
        agent.q_network.load_state_dict(torch.load('AgentDictTesting.pth'))
        agent.target_network.load_state_dict(torch.load('AgentTargetTesting.pth'))
        torch.serialization.add_safe_globals([deque])
        agent.memory = torch.load('AgentMemoryTesting.pth', weights_only=False)
        print("New File doesn't exist, loading old file")

    #Initializng all our variables, most importantly multiplayer_mode and player_role
    #These parameters easily allow us to switch between multiplayer and single player mode using the same code for the vs agent
    def __init__(self, root, multiplayer_mode=False, player_role=1):
        self.root = root
        self.root.title("Connect 4")
        self.root.configure(bg="blue")
        self.game = Connect4()
        self.buttons = []
        self.multiplayer_mode = multiplayer_mode
        self.player_role = player_role
        self.last_move = None
        self.create_widgets()
        if not self.multiplayer_mode and self.player_role == 2:
            self.agent_move()

    def create_widgets(self):
        frame = tk.Frame(self.root, bg="blue")
        frame.pack(pady=10)
        #Creating a label to show which player is either red or yellow and displayed on the top of the window
        bot_label = tk.Label(frame, text="Bot: Red", fg="red", bg="blue", font=("Arial", 14, "bold"))
        bot_label.grid(row=0, column=0, columnspan=3, pady=10)
        human_label = tk.Label(frame, text="Human: Yellow", fg="gold", bg="blue", font=("Arial", 14, "bold"))
        human_label.grid(row=0, column=4, columnspan=3, pady=10)
        #creating a resign button that allows the human player to quickly exit the game sending them to the game end window
        resign_button = tk.Button(frame, text="Resign", bg="yellow", command=lambda: self.end_game("Human Resigns!"), width=20, height=2)
        resign_button.grid(row=0, column=7, pady=10)
        #Creating the connect 4 board canvas 700x600 because we want to match the dimensions of a connect 4 board
        self.canvas = tk.Canvas(frame, width=700, height=600, bg="blue")
        self.canvas.grid(row=1, column=0, columnspan=self.game.columns)
        #Creating buttons for each column to which calls the human_insert function to drop the piece in the column
#Objective 2AII, 7x6 gameboard window must be displayed with all column elements shown
        for col in range(self.game.columns): #Iterates from 1 to 7 to create the buttons for each column
            button = tk.Button(frame, text=str(col), command=lambda c=col: self.human_insert(c), width=10, height=2, bg="yellow", fg="black", font=("Arial", 10, "bold"))
            button.grid(row=2, column=col, pady=5)
            self.buttons.append(button)
        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all") #Deletes the current board and redraws the board with the new pieces
        #Creates a 7x6 board with circles of diameters 97 (this is to make sure the circles do not overlap each other when creating the board)
        #Board dimensions are 700x600 so having diameter of 100 would make the gaps between too thin which isnt like a usual c4 board
        for r in range(self.game.rows):
            for c in range(self.game.columns):
                x1 = c * 100
                y1 = r * 100
                x2 = x1 + 97
                y2 = y1 + 97
                color = "black"
                if self.game.board[r][c] == 1: #Checks numpy board to see if there are are any 1s or 2s in the board and assigns the color accordingly
                    color = "red"
                elif self.game.board[r][c] == 2:
                    color = "yellow"
                self.canvas.create_oval(x1, y1, x2, y2, fill=color, outline=color)
        #Highlighting last move: user feedback
        if self.last_move:
            r, c = self.last_move
            #print(r)
            x1 = c * 100
            y1 = r * 100
            x2 = x1 + 97
            y2 = y1 + 97
            self.canvas.create_oval(x1,y1,x2,y2, outline="orange", width=5) #Creates a green outline around the last move made by the player
    def agent_move(self): 
        state = self.game.get_state()
        self.agent_action = agent.choose_action(state, self.game.legal_actions(), training=False) #Epislon set to zero
        row = self.game.drop_piece(self.agent_action) #Agent automatically places pieces without needing to interact with buttons
        self.last_move = (row, self.agent_action)
        self.draw_board()

        done = (not self.game.legal_actions()) or self.game.is_winning_move(1) 
        next_state = self.game.get_state()
        #Checks if game is completed after the agent's move and rewards the agent accordingly
#Objective 5A, model should append its experiences with the player and train on that data
#Objective 3CI, when the board is complete a message is displayed to the user showing who won
#Objective 3CII, when the board is complete should count the win rate and display game end window
        if done:
            reward = 1.0 if self.game.is_winning_move(1) else 0.5
            Connect4GUI.agent_wins += 1
            #Agent stores every turn in its memory and trains on the data whenever the game properly ends
            agent.store_experience(state, self.agent_action, reward, next_state, done)
            print("Agent training!")
            agent.train()
            torch.save(agent.q_network.state_dict(), 'AgentDictNewFile.pth') 
            torch.save(agent.target_network.state_dict(), 'AgentTargetNewFile.pth')
            torch.save(agent.memory,'AgentMemoryNewFile.pth') 
            self.end_game("Agent Wins") if self.game.is_winning_move(1) else self.end_game("Draw")
            return
        self.game.switch_player()

    def human_insert(self, col): 
        state = self.game.get_state()
        if col in self.game.legal_actions():
            row = self.game.drop_piece(col)
            self.last_move = (row, self.agent_action) 
            self.draw_board()
            next_state = self.game.get_state()
            done = (not self.game.legal_actions()) or self.game.is_winning_move(2) or self.game.is_winning_move(1) 
            if done:
                if self.multiplayer_mode:
                    self.end_game(f"Player {self.game.player_turn} Wins!")
                else: 
                    reward = -1.0 if self.game.is_winning_move(2) else 0.5
                    Connect4GUI.human_wins += 1
                    agent.store_experience(state, self.agent_action, reward, next_state, done)
                    print("Agent training!")
                    agent.train()
                    #implement saving to the new pth file
                    torch.save(agent.q_network.state_dict(), 'AgentDictNewFile.pth') #Saves the model dictionary and the target network dictionary
                    torch.save(agent.target_network.state_dict(), 'AgentTargetNewFile.pth')
                    torch.save(agent.memory,'AgentMemoryNewFile.pth') 
                    print("Saved model and memory (training)")
                    self.end_game("Human Wins") if self.game.is_winning_move(2) else self.end_game("Draw")
                return
        
        # Small negative reward to encourage faster wins
        if self.multiplayer_mode:
            self.game.switch_player()
        else:
            reward = -0.05
            agent.store_experience(state, self.agent_action, reward, next_state, done)
            self.game.switch_player()
            self.agent_move()

    def end_game(self, message): #closes the current window and opens a the game end window to display who won
        self.root.destroy()
        end_game(message)


def start_game(multiplayer_mode=False, player_role=1): #Creating the main game window and passing the multiplayer mode to the Connect4GUI class
    root = tk.Tk()
    app = Connect4GUI(root, multiplayer_mode=multiplayer_mode, player_role=player_role)
    root.mainloop()


def end_game(message): #Creating the game end window to display who won and the win rate
    end_root = tk.Tk()
    end_root.title("Connect 4 End")
    end_root.geometry("400x300")
    end_root.configure(bg="blue")

    frame = tk.Frame(end_root, bg="blue")
    frame.pack(pady=20)
    result_label = tk.Label(frame, text=message, bg="blue", fg="white", font=("Arial", 16, "bold"))
    result_label.pack(pady=10)
    win_rate = f"Human Wins: {Connect4GUI.human_wins} | Agent Wins: {Connect4GUI.agent_wins}"
    win_rate_label = tk.Label(frame, text=win_rate, bg="blue", fg="white", font=("Arial", 12))
    win_rate_label.pack(pady=10)
#Objective 5B, win rate displayed as percentage on game end window
    win_rate = "Human Win_rate:" , (Connect4GUI.human_wins / (Connect4GUI.agent_wins + Connect4GUI.human_wins)) * 100 if (Connect4GUI.agent_wins + Connect4GUI.human_wins) > 0 else 0.0,"%"
    win_rate_label = tk.Label(frame, text=win_rate, bg="blue", fg="white", font=("Arial", 12))
    win_rate_label.pack(pady=10)
#Objective 6A, play again button on game end window allowing the user to quickly play again
    play_button = tk.Button(frame, text="Play Again?", bg="yellow", command=lambda: [end_root.destroy(), selectplayer(), agent.train()], width=20, height=2)
    play_button.pack(pady=5)
    exit_button = tk.Button(frame, text="To Main Menu", bg="yellow", command=lambda: [end_root.destroy,main_menu()], width=20, height=2)
    exit_button.pack(pady=5)
#Objective 2BIII, reset agent button should only work given there is a seperate file created other than the original
def reset_agent():
    if os.path.exists('AgentDictNewFile.pth'): #Checks if new file exists and removes it else outputs not found error
        os.remove('AgentDictNewFile.pth')
        os.remove('AgentTargetNewFile.pth')
        os.remove('AgentMemoryNewFile.pth')
        print("AgentDictNewFile.pth, AgentMemoryNewFile.pth and AgentTargetNewFileremoved")
    else:
        print("AgentDictNew.pth not found")

#Objective 2AI, allows users to pick between player 1 or player 2 against the agent
#Objective 6AI
def selectplayer():
    select_root = tk.Tk()
    select_root.title("Select Player")
    select_root.geometry("400x300")
    select_root.configure(bg="blue")

    frame = tk.Frame(select_root, bg="blue")
    frame.pack(pady=20)

    label = tk.Label(frame, text="Select player", bg="blue", fg="white", font=("Arial", 16, "bold"))
    label.pack(pady=10)
    #Creation of both player 1/2 buttons assigning the player role to either 1 or 2 (red or yellow)
    player1_button = tk.Button(frame, text="Pick Player 1", bg="yellow", command=lambda: [select_root.destroy(), start_game(player_role=1)], width=20, height=2)
    player1_button.pack(pady=5)
    player2_button = tk.Button(frame, text="Pick Player 2", bg="yellow", command=lambda: [select_root.destroy(), start_game(player_role=2)], width=20, height=2)
    player2_button.pack(pady=5)
    exit_button = tk.Button(frame, text="Exit Game", bg="yellow", command=select_root.destroy, width=20, height=2)
    exit_button.pack(pady=5)
    select_root.mainloop()

def main_menu():
    menu_root = tk.Tk()
    menu_root.title("Connect 4 Menu")
    menu_root.geometry("400x300")
    menu_root.configure(bg="blue")

    frame = tk.Frame(menu_root, bg="blue")
    frame.pack(pady=20)

    logo_canvas = tk.Canvas(frame, width=200, height=50, bg="blue", highlightthickness=0)
    logo_canvas.pack(pady=10)

    colors = ["red", "yellow"]
    #creates alternating red and yellow circles representing the connect 4 logo
    for i in range(4):
        x1 = i * 50
        y1 = 0
        x2 = x1 + 45
        y2 = 45
        logo_canvas.create_oval(x1, y1, x2, y2, fill=colors[i % 2], outline=colors[i % 2])
#Objective 2A, creating the main menu with play, multiplayer and exit buttons
    play_button = tk.Button(frame, text="Play", bg="yellow", command=lambda: [menu_root.destroy(), selectplayer()], width=20, height=2)
    play_button.pack(pady=5)
    #I created a multiplayer mode which was one of the primary objectives from the questionaire
    multiplayer_button = tk.Button(frame, text="Multiplayer", bg="yellow", command=lambda: [menu_root.destroy(), start_game(multiplayer_mode=True)], width=20, height=2)
    multiplayer_button.pack(pady=5)
    #Functions same way as the exit button in the end_game function
    exit_button = tk.Button(frame, text="Exit", bg="yellow", command=menu_root.destroy, width=20, height=2)
    exit_button.pack(pady=5)
    #Reset the agent button to allow the user to set the agent back to its original state
    reset_button = tk.Button(frame,text="Reset Agent", bg="red", command=lambda: [reset_agent()], width=20, height=2)
    reset_button.pack(pady=5)

    menu_root.mainloop()

if __name__ == "__main__":
    main_menu()
