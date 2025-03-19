# Remember to adjust your student ID in meta.xml
import numpy as np
import torch
import random
import gym

# Global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def preprocess_state(obs):
    """
    Convert observation to a tensor for the neural network.
    """
    # Extract key components from the observation
    taxi_row, taxi_col, station0_row, station0_col, station1_row, station1_col, \
    station2_row, station2_col, station3_row, station3_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs
    
    # Create feature vector with meaningful information
    features = [
        # Taxi position and obstacle information
        taxi_row,
        taxi_col,
        obstacle_north,
        obstacle_south, 
        obstacle_east,
        obstacle_west,
    ]
    
    return torch.FloatTensor(features).to(DEVICE)

# Load the model
model = QNetwork(6, 6).to(DEVICE)
try:
    model.load_state_dict(torch.load("q_network.pt", map_location=DEVICE))
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback to random actions if model can't be loaded
    model = None

def get_action(obs):
    """
    Takes an observation as input and returns an action (0-5).
    Uses the trained Q-network to select the best action.
    """
    if model is None:
        return random.choice([0, 1, 2, 3, 4, 5])
    
    state_tensor = preprocess_state(obs)
    
    with torch.no_grad():
        q_values = model(state_tensor)
    
    return torch.argmax(q_values).item()
