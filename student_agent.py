# Remember to adjust your student ID in meta.xml
import numpy as np
import torch
import random
import gym
import pickle
import os
from collections import Counter

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
        obstacle_north,
        obstacle_south, 
        obstacle_east,
        obstacle_west,
    ]
    
    return torch.FloatTensor(features).to(DEVICE)

# # Load the model
model = QNetwork(4, 6).to(DEVICE)
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
    
    try:
        state_tensor = preprocess_state(obs)
        
        # Check if state_tensor is None or contains NaN values
        if state_tensor is None or torch.isnan(state_tensor).any():
            print("Warning: Invalid state tensor detected")
            return random.choice([0, 1, 2, 3, 4, 5])
        
        with torch.no_grad():
            q_values = model(state_tensor)
            
            # Check if q_values contains NaN or very large values
            if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                print("Warning: Invalid Q-values detected")
                return random.choice([0, 1, 2, 3, 4, 5])
        
        return torch.argmax(q_values).item()
    except Exception as e:
        print(f"Error in get_action: {e}")
        return random.choice([0, 1, 2, 3, 4, 5])

# implement q_table version
# def get_action(obs):
#     q_table = pickle.load(open("q_table.pkl", "rb"))
#     state_key = tuple(preprocess_state(obs).cpu().numpy())

#     if state_key not in q_table:
#         return random.choice([0, 1, 2, 3, 4, 5])
    
#     q_values = q_table[state_key]

#     return np.argmax(q_values)

# Ensemble version

# Load ensemble models
# NUM_MODELS = 5  # Match the number used in training
# models = []
# model_path_prefix = "q_network"

# # Try to load ensemble models
# try:
#     for i in range(NUM_MODELS):
#         model_path = f"{model_path_prefix}_{i}.pt"
#         if os.path.exists(model_path):
#             model = QNetwork(6, 6).to(DEVICE)
#             model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#             model.eval()
#             models.append(model)
    
#     if not models:  # If no ensemble models found, try loading single model
#         model = QNetwork(6, 6).to(DEVICE)
#         model.load_state_dict(torch.load("q_network.pt", map_location=DEVICE))
#         model.eval()
#         models.append(model)
    
#     print(f"Successfully loaded {len(models)} model(s)")
# except Exception as e:
#     print(f"Error loading models: {e}")
#     models = []

# def get_action(obs):
#     """
#     Takes an observation as input and returns an action (0-5).
#     Uses ensemble voting from multiple models to select the best action.
#     Falls back to Q-table or random actions if models aren't available.
#     """
#     if models is None:
#         return random.choice([0, 1, 2, 3, 4, 5])
    
#     try:
#         state_tensor = preprocess_state(obs)
        
#         # Check if state_tensor is None or contains NaN values
#         if state_tensor is None or torch.isnan(state_tensor).any():
#             print("Warning: Invalid state tensor detected")
#             return random.choice([0, 1, 2, 3, 4, 5])
        
#         # If we have models, use ensemble voting
#         if models:
#             actions = []
#             with torch.no_grad():
#                 for model in models:
#                     q_values = model(state_tensor)
                    
#                     # Check if q_values contains NaN or very large values
#                     if torch.isnan(q_values).any() or torch.isinf(q_values).any():
#                         continue
                    
#                     actions.append(torch.argmax(q_values).item())
            
#             # Return the most common action (voting)
#             if actions:
#                 # print(actions)
#                 # print(Counter(actions).most_common(1)[0][0])
#                 return Counter(actions).most_common(1)[0][0]
#                 # return actions

#         # Fallback to random action if everything else fails
#         return random.choice([0, 1, 2, 3, 4, 5])
    
#     except Exception as e:
#         print(f"Error in get_action: {e}")
#         return random.choice([0, 1, 2, 3, 4, 5])
