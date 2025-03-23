# # Remember to adjust your student ID in meta.xml
# import numpy as np
# import torch
# import random
# import gym
# import pickle
# import os
# from train_dqn import DQN_agent
from train_ensemble_dqn import DQN
import torch

# # from collections import Counter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = DQN(state_size=4, action_size=6, gamma=0.99, batch_size=128)
agent.load('q_network_ensemble_2.pt')

def preprocess_state(obs):
    # Extract key components from the observation
    taxi_row, taxi_col, station0_row, station0_col, station1_row, station1_col, \
    station2_row, station2_col, station3_row, station3_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs

    
    # Create feature vector with more meaningful relative information
    features = [
        # Obstacle information (binary)
        # taxi_row,
        # taxi_col,
        obstacle_north,
        obstacle_south, 
        obstacle_east,
        obstacle_west,
        
        # # Passenger and destination information (binary)
        # passenger_look,
        # destination_look,
        # distances_to_stations[0],
        # distances_to_stations[1],
        # distances_to_stations[2],
        # distances_to_stations[3]
    ]
    
    return torch.FloatTensor(features).to(DEVICE)

def get_action(obs):
    state_tensor = preprocess_state(obs)
    
    with torch.no_grad():
        q_values = agent.policy_net(state_tensor)
    
    return torch.argmax(q_values).item()

# # # Global variables
# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # class QNetwork(torch.nn.Module):
# #     def __init__(self, input_dim, output_dim):
# #         super(QNetwork, self).__init__()
# #         self.network = torch.nn.Sequential(
# #             torch.nn.Linear(input_dim, 128),
# #             torch.nn.GELU(),
# #             torch.nn.Linear(128, 64),
# #             torch.nn.GELU(),
# #             torch.nn.Linear(64, output_dim)
# #         )
    
# #     def forward(self, x):
# #         return self.network(x)

# passenger_station = None
# destination_station = None
# stations_not_visited = []
# i = 0

# def calculate_dist(a, b):
#     return abs(a[0] - b[0]) + abs(a[1] - b[1])

# def preprocess_state(obs):
#     """
#     Convert observation to a tensor for the neural network.
#     """
#     # Extract key components from the observation
#     taxi_row, taxi_col, station0_row, station0_col, station1_row, station1_col, \
#     station2_row, station2_col, station3_row, station3_col, \
#     obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
#     passenger_look, destination_look = obs

#     # update next state
#     taxi_row, taxi_col = obs[0:2]
#     cur_north = (taxi_row - 1, taxi_col)
#     cur_south = (taxi_row + 1, taxi_col)
#     cur_east = (taxi_row, taxi_col + 1)
#     cur_west = (taxi_row, taxi_col - 1)

#     if (obs[14] == 1 and passenger_station is None):
#         if (cur_north in stations_not_visited):
#             passenger_station = cur_north
#         elif (cur_south in stations_not_visited):
#             passenger_station = cur_south
#         elif (cur_east in stations_not_visited):
#             passenger_station = cur_east
#         elif (cur_west in stations_not_visited):
#                     passenger_station = cur_west
#         elif ((taxi_row, taxi_col) in stations_not_visited):
#             passenger_station = (taxi_row, taxi_col)

#         if (obs[15] == 1 and destination_station is None):
#             if (cur_north in stations_not_visited):
#                 destination_station = cur_north
#             elif (cur_south in stations_not_visited):
#                 destination_station = cur_south
#             elif (cur_east in stations_not_visited):
#                 destination_station = cur_east
#             elif (cur_west in stations_not_visited):
#                 destination_station = cur_west
#             elif ((taxi_row, taxi_col) in stations_not_visited):
#                 destination_station = (taxi_row, taxi_col)

#         if (obs[0:2] in stations_not_visited):
#             stations_not_visited.remove(obs[0:2])
#         if (cur_north in stations_not_visited):
#             stations_not_visited.remove(cur_north)
#         if (cur_south in stations_not_visited):
#             stations_not_visited.remove(cur_south)
#         if (cur_east in stations_not_visited):
#             stations_not_visited.remove(cur_east)
#         if (cur_west in stations_not_visited):
#             stations_not_visited.remove(cur_west)

#         if (action == 4 and obs[0:2] == passenger_station):
#             has_passenger = True

        
#         if (passenger_station is None and destination_station is None):
#             target_dist = min([calculate_dist(obs[0:2], station) for station in stations_not_visited])

#         elif not has_passenger and passenger_station is not None:
#             target_dist = calculate_dist(obs[0:2], passenger_station)

#         elif has_passenger and destination_station is None:
#             target_dist = min([calculate_dist(obs[0:2], station) for station in stations_not_visited])

#         elif has_passenger and destination_station is not None:
#             target_dist = calculate_dist(obs[0:2], destination_station)

#         if passenger_station is not None:
#             know_passenger_station = True

#         if destination_station is not None:
#             know_destination_station = True


    
#     # Create feature vector with meaningful information
#     features = [
#         obstacle_north,
#         obstacle_south, 
#         obstacle_east,
#         obstacle_west,
#         has_passenger,
#         target_dist,
#         know_passenger_station,
#         know_destination_station
#     ]
    
#     return np.array(features)

# # # Load the model
# # model = QNetwork(4, 6).to(DEVICE)
# # try:
# #     model.load_state_dict(torch.load("q_network.pt", map_location=DEVICE))
# #     model.eval()
# # except Exception as e:
# #     print(f"Error loading model: {e}")
# #     # Fallback to random actions if model can't be loaded
# #     model = None

# # def get_action(obs):
# #     """
# #     Takes an observation as input and returns an action (0-5).
# #     Uses the trained Q-network to select the best action.
# #     """
# #     if model is None:
# #         return random.choice([0, 1, 2, 3, 4, 5])
    
# #     try:
# #         state_tensor = preprocess_state(obs)
        
# #         # Check if state_tensor is None or contains NaN values
# #         if state_tensor is None or torch.isnan(state_tensor).any():
# #             print("Warning: Invalid state tensor detected")
# #             return random.choice([0, 1, 2, 3, 4, 5])
        
# #         with torch.no_grad():
# #             q_values = model(state_tensor)
            
# #             # Check if q_values contains NaN or very large values
# #             if torch.isnan(q_values).any() or torch.isinf(q_values).any():
# #                 print("Warning: Invalid Q-values detected")
# #                 return random.choice([0, 1, 2, 3, 4, 5])
        
# #         return torch.argmax(q_values).item()
# #     except Exception as e:
# #         print(f"Error in get_action: {e}")
# #         return random.choice([0, 1, 2, 3, 4, 5])

# # implement q_table version
# def get_action(obs):
#     return random.choice([0, 1, 2, 3, 4, 5])

# # Ensemble version

# # # Load ensemble models
# # NUM_MODELS = 11  # Match the number used in training
# # models = []
# # model_path_prefix = "q_network"

# # # Try to load ensemble models
# # try:
# #     for i in range(NUM_MODELS):
# #         model_path = f"{model_path_prefix}_{i}.pt"
# #         if os.path.exists(model_path):
# #             model = QNetwork(4, 6).to(DEVICE)
# #             model.load_state_dict(torch.load(model_path, map_location=DEVICE))
# #             model.eval()
# #             models.append(model)
    
# #     if not models:  # If no ensemble models found, try loading single model
# #         model = QNetwork(4, 6).to(DEVICE)
# #         model.load_state_dict(torch.load("q_network.pt", map_location=DEVICE))
# #         model.eval()
# #         models.append(model)
    
# #     print(f"Successfully loaded {len(models)} model(s)")
# # except Exception as e:
# #     print(f"Error loading models: {e}")
# #     models = []

# # def get_action(obs):
# #     """
# #     Takes an observation as input and returns an action (0-5).
# #     Uses ensemble voting from multiple models to select the best action.
# #     Falls back to Q-table or random actions if models aren't available.
# #     """
# #     if models is None:
# #         return random.choice([0, 1, 2, 3, 4, 5])
    
# #     try:
# #         state_tensor = preprocess_state(obs)
        
# #         # Check if state_tensor is None or contains NaN values
# #         if state_tensor is None or torch.isnan(state_tensor).any():
# #             print("Warning: Invalid state tensor detected")
# #             return random.choice([0, 1, 2, 3, 4, 5])
        
# #         # If we have models, use ensemble voting
# #         if models:
# #             actions = []
# #             with torch.no_grad():
# #                 for model in models:
# #                     q_values = model(state_tensor)
                    
# #                     # Check if q_values contains NaN or very large values
# #                     if torch.isnan(q_values).any() or torch.isinf(q_values).any():
# #                         continue
                    
# #                     actions.append(torch.argmax(q_values).item())
            
# #             # Return the most common action (voting)
# #             if actions:
# #                 # print(actions)
# #                 # print(Counter(actions).most_common(1)[0][0])
# #                 return Counter(actions).most_common(1)[0][0]
# #                 # return actions

# #         # Fallback to random action if everything else fails
# #         return random.choice([0, 1, 2, 3, 4, 5])
    
# #     except Exception as e:
# #         print(f"Error in get_action: {e}")
# #         return random.choice([0, 1, 2, 3, 4, 5])


# def get_action(obs):
#     return agent.select_action(obs, 0)

# agent = DQN_agent(16, 6)
# agent.load_model('best_model.pt')