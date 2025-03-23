# # Remember to adjust your student ID in meta.xml
# import numpy as np
# import torch
# import random
# import gym
# import pickle
# import os
# from train_dqn import DQN_agent
# from train_ensemble_dqn import DQN
import numpy as np
import torch
import random
from train_ensemble_dqn import DQN
from train_ensemble_dqn import preprocess_state

# # # Global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# # # Load the model
model = DQN(state_size=8, action_size=6, gamma=0.99, batch_size=128)

model.load("q_network_ensemble_2.pt")

def get_action(obs, epsilon=0):  # Add a small exploration probability
    """
    Takes an observation as input and returns an action (0-5).
    Uses the trained Q-network to select the best action with some robustness for unseen states.
    """
    if model is None:
        return random.choice([0, 1, 2, 3, 4, 5])
    
    # Always maintain some exploration to handle unseen states
    if random.random() < epsilon:
        return random.choice([0, 1, 2, 3, 4, 5])
    
    state_tensor = preprocess_state(obs)
    try:
        return torch.argmax(model.target_net(state_tensor)).item()
    except Exception as e:
        print(f"Error in get_action: {e}")
        return random.choice([0, 1, 2, 3, 4, 5])

# # implement q_table version
# def get_action(obs):
#     return random.choice([0, 1, 2, 3, 4, 5])

# # Ensemble version

# # # Load ensemble models
# # # NUM_MODELS = 11  # Match the number used in training
# # # models = []
# # # model_path_prefix = "q_network"

# # # # Try to load ensemble models
# # # try:
# # #     for i in range(NUM_MODELS):
# # #         model_path = f"{model_path_prefix}_{i}.pt"
# # #         if os.path.exists(model_path):
# # #             model = QNetwork(4, 6).to(DEVICE)
# # #             model.load_state_dict(torch.load(model_path, map_location=DEVICE))
# # #             model.eval()
# # #             models.append(model)
    
# # #     if not models:  # If no ensemble models found, try loading single model
# # #         model = QNetwork(4, 6).to(DEVICE)
# # #         model.load_state_dict(torch.load("q_network.pt", map_location=DEVICE))
# # #         model.eval()
# # #         models.append(model)
    
# # #     print(f"Successfully loaded {len(models)} model(s)")
# # # except Exception as e:
# # #     print(f"Error loading models: {e}")
# # #     models = []

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