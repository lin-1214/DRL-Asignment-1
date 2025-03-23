# Remember to adjust your student ID in meta.xml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from tqdm import tqdm
import pickle

# Global variables
MODEL_FILE = "q_network_ensemble_2.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

        # Initialize weights with custom initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # He initialization for weights (good for ReLU activations)
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            # Initialize bias to small random values
            if module.bias is not None:
                nn.init.uniform_(module.bias, -0.01, 0.01)
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

def preprocess_state(obs):
    """
    Convert observation to a tensor for the neural network.
    This function uses relative positions and distances between key elements.
    """
    # Extract key components from the observation
    taxi_row, taxi_col, station0_row, station0_col, station1_row, station1_col, \
    station2_row, station2_col, station3_row, station3_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs

    
    # Create feature vector with more meaningful relative information
    features = [
        # Obstacle information (binary)
        taxi_row,
        taxi_col,
        obstacle_north,
        obstacle_south, 
        obstacle_east,
        obstacle_west,
        
        # # Passenger and destination information (binary)
        passenger_look,
        destination_look,
        # distances_to_stations[0],
        # distances_to_stations[1],
        # distances_to_stations[2],
        # distances_to_stations[3]
    ]
    
    return torch.FloatTensor(features).to(DEVICE)

def get_action(obs):
    """
    Takes an observation as input and returns an action (0-5).
    Uses the trained Q-network to select the best action.
    """
    
    state_tensor = preprocess_state(obs)
    
    with torch.no_grad():
        q_values = get_action.model(state_tensor)
    
    return torch.argmax(q_values).item()

def shape_reward(info, obs, next_obs, action, reward):
    """
    Apply reward shaping to encourage more efficient behavior based on distance metrics.
    """
    # Extract information from observations
    taxi_row, taxi_col, station0_row, station0_col, station1_row, station1_col, \
    station2_row, station2_col, station3_row, station3_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs
    
    next_taxi_row, next_taxi_col, next_station0_row, next_station0_col, next_station1_row, next_station1_col, \
    next_station2_row, next_station2_col, next_station3_row, next_station3_col, \
    next_obstacle_north, next_obstacle_south, next_obstacle_east, next_obstacle_west, \
    next_passenger_look, next_destination_look = next_obs
    
    shaped_reward = reward

    if (action == 0 and obstacle_south == 1) or (action == 1 and obstacle_north == 1)  or (action == 2 and obstacle_east == 1) or (action == 3 and obstacle_west == 1):
        shaped_reward -= 20.0


    # pick up passenger
    if action == 4 and (taxi_row, taxi_col) in info.stations and passenger_look == 1:
        shaped_reward += 10.0
        info.passenger = True
    elif action == 4:
        shaped_reward -= 20

    # find destination
    if (taxi_row, taxi_col) in info.stations and destination_look == 1 and info.destination == []:
        info.destination = (taxi_row, taxi_col)
        shaped_reward += 10

    # drop off passenger
    if action == 5 and (taxi_row, taxi_col) in info.stations and destination_look == 1 and info.passenger == True:
        shaped_reward += 100
    elif action == 5:
        shaped_reward -= 20

    if taxi_row == next_taxi_row and taxi_col == next_taxi_col:
        shaped_reward -= 50 

    # if distance_to_stations[0] <= 1 or distance_to_stations[1] <= 1 or distance_to_stations[2] <= 1 or distance_to_stations[3] <= 1 and passenger_look == 1:
    #     shaped_reward += 10.0

    

    return shaped_reward

def soft_update(target_net, policy_net, tau=0.001):
    """Soft update of target network: θ′ ← τθ + (1 - τ)θ′"""
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

class DQN(nn.Module):
    def __init__(self, state_size, action_size, gamma=0.99, batch_size=64, lr=1e-4, device=DEVICE):
        super().__init__()
        self.policy_net = QNetwork(state_size, action_size).to(device)
        self.target_net = QNetwork(state_size, action_size).to(device)
        
        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.tau = 0.001  # Soft update parameter
        self.device = device
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=200, verbose=True)
        
        # Use your existing ReplayBuffer instead of Memory
        self.memory = ReplayBuffer(capacity=50000)
        self.batch_size = batch_size
        self.gamma = gamma
        self.criterion = nn.SmoothL1Loss()  # Huber loss
        
        # State tracking
        self.stations = [[0, 0], [0, 0], [0, 0], [0, 0]]
        self.passenger = False
        self.destination = []
    
    def load(self, model_path=MODEL_FILE):
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, model_path=MODEL_FILE):
        torch.save(self.policy_net.state_dict(), model_path)
    
    def reward_shaping(self, obs, next_obs, action, reward):
        # Use your existing reward shaping function

        return shape_reward(self,obs, next_obs, action, reward)
    def reset(self, obs):
        self.passenger = False
        self.obstacles = []
        self.destination = []

        self.update_map(obs)

    def get_action(self, obs, epsilon):
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            return random.choice([0, 1, 2, 3, 4, 5])
        
        # Use your existing preprocess_state function
        state_tensor = preprocess_state(obs)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        
        return torch.argmax(q_values).item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        try:
            # Sample a batch from replay buffer
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            
            # Convert to tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            
            # Compute current Q values
            current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute target Q values more stably
            with torch.no_grad():
                # Use target network directly for more stable learning
                next_q_values = self.target_net(next_states).max(1)[0]
                # Detach to ensure no gradients flow back
                target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
                # Clip target values to reduce variance
                target_q_values = torch.clamp(target_q_values, -100, 100)
            
            # Compute loss with error handling
            loss = self.criterion(current_q_values, target_q_values)
            
            # Check if loss is valid
            if not torch.isfinite(loss):
                print("Warning: Non-finite loss detected. Skipping update.")
                return
            
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            
            # Apply gradient clipping with a smaller norm to prevent instability
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
            
            # Check for NaN gradients
            for param in self.policy_net.parameters():
                if param.grad is not None and not torch.all(torch.isfinite(param.grad)):
                    print("Warning: Non-finite gradients detected. Skipping update.")
                    return
                
            self.optimizer.step()
            
            # Soft update target network
            self.soft_update()
            
        except Exception as e:
            print(f"Error during update: {e}")
            return
    
    def soft_update(self):
        """Soft update of target network: θ′ ← τθ + (1 - τ)θ′"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def update_map(self, obs):
        taxi_row, taxi_col, station0_row, station0_col, station1_row, station1_col, \
        station2_row, station2_col, station3_row, station3_col, \
        obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
        passenger_look, destination_look = obs

        self.stations = [[station0_row, station0_col], [station1_row, station1_col], [station2_row, station2_col], [station3_row, station3_col]]

        # if obstacle_north == 1 and [taxi_row - 1, taxi_col] not in self.obstacles:
        #     self.obstacles.append([taxi_row - 1, taxi_col])
        # if obstacle_south == 1 and [taxi_row + 1, taxi_col] not in self.obstacles:
        #     self.obstacles.append([taxi_row + 1, taxi_col])
        # if obstacle_east == 1 and [taxi_row, taxi_col + 1] not in self.obstacles:
        #     self.obstacles.append([taxi_row, taxi_col + 1])
        # if obstacle_west == 1 and [taxi_row, taxi_col - 1] not in self.obstacles:
        #     self.obstacles.append([taxi_row, taxi_col - 1])

def train_agent(num_episodes=10000, gamma=0.99, batch_size=128):
    """
    Train the agent using DQN with experience replay and target network.
    """
    from simple_custom_taxi_env import SimpleTaxiEnv
    
    # Initialize environment
    env = SimpleTaxiEnv()
    
    # Load existing Q-table if available
    q_table = None
    q_table_path = "q_table.pkl"
    if os.path.exists(q_table_path):
        try:
            with open(q_table_path, "rb") as f:
                q_table = pickle.load(f)
            print(f"Loaded Q-table with {len(q_table)} entries")
        except Exception as e:
            print(f"Error loading Q-table: {e}")
    
    # Initialize agent (single DQN)
    agent = DQN(state_size=8, action_size=6, gamma=gamma, batch_size=batch_size)
    
    # Training parameters
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9999
    rewards = np.array([])
    
    # Training loop
    best_reward = -float('inf')
    
    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        agent.reset(obs)  # Reset agent's state tracking
        
        done = False
        total_reward = 0
        episode_losses = []
        
        while not done:
            # Get action using epsilon-greedy policy
            action = agent.get_action(obs, epsilon)
            
            # Take action and observe next state
            next_obs, reward, done, _ = env.step(action)
            
            # Apply reward shaping
            shaped_reward = agent.reward_shaping(obs, next_obs, action, reward)
            
            # Store transition in replay buffer
            state_tensor = preprocess_state(obs)
            next_state_tensor = preprocess_state(next_obs)
            
            agent.memory.push(
                state_tensor.cpu().numpy(),
                action,
                shaped_reward,
                next_state_tensor.cpu().numpy(),
                done
            )
            
            # Update agent
            agent.update()
            
            total_reward += reward  # Track original reward for evaluation
            obs = next_obs
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards = np.append(rewards, total_reward)
        # Track best reward and save model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save()
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print("Environment Information:")
            print(f"Fuel Limit: {env.fuel_limit}")
            print(f"Grid Size: {env.grid_size}")
            print(f"Stations: {agent.stations}")
            print(f"Obstacle counts: {len(env.obstacles)}")
            print(f"State: {state_tensor}")

            print(f"Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")
    
    print("Training completed and model saved.")
    
    # Save final Q-table at the end of training
    q_table = {}
    with torch.no_grad():
        for state_key in tqdm(list(q_table.keys()) if q_table else [], desc="Creating Q-table"):
            state_tensor = torch.FloatTensor(state_key).to(DEVICE)
            q_values = agent.policy_net(state_tensor).cpu().numpy()
            q_table[state_key] = q_values
            
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)

if __name__ == "__main__":
    # This will only run when you execute this file directly
    train_agent(num_episodes=40000)

# else:
#     model_path = "q_network.pt"
#     # TODO: load the model
#     model = QNetwork(8, 6).to(DEVICE)
#     model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#     model.eval()
