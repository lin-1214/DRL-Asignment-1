# Remember to adjust your student ID in meta.xml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from tqdm import tqdm

# Global variables
MODEL_FILE = "q_network.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EPSILON = 0.1  # For exploration during testing

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Add a ReplayBuffer class for experience replay
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
    This function uses relative positions and features that work regardless of grid size.
    """
    # Extract key components from the observation
    taxi_row, taxi_col, station0_row, station0_col, station1_row, station1_col, \
    station2_row, station2_col, station3_row, station3_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs
    
    # Instead of trying to determine the exact grid size, we'll use a normalization
    # approach that works for any reasonable grid size (assuming it's at least 5x5)
    # We'll use a reference size of 10 which should handle most test cases
    reference_size = 10.0
    
    # Create feature vector with normalized positions
    features = [
        taxi_row / reference_size,  # Normalize to [0,1] range
        taxi_col / reference_size,  # Normalize to [0,1] range
        obstacle_north,
        obstacle_south, 
        obstacle_east,
        obstacle_west,
        passenger_look,
        destination_look
    ]
    
    return torch.FloatTensor(features).to(DEVICE)

def get_action(obs):
    """
    Takes an observation as input and returns an action (0-5).
    Uses the trained Q-network to select the best action.
    """
    # Load model if it exists and hasn't been loaded yet
    if not hasattr(get_action, "model"):
        if os.path.exists(MODEL_FILE):
            get_action.model = QNetwork(8, 6).to(DEVICE)
            get_action.model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
            get_action.model.eval()
        else:
            # If model doesn't exist, return random actions
            return random.choice([0, 1, 2, 3, 4, 5])
    
    # Epsilon-greedy policy for exploration during testing
    if random.random() < EPSILON:
        return random.choice([0, 1, 2, 3, 4, 5])
    
    # Preprocess state and get Q-values
    state_tensor = preprocess_state(obs)
    with torch.no_grad():
        q_values = get_action.model(state_tensor)
    
    # Return action with highest Q-value
    return torch.argmax(q_values).item()

    # return random.choice([0, 1, 2, 3, 4, 5])

def shape_reward(obs, next_obs, action, reward):
    """
    Apply reward shaping to encourage more efficient behavior.
    """
    # Extract information from observations
    taxi_row, taxi_col, _, _, _, _, _, _, _, _, _, _, _, _, passenger_look, destination_look = obs
    next_taxi_row, next_taxi_col, _, _, _, _, _, _, _, _, _, _, _, _, next_passenger_look, next_destination_look = next_obs
    
    shaped_reward = reward
    
    # Reward for getting closer to passenger when not carrying
    if passenger_look == 0 and next_passenger_look == 1:
        shaped_reward += 0.5  # Reward for getting closer to passenger
    
    # Reward for getting closer to destination when carrying passenger
    if passenger_look == 1 and destination_look == 0 and next_destination_look == 1:
        shaped_reward += 0.5  # Reward for getting closer to destination
    
    # Penalty for trying pickup when no passenger is present
    if action == 4 and passenger_look == 0:
        shaped_reward -= 0.2
    
    # Penalty for trying dropoff when not at destination
    if action == 5 and destination_look == 0:
        shaped_reward -= 0.2
    
    return shaped_reward

def train_agent(num_episodes=10000, gamma=0.99, batch_size=64):
    """
    Train the agent using DQN with experience replay and target network.
    """
    from simple_custom_taxi_env import SimpleTaxiEnv
    
    # Initialize environment
    env = SimpleTaxiEnv()
    
    # Initialize Q-networks (policy and target)
    policy_net = QNetwork(8, 6).to(DEVICE)
    target_net = QNetwork(8, 6).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target network is only used for inference
    
    # Initialize optimizer with a more appropriate learning rate
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=50000)
    
    # Training parameters
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9999
    target_update_frequency = 10  # Update target network every N episodes
    
    # Training loop
    best_reward = -float('inf')
    losses = []
    episode_rewards = []  # Renamed to avoid confusion
    
    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        state_tensor = preprocess_state(obs)
        done = False
        total_reward = 0
        episode_losses = []
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice([0, 1, 2, 3, 4, 5])
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                action = torch.argmax(q_values).item()
            
            # Take action and observe next state
            next_obs, reward, done, _, _ = env.step(action)
            next_state_tensor = preprocess_state(next_obs)
            
            # Apply reward shaping
            shaped_reward = shape_reward(obs, next_obs, action, reward)
            
            # Store transition in replay buffer
            replay_buffer.push(
                state_tensor.cpu().numpy(),
                action,
                shaped_reward,
                next_state_tensor.cpu().numpy(),
                done
            )
            
            total_reward += reward  # Track original reward for evaluation
            
            # Move to next state
            obs = next_obs
            state_tensor = next_state_tensor
            
            # Train on a batch of transitions if buffer has enough samples
            if len(replay_buffer) >= batch_size:
                # Sample a batch from replay buffer
                states, actions, batch_rewards, next_states, dones = replay_buffer.sample(batch_size)  # Renamed to batch_rewards
                
                # Convert to tensors
                states = torch.FloatTensor(states).to(DEVICE)
                actions = torch.LongTensor(actions).to(DEVICE)
                batch_rewards = torch.FloatTensor(batch_rewards).to(DEVICE)  # Using batch_rewards instead
                next_states = torch.FloatTensor(next_states).to(DEVICE)
                dones = torch.FloatTensor(dones).to(DEVICE)
                
                # Compute current Q values
                current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Compute target Q values with target network
                with torch.no_grad():
                    max_next_q_values = target_net(next_states).max(1)[0]
                    target_q_values = batch_rewards + gamma * max_next_q_values * (1 - dones)  # Using batch_rewards
                
                # Compute loss and update
                loss = criterion(current_q_values, target_q_values)
                
                # Gradient clipping to prevent exploding gradients
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Track loss
                episode_losses.append(loss.item())
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Update target network periodically
        if (episode + 1) % target_update_frequency == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Track metrics
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses.append(avg_loss)
        episode_rewards.append(total_reward)  # Using episode_rewards list
        
        # Track best reward
        if total_reward > best_reward:
            best_reward = total_reward
            # Save best model
            torch.save(policy_net.state_dict(), MODEL_FILE)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward:.2f}, Best: {best_reward:.2f}, Loss: {avg_loss:.6f}, Epsilon: {epsilon:.4f}")
    
    # Save training metrics
    np.save("training_losses.npy", np.array(losses))
    np.save("training_rewards.npy", np.array(episode_rewards))  # Using episode_rewards
    
    print("Training completed and model saved.")

if __name__ == "__main__":
    # This will only run when you execute this file directly
    train_agent(num_episodes=10000)

# else:
#     model_path = "q_network.pt"
#     # TODO: load the model
#     model = QNetwork(8, 6).to(DEVICE)
#     model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#     model.eval()
