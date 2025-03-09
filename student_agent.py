# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import os

# Global variables
Q_TABLE_FILE = "q_table.pkl"
q_table = {}
# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT = 0.99
EPSILON = 0.1  # For exploration during testing

def get_state_key(obs):
    """
    Convert observation to a hashable key for the Q-table.
    This function extracts the essential information from the observation.
    """
    # Extract key components from the observation
    taxi_row, taxi_col, station0_row, station0_col, station1_row, station1_col, \
    station2_row, station2_col, station3_row, station3_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs
    
    # Create a tuple that captures the essential state information
    state_key = (
        taxi_row, taxi_col,
        passenger_look, destination_look,
        obstacle_north, obstacle_south, obstacle_east, obstacle_west
    )
    
    return state_key

def get_action(obs):
    """
    Takes an observation as input and returns an action (0-5).
    Uses the Q-table to select the best action for the current state.
    """
    # Convert observation to state key
    state_key = get_state_key(obs)
    
    # Load Q-table if it exists and hasn't been loaded yet
    global q_table
    if not q_table and os.path.exists(Q_TABLE_FILE):
        try:
            with open(Q_TABLE_FILE, "rb") as f:
                q_table = pickle.load(f)
        except:
            # If loading fails, initialize an empty Q-table
            q_table = {}
    
    # Epsilon-greedy policy for exploration during testing
    if random.random() < EPSILON:
        return random.choice([0, 1, 2, 3, 4, 5])
    
    # If state is not in Q-table, return a random action
    if state_key not in q_table:
        return random.choice([0, 1, 2, 3, 4, 5])
    
    # Return the action with the highest Q-value
    return np.argmax(q_table[state_key])

# The following code is for training the agent
# It won't be used during evaluation but is included for completeness

def train_agent(num_episodes=5000):
    """
    Train the agent using Q-learning.
    """
    from simple_custom_taxi_env import SimpleTaxiEnv
    
    global q_table
    q_table = {}
    
    env = SimpleTaxiEnv()
    
    # Training parameters
    alpha = LEARNING_RATE
    gamma = DISCOUNT
    epsilon = 1.0  # Start with high exploration
    min_epsilon = 0.01
    epsilon_decay = 0.995
    
    # Training loop
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state_key = get_state_key(obs)
        done = False
        total_reward = 0
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice([0, 1, 2, 3, 4, 5])
            else:
                if state_key not in q_table:
                    q_table[state_key] = np.zeros(6)
                action = np.argmax(q_table[state_key])
            
            # Take action and observe next state
            next_obs, reward, done, _, _ = env.step(action)
            next_state_key = get_state_key(next_obs)
            total_reward += reward
            
            # Initialize Q-values if needed
            if state_key not in q_table:
                q_table[state_key] = np.zeros(6)
            if next_state_key not in q_table:
                q_table[next_state_key] = np.zeros(6)
            
            # Q-learning update
            best_next_action = np.argmax(q_table[next_state_key])
            q_table[state_key][action] = (1 - alpha) * q_table[state_key][action] + \
                                         alpha * (reward + gamma * q_table[next_state_key][best_next_action] * (not done))
            
            # Move to next state
            state_key = next_state_key
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward}, Epsilon: {epsilon:.4f}")
    
    # Save Q-table
    with open(Q_TABLE_FILE, "wb") as f:
        pickle.dump(q_table, f)
    
    print("Training completed and Q-table saved.")

if __name__ == "__main__":
    # This will only run when you execute this file directly
    train_agent(num_episodes=5000)

