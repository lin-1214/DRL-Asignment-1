# Remember to adjust your student ID in meta.xml
import numpy as np
import random
import pickle

# Define these as global variables
passenger_picked_up = False
previous_action = None
q_table = {}

with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

def get_action(obs):
    # Add global declarations to access and modify the module-level variables
    global passenger_picked_up, previous_action
    
    def extract_state_features(obs, passenger_picked_up, previous_action):
        """Extract relevant state features from the environment observation."""
        # Unpack observation
        (taxi_row, taxi_col, 
         station0_row, station0_col, 
         station1_row, station1_col, 
         station2_row, station2_col, 
         station3_row, station3_col, 
         obstacle_north, obstacle_south, obstacle_east, obstacle_west, 
         passenger_look, destination_look) = obs
        
        # Collect all station positions
        stations = [
            (station0_row, station0_col),
            (station1_row, station1_col),
            (station2_row, station2_col),
            (station3_row, station3_col)
        ]
        
        # Check for stations in adjacent positions
        taxi_pos = (taxi_row, taxi_col)
        station_middle = taxi_pos in stations
        station_north = (taxi_row - 1, taxi_col) in stations
        station_south = (taxi_row + 1, taxi_col) in stations
        station_east = (taxi_row, taxi_col + 1) in stations
        station_west = (taxi_row, taxi_col - 1) in stations
        
        # Return state representation as a tuple with reordered elements to match training
        return (
            passenger_picked_up,
            passenger_look, destination_look,
            station_north, station_south, station_east, station_west, station_middle,
            obstacle_north, obstacle_south, obstacle_east, obstacle_west,
            previous_action
        )

    def softmax(x):
        """Compute softmax values for array x."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    # Get current state using the same function as in training
    state = extract_state_features(obs, passenger_picked_up, previous_action)
    
    # Action constants for better readability
    PICKUP = 4
    DROPOFF = 5
    
    # Choose action based on Q-table
    if state not in q_table:
        action = np.random.randint(4)  # Random action if state not in Q-table
    else:
        probs = softmax(q_table[state])
        action = np.random.choice(range(6), p=probs)  # Use softmax probabilities
    
    # Update passenger status based on action
    at_station = state[7]  # station_middle
    if action == PICKUP and state[1] == 1 and at_station:  # Pickup action at passenger location and at station
        passenger_picked_up = True
    elif action == DROPOFF and state[2] == 1 and at_station and passenger_picked_up:  # Dropoff action at destination, at station, with passenger
        passenger_picked_up = False
    
    previous_action = action
    
    return action


