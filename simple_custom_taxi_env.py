import gym
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
import random


class SimpleTaxiEnv:
    """A simplified taxi environment where the agent must pick up a passenger and drop them at a destination."""
    
    # Action definitions
    SOUTH = 0
    NORTH = 1
    EAST = 2
    WEST = 3
    PICKUP = 4
    DROPOFF = 5
    
    def __init__(self, grid_size=5, fuel_limit=50):
        self.grid_size = 0
        self.fuel_limit = fuel_limit
        self.current_fuel = 0
        self.passenger_picked_up = False
        self.obstacles = set()
        self.taxi_pos = None
        self.taxi_loc = None
        self.stations = None
        self.passenger_loc = None
        self.destination = None
        self.previous_action = None

    def reset(self):
        """Reset the environment to a new random state."""
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False
        self.grid_size = random.randint(5, 10)
        self.grid_min = 5

        assert self.grid_size >= self.grid_min, "Grid size must be greater than or equal to grid_min"
        
        # Generate all possible grid positions
        all_locations = set((i, j) for i in range(self.grid_size) for j in range(self.grid_size))
        
        # Place random obstacles
        obstacle_count = random.randint(0, self.grid_size ** 2 // self.grid_min)
        self.obstacles = set(random.sample(list(all_locations), obstacle_count))
        
        # Remove obstacles from available locations
        all_locations -= self.obstacles
        
        # Place taxi
        self.taxi_pos = random.choice(list(all_locations))
        self.taxi_loc = self.taxi_pos
        
        # Place stations
        self.stations = random.sample(list(all_locations), 4)
        
        # Select passenger location and destination from stations
        self.passenger_loc, self.destination = random.sample(self.stations, 2)

        self.previous_action = None
        
        return self.get_state(), {}

    def step(self, action):
        """Take a step in the environment based on the given action."""
        self.current_fuel -= 1
        reward = 0
        done = False
        truncated = False

        # Movement actions
        if action in [self.SOUTH, self.NORTH, self.EAST, self.WEST]:
            reward -= 0.1
            
            next_row, next_col = self.taxi_pos
            if action == self.SOUTH:
                next_row += 1
            elif action == self.NORTH:
                next_row -= 1
            elif action == self.EAST:
                next_col += 1
            elif action == self.WEST:
                next_col -= 1
            
            # Check if move is valid
            if self._is_valid_position(next_row, next_col):
                self.taxi_pos = (next_row, next_col)
                self.taxi_loc = self.taxi_pos
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
            else:
                reward -= 5
        
        # Pickup action
        elif action == self.PICKUP:
            if (self.taxi_pos == self.passenger_loc) and (not self.passenger_picked_up):
                self.passenger_picked_up = True
            else:
                reward -= 10
        
        # Dropoff action
        elif action == self.DROPOFF:
            if (self.taxi_pos == self.destination) and self.passenger_picked_up:
                reward += 50
                done = True
            else:
                reward -= 10
        
        # Check if out of fuel
        if self.current_fuel <= 0:
            truncated = True
            reward -= 10

        return self.get_state(), reward, done, truncated, {}

    def _is_valid_position(self, row, col):
        """Check if a position is valid (within grid and not an obstacle)."""
        return (0 <= row < self.grid_size and 
                0 <= col < self.grid_size and 
                (row, col) not in self.obstacles)

    def get_state(self):
        """Return the current state representation."""
        taxi_row, taxi_col = self.taxi_pos
        
        # Check for obstacles or boundaries in each direction
        obstacle_north = int(taxi_row == 0 or ((taxi_row - 1, taxi_col) in self.obstacles))
        obstacle_south = int(taxi_row == self.grid_size - 1 or ((taxi_row + 1, taxi_col) in self.obstacles))
        obstacle_east = int(taxi_col == self.grid_size - 1 or ((taxi_row, taxi_col + 1) in self.obstacles))
        obstacle_west = int(taxi_col == 0 or ((taxi_row, taxi_col - 1) in self.obstacles))

        # Check if passenger is visible from current position
        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle = int((taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = any([passenger_loc_north, passenger_loc_south, 
                             passenger_loc_east, passenger_loc_west, passenger_loc_middle])
       
        # Check if destination is visible from current position
        destination_loc_north = int((taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int((taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east = int((taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west = int((taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle = int((taxi_row, taxi_col) == self.destination)
        destination_look = any([destination_loc_north, destination_loc_south, 
                               destination_loc_east, destination_loc_west, destination_loc_middle])
        
        # Construct state tuple
        state = (
            taxi_row, taxi_col, 
            self.stations[0][0], self.stations[0][1],
            self.stations[1][0], self.stations[1][1],
            self.stations[2][0], self.stations[2][1],
            self.stations[3][0], self.stations[3][1],
            obstacle_north, obstacle_south, obstacle_east, obstacle_west, 
            passenger_look, destination_look
        )
        return state

    def render_env(self, taxi_pos=None, action=None, step=None, fuel=None):
        """Render the environment as ASCII art."""
        clear_output(wait=True)

        if taxi_pos is None:
            taxi_pos = self.taxi_pos
        if fuel is None:
            fuel = self.current_fuel

        # Create empty grid
        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]

        # Place obstacles
        for obstacle in self.obstacles:
            grid[obstacle[0]][obstacle[1]] = 'X'
        
        # Place stations
        station_markers = ['R', 'G', 'Y', 'B']
        for i, station in enumerate(self.stations):
            grid[station[0]][station[1]] = station_markers[i]
            
        # Place passenger
        py, px = self.passenger_loc
        if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
            grid[py][px] = 'P'
            
        # Place destination
        dy, dx = self.destination
        if 0 <= dx < self.grid_size and 0 <= dy < self.grid_size:
            grid[dy][dx] = 'D'
            
        # Place taxi
        ty, tx = taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            grid[ty][tx] = '🚖'

        # Print status information
        print(f"\nStep: {step}")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")

        # Print grid
        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        """Convert action number to human-readable name."""
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"


def run_agent(agent_file, env_config, render=False):
    """Run an agent from a file in the environment."""
    # Import agent from file
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    # Initialize environment
    env = SimpleTaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    truncated = False
    step_count = 0
    
    # Initial render
    if render:
        env.render_env(taxi_pos=env.taxi_pos, action=None, step=step_count, fuel=env.current_fuel)
    
    # Run episode
    while not done and not truncated:
        action = student_agent.get_action(obs)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        step_count += 1

        if render:
            env.render_env(taxi_pos=env.taxi_pos, action=action, step=step_count, fuel=env.current_fuel)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward


if __name__ == "__main__":
    env_config = {
        "fuel_limit": 5000
    }
    
    agent_score = run_agent("student_agent.py", env_config, render=True)
    print(f"Final Score: {agent_score}")