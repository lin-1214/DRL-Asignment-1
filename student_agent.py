# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)

def get_action(obs):
    state_read = obs[10:]

    if state_read in q_table:
        return np.argmax(q_table[state_read])
    else:
        return random.choice([0, 1, 2, 3, 4, 5])
