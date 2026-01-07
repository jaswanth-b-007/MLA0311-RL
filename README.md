  import numpy as np
import gymnasium as gym
from gymnasium import spaces
class WarehouseEnv(gym.Env):
def __init__(self):
super().__init__()
self.grid_size = 4
self.action_space = spaces.Discrete(4)
self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)
self.goal = (3, 3)
self.item = (1, 1)
self.obstacle = (2, 2)
def state_to_pos(self, state):
return (state // self.grid_size, state % self.grid_size)
def pos_to_state(self, pos):
return pos[0] * self.grid_size + pos[1]
def step(self, state, action):
row, col = self.state_to_pos(state)
if action == 0: row -= 1 # Up
elif action == 1: col += 1 # Right
elif action == 2: row += 1 # Down
elif action == 3: col -= 1 # Left
row = np.clip(row, 0, self.grid_size - 1)
col = np.clip(col, 0, self.grid_size - 1)
new_state = self.pos_to_state((row, col))
reward = 0
if (row, col) == self.item:
reward = 2
elif (row, col) == self.goal:
reward = 5
elif (row, col) == self.obstacle:
reward = -2
return new_state, reward
gamma = 0.9
theta = 0.0001
env = WarehouseEnv()
num_states = env.observation_space.n
num_actions = env.action_space.n
V = np.zeros(num_states)
policy = np.ones((num_states, num_actions)) / num_actions
while True:
delta = 0
for s in range(num_states):
v = V[s]
new_v = 0
for a in range(num_actions):
next_state, reward = env.step(s, a)
new_v += policy[s][a] * (reward + gamma * V[next_state])
V[s] = new_v
delta = max(delta, abs(v - new_v))
if delta < theta:
break
print("Value Function:")
print(V.reshape(4, 4))
