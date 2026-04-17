import random
from typing import Optional
import gymnasium as gym
import numpy as np


class MazePOMDPEnv(gym.Env):
    def __init__(self, size: int = 5):
        self.size = size

        self._agent_location = np.array([0, 0], dtype = np.int32)
        self._target_location = np.array([-1, -1], dtype = np.int32)

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   # [x, y] coordinates
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [x, y] coordinates
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([0, -1]),
            2: np.array([-1, 0]),
            3: np.array([1, 0])
        }

    def _get_obs(self):
        return {"agent" : self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Randomly place target, ensuring it's different from agent position
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
                # Map the discrete action (0-3) to a movement direction
        direction = self._action_to_direction[action]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Check if agent reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        reward = 1 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

gym.register(
    id="MazeEnv",
    entry_point="maze_pomdp:MazePOMDPEnv",
    max_episode_steps=300,  # Prevent infinite episodes
)

class MazePOMDP:
    def __init__(self, maze_size, observation_noise):
        self.maze_size = maze_size
        self.states = [(x, y) for x in range(maze_size) for y in range(maze_size)]
        self.actions = ["up", "down", "left", "right"]
        self.observations = [(x, y) for x in range(maze_size) for y in range(maze_size)]  # All possible positions
        self.observation_noise = observation_noise

    def transition(self, state, action):
        x, y = state
        if action == "up":
            return (max(x - 1, 0), y)
        elif action == "down":
            return (min(x + 1, self.maze_size - 1), y)
        elif action == "left":
            return (x, max(y - 1, 0))
        elif action == "right":
            return (x, min(y + 1, self.maze_size - 1))

    def observation(self, state, action, next_state):
        if random.random() < self.observation_noise:
            return next_state  # Noisy observation is the true position
        else:
            return random.choice(self.observations)  # Random position as noisy observation

    def reward(self, state, action):
        if state == (self.maze_size - 1, self.maze_size - 1):  # Goal state
            return 10
        elif state in [(1, 1), (2, 2), (3, 3)]:  # Obstacles
            return -5
        else:
            return -1

def print_maze(agent_position, maze_size):
    for i in range(maze_size):
        for j in range(maze_size):
            if (i, j) == agent_position:
                print("A", end=" ")  # Agent
            elif (i, j) == (maze_size - 1, maze_size - 1):
                print("G", end=" ")  # Goal
            elif (i, j) in [(1, 1), (2, 2), (3, 3)]:
                print("X", end=" ")  # Obstacle
            else:
                print(".", end=" ")  # Empty space
        print()