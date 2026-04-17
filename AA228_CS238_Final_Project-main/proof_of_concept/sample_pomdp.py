import random
from maze_pomdp import MazePOMDP, print_maze

def main():
    maze_size = 5
    observation_noise = 0.2  # Noise level for observations
    pomdp = MazePOMDP(maze_size, observation_noise)
    num_simulations = 10
    belief = (0, 0)  # Initial belief (assume starting from (0, 0))

    for i in range(num_simulations):
        action = random.choice(pomdp.actions)  # Random action selection
        next_state = pomdp.transition(belief, action)
        observation = pomdp.observation(belief, action, next_state)
        reward = pomdp.reward(next_state, action)
        
        print("Step:", i+1)
        print("Action:", action)
        print("Next State:", next_state)
        print("Observation:", observation)
        print("Reward:", reward)
        
        print_maze(next_state, maze_size)
        print()
        
        belief = observation  # Update belief to the observed position

if __name__ == "__main__":
    # main()
    ...