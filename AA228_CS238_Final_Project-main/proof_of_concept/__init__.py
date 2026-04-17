from gymnasium.envs.registration import register

register(
    id="MazeEnv",
    entry_point="maze_pomdp:MazePOMDPEnv",
    max_episode_steps=300,  # Prevent infinite episodes
)