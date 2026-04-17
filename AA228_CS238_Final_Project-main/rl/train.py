"""
Train an A2C agent on the DARE ramjet gym environment.
Usage:
    python train.py [--timesteps 2000] [--save-path a2c_dare]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from dare_env import DareEnv
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument("--save-path", type=str, default="a2c_dare")
    args = parser.parse_args()

    env = Monitor(DareEnv())

    model = A2C("MlpPolicy", env, verbose=1, n_steps=5, gamma=1.0)
    print(f"Training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)

    model.save(args.save_path)
    print(f"Model saved to {args.save_path}.zip")

    # Quick evaluation
    print("\nEvaluating over 5 episodes...")
    obs, _ = env.reset()
    for ep in range(5):
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        print(f"  Episode {ep+1}: reward={total_reward:.1f}, steps={steps}, "
              f"mach={info['mach_mixture']:.3f}, x_norm={info['x_norm']:.3f}, "
              f"er={info['equivalence_ratio']:.4f}")
        obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
