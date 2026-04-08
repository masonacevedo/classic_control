import gymnasium as gym
import numpy as np
from agent import Agent


def evaluateAgent(a,b,c,d, numEpisodes=10):
    total_reward_across_episodes = 0
    for _ in range(0, numEpisodes):
        env = gym.make("CartPole-v1", render_mode=None)
        observation, info = env.reset()

        episode_over = False
        episode_reward = 0
        agentToTest = Agent(a,b,c,d)
        while not episode_over:
            action = agentToTest.chooseAction(observation)
            observation, time_step_reward, terminated, truncated, info = env.step(action)
            episode_reward += time_step_reward
            episode_over = terminated or truncated

        print(f"Episode finished! Episode reward: {episode_reward}")
        env.close()
        total_reward_across_episodes += episode_reward
    return total_reward_across_episodes/numEpisodes



print("0,0,0,0:", evaluateAgent(0,0,0,0))