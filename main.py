import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()
print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

count = 0
while not episode_over:
    action = count % 2
    count += 1
    
    # print(f"Action: {action}")
    # print("type(action): ", type(action))
    observation, reward, terminated, truncated, info = env.step(action)
    # print(f"Observation: {observation}")
    total_reward += reward
    episode_over = terminated or truncated
    
print(f"Episode finished! Total reward: {total_reward}")
env.close()