import gymnasium as gym
import highway_env as highway


ENV_ID = "highway-v0"


if __name__ == "__main__":
    print(f"Using highway_env {highway.__version__}")
    env = gym.make(ENV_ID, render_mode="human")

    env.unwrapped.configure(
        {
            # Core congestion controls
            "vehicles_count": 70,
            "vehicles_density": 2.5,
            "lanes_count": 3,
            # Dynamics
            "duration": 40,
            "simulation_frequency": 30,
            "policy_frequency": 5,
            # Traffic behavior
            "ego_spacing": 1.0,
            "initial_spacing": 1.0,
            # Rendering
            "screen_width": 1200,
            "screen_height": 400,
        }
    )

    num_episodes = 5
    max_steps = 300

    for ep in range(num_episodes):
        obs, info = env.reset()
        print(f"Episode {ep + 1}")

        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            if done or truncated:
                break

    env.close()
