import numpy as np

from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle


class NormalVehicle(IDMVehicle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = (0, 255, 0)


class AggressiveVehicle(IDMVehicle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = (255, 0, 0)
        self.target_speed = 35
        self.TIME_WANTED = 0.5
        self.MAX_ACCELERATION = 5.0
        self.COMFORT_ACC_MAX = 4.0
        self.COMFORT_ACC_MIN = -6.0
        self.POLITENESS = 0.0


class LazyVehicle(IDMVehicle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = (0, 0, 255)
        self.target_speed = 15
        self.TIME_WANTED = 2.5
        self.POLITENESS = 0.8

    def act(self, action=None):
        super().act(action)

        if np.random.rand() < 0.02:
            self.change_lane_policy()


class EgoVehicle(ControlledVehicle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = (255, 255, 0)


class CustomHighwayEnv(HighwayEnv):
    def _create_vehicles(self):
        self.road.vehicles = []
        n = self.config["vehicles_count"]

        for _ in range(n):
            r = np.random.rand()

            if r < 0.2:
                vehicle = AggressiveVehicle.create_random(self.road)
            elif r < 0.4:
                vehicle = LazyVehicle.create_random(self.road)
            else:
                vehicle = NormalVehicle.create_random(self.road)

            self.road.vehicles.append(vehicle)

        ego = EgoVehicle.create_random(self.road)
        self.road.vehicles.append(ego)
        self.vehicle = ego


if __name__ == "__main__":
    env = CustomHighwayEnv(render_mode="human")

    env.configure(
        {
            "vehicles_count": 50,
            "duration": 40,
            "simulation_frequency": 30,
            "policy_frequency": 5,
            "screen_width": 1200,
            "screen_height": 400,
            "lanes_count": 4,
            "vehicles_density": 30,
        }
    )

    num_episodes = 50
    max_steps = 300

    for episode in range(num_episodes):
        obs, info = env.reset()
        print(f"Episode {episode + 1}")

        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            if done or truncated:
                break

    env.close()
