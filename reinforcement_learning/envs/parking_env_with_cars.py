import gym
import highway_env
from typing import Tuple
from gym.envs.registration import register
from gym import GoalEnv
import numpy as np
from numpy.core._multiarray_umath import ndarray
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle, Obstacle
from highway_env.envs.parking_env import ParkingEnv
from highway_env.envs.common.observation import observation_factory
from highway_env.road.objects import Landmark

car_count = 8

class ParkingEnv_1(ParkingEnv):
    
    def __init__(self, config: dict = None) -> None:
        super(ParkingEnv_1, self).__init__()
        self.obsCars = None
        self.configKinematics = self.default_config_kinematics()
    
    #confi 2 for kinematics
    @classmethod
    def default_config_kinematics(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False
            },
            "action": {
                "type": "Continuous"
            },
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "screen_width": 600,
            "screen_height": 300,
            "centering_position": [0.5, 0.5],
            "scaling": 7
        })
        return config
        
    def define_spaces(self) -> None:
        self.observation = observation_factory(self, self.config["observation"])
        self.obsCars = observation_factory(self, self.configKinematics["observation"])
        self.observation.space()["kinematics"] = self.obsCars.space
        self.observation_space = self.observation.space()

        if self.config["action"]["type"] == "Discrete":
            self.action_space = spaces.Discrete(len(self.ACTIONS))
        elif self.config["action"]["type"] == "Continuous":
            self.action_space = spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
            
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminal, info = super().step(action)
        obs["kinematics"] = self.obsCars.observe()
        info.update({"is_success": self._is_success(obs['achieved_goal'], obs['desired_goal'])})
        return obs, reward, terminal, info

        
    def _create_vehicles(self) -> None:
        self.vehicle = Vehicle(self.road, [0, 0], 2*np.pi*self.np_random.rand(), 0)
        self.road.vehicles.append(self.vehicle)

        positions = self.np_random.choice(self.road.network.lanes_list(), size=car_count+1, replace=False)

        for x in range(car_count):
            lane2 = positions[x]
            vehicle2 =  Vehicle(self.road, lane2.position(lane2.length/2, 0), lane2.heading, 0)
            self.road.vehicles.append(vehicle2)


        lane = positions[-1]
        self.goal = Landmark(self.road, lane.position(lane.length/2, 0), heading=lane.heading)
        self.road.objects.append(self.goal)


    

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "KinematicsGoal",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False
            },
            "action": {
                "type": "Continuous"
            },
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "screen_width": 600,
            "screen_height": 300,
            "centering_position": [0.5, 0.5],
            "scaling": 7
        })
        return config








register(
    id='parking-v1',
    entry_point='reinforcement_learning.envs:ParkingEnv_1',
    max_episode_steps=100
)
