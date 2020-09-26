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
from highway_env.road.objects import Landmark


class ParkingEnv_1(ParkingEnv):
    
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
                "type": "ContinuousAction"
            },
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "screen_width": 600,
            "screen_height": 300,
            "centering_position": [0.5, 0.5],
            "scaling": 7
        })
        return config
            
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminal, info = super().step(action)
        info.update({"is_success": self._is_success(obs['achieved_goal'], obs['desired_goal'])})
        return obs, reward, terminal, info

        
    def _create_vehicles(self) -> None:
        self.vehicle = Vehicle(self.road, [0, 0], 2*np.pi*self.np_random.rand(), 0)
        self.road.vehicles.append(self.vehicle)


        """
        positions = self.np_random.choice(self.road.network.lanes_list(), size=car_count+1, replace=False)
        for x in range(car_count):
            lane2 = positions[x]
            vehicle2 =  Vehicle(self.road, lane2.position(lane2.length/2, 0), lane2.heading, 0)
            self.road.vehicles.append(vehicle2)


        lane = positions[-1]
        self.goal = Landmark(self.road, lane.position(lane.length/2, 0), heading=lane.heading)
        self.road.objects.append(self.goal)
        """

        
        """
        add vehicels to free parking slots
        """
        spots = len(self.road.network.lanes_list())
        lane_pos = self.np_random.choice(spots)
        lane = self.road.network.lanes_list()[lane_pos]
        self.goal = Landmark(self.road, lane.position(lane.length/2, 0), heading=lane.heading)
        self.road.objects.append(self.goal)
        for x in range(spots):
            if x != lane_pos:
                lane_tmp = self.road.network.lanes_list()[x]
                vehicle_tmp =  Vehicle(self.road, lane_tmp.position(lane_tmp.length/2, 0), lane_tmp.heading, 0)
                self.road.vehicles.append(vehicle_tmp)


    """
    add collision reward for vehicels
    """
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        collision_reward = self.vehicle.crashed * (-5)        
        return -np.power(np.dot(np.abs(achieved_goal - desired_goal), self.REWARD_WEIGHTS), p) + collision_reward




register(
    id='parking-v1',
    entry_point='reinforcement_learning.envs:ParkingEnv_1',
    max_episode_steps=100
)
