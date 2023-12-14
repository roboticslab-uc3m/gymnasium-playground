import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GrippersEnv(gym.Env):
    
    def __init__(self, render_mode=None, init_angle=0., goal_angle=0.5):
        #super(GrippersEnv, self).__init__()
        """
        The environment emulates two grippers,
        wher the second one have the capability to 
        rotate over the first one.

        The goal is to achieve a certain rotation angle.

        The angle values are normalized:
        value 1 -> pi rad
        """

        # Initial position
        self.initial_angle = np.array([init_angle])

        # goal
        self.goal_angle = np.array([goal_angle])

        # set max vel 
        self.max_w = 0.1 # 0.3rad/step or 18grad/step 
        self.max_dif = self.goal_angle - self.initial_angle

        # Status: angle of the second gripper
        self.angle = np.zeros(1)

        # Define action space: angle increment (normalized)
        self.action_space = spaces.Box(low = -1,
                                       high = 1,
                                       shape=(1,),
                                       dtype=np.float64)

        # Define observation space: current angle of second gripper
        self.observation_space = spaces.Box(low = -1,
                                            high = 1,
                                            shape=(1,),
                                            dtype=np.float64)

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.max_dif = self.goal_angle - self.initial_angle
        self.angle = np.copy(self.initial_angle)

        observation = np.copy(self.angle)
        info = {}

        return observation, info

    
    def step(self, action):
        action = np.clip(action, -1,1)
        self.angle = self.angle + action*self.max_w
        dif = np.abs(self.goal_angle[0] - self.angle[0])
        reward = -1 * dif / self.max_dif[0]

        terminated = False
        if dif < self.max_w:
            terminated = True

        observation = np.copy(self.angle)
        info = {}

        return observation, reward, terminated, False, info


    def render(self):
        print("___________")
        print("Angle: ", self.angle)

