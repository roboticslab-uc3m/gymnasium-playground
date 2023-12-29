import gymnasium as gym
import numpy as np
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter
import math



class RelativeRingEnv(gym.Env):
    
    def __init__(self, render_mode=None, random_init=False, init_angle=0., goal_difference=0.5):
        #super(RelativeRingEnv, self).__init__()
        """
        The environment emulates a position in a numeric angle relative_ring, where the goal is to achieve a certain angle.
        """

        # Initial position
        self.random_init = random_init
        self.initial_angles = np.array([init_angle, init_angle])

        # goal
        self.goal_difference = np.array([goal_difference])

        # current step
        self.current_step =0

        # current reward
        self.current_reward = 0

        # set max vel 
        self.max_w = 0.1 
        self.max_dif = self.goal_difference #self.angular_distance(self.goal_difference, self.initial_angles)

        # Status: current_angle of the second gripper
        self.current_angle = np.zeros(2)

        # Define action space: current_angle increment (normalized)
        self.action_space = spaces.Box(low = -1,
                                       high = 1,
                                       shape=(2,),
                                       dtype=np.float64)

        # Define observation space: current current_angle 
        self.observation_space = spaces.Box(low = -1,
                                            high = 1,
                                            shape=(2,),
                                            dtype=np.float64)
        
        # summary writer
        self.writer = SummaryWriter()


    def reset(self, seed=None):
        super().reset(seed=seed)

        if self.random_init:
            self.initial_angles = self.norm_to_rad(np.random.uniform(-1, 1, 2))

        self.max_dif = self.goal_difference # self.angular_distance(self.goal_difference, self.initial_angles)
        self.current_angle = np.copy(self.initial_angles)
        self.current_step = 0

        observation = self.rad_to_norm(self.current_angle)
        info = {}

        return observation, info

    
    def step(self, action):
        self.current_step +=1
        action = np.clip(action, -1,1)
        self.current_angle = self.angular_clip(self.current_angle + action*self.max_w) #limits of the relative_ring
        dif = self.goal_difference - self.angular_distance(self.current_angle[0], self.current_angle[1])
        reward = -1 * abs(dif) / self.max_dif[0]
        penalization = 0.99**(self.current_step-1)
        reward *= penalization

        self.current_reward =reward
        # self.writer.add_scalar('reward_each_step', reward)

        terminated = False
        if dif < 2*self.max_w:
            terminated = True

        observation = self.rad_to_norm(self.current_angle)
        info = {}

        return observation, reward, terminated, False, info


    def render(self):
        print("___________")
        print("Current angle: ", self.current_angle)


    def norm_to_rad(self, norm_value):
        rad_value = (norm_value + 1) / 2 * 2 * math.pi
        return rad_value
    

    def rad_to_norm(self, rad_value):
        norm_value = rad_value / (2 * math.pi) * 2 - 1
        return norm_value
    
    def angular_distance(self, a1, a2):
        dist = min(abs(a1-a2), 2*math.pi - abs(a1-a2))
        return dist
    
    def angular_clip(self, angle):
        clipped_angle = angle % (2*math.pi)
        return clipped_angle

