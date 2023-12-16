import gymnasium as gym
import numpy as np
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter



class LineEnv(gym.Env):
    
    def __init__(self, render_mode=None, random_init=False, init_position=0., goal_position=0.5):
        #super(LineEnv, self).__init__()
        """
        The environment emulates a position in anumeric line, where the goal is to achieve a certain position.
        """

        # Initial position
        self.random_init = random_init
        self.initial_position = np.array([init_position])

        # goal
        self.goal_position = np.array([goal_position])

        # current step
        self.current_step =0

        # set max vel 
        self.max_w = 0.1 # 0.3rad/step or 18grad/step 
        self.max_dif = self.goal_position - self.initial_position

        # Status: current_position of the second gripper
        self.current_position = np.zeros(1)

        # Define action space: current_position increment (normalized)
        self.action_space = spaces.Box(low = -1,
                                       high = 1,
                                       shape=(1,),
                                       dtype=np.float64)

        # Define observation space: current current_position of second gripper
        self.observation_space = spaces.Box(low = -1,
                                            high = 1,
                                            shape=(1,),
                                            dtype=np.float64)
        
        # summary writer
        self.writer = SummaryWriter()


    def reset(self, seed=None):
        super().reset(seed=seed)

        if self.random_init:
            self.initial_position = np.random.uniform(-1, 1, 1)

        self.max_dif = np.abs(self.goal_position - self.initial_position)
        self.current_position = np.copy(self.initial_position)
        self.current_step = 0

        observation = np.copy(self.current_position)
        info = {}

        return observation, info

    
    def step(self, action):
        self.current_step +=1
        action = np.clip(action, -1,1)
        self.current_position = np.clip(self.current_position + action*self.max_w, -1, 1) #limits of the line
        dif = np.abs(self.goal_position[0] - self.current_position[0])
        reward = -1 * dif / self.max_dif[0]
        penalization = 0.99**(self.current_step-1)
        reward *= penalization

        self.writer.add_scalar('reward_each_step', reward)

        terminated = False
        if dif < self.max_w:
            terminated = True

        observation = np.copy(self.current_position)
        info = {}

        return observation, reward, terminated, False, info


    def render(self):
        print("___________")
        print("Current position: ", self.current_position)

