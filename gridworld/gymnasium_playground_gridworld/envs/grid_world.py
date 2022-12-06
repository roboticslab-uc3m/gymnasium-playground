import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

"""
# Coordinate Systems for `.csv` and `print(numpy)`

X points down (rows); Y points right (columns); Z would point outwards.

*--> Y (columns)
|
v
X (rows)
"""

SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
COLOR_BACKGROUND = (0, 0, 0)
COLOR_WALL = (255, 255, 255)
COLOR_ROBOT = (255, 0, 0)

class GridWorldEnv(gym.Env):
    # classic "render_modes" use: "human" (pygame window), "rgb_array" (pygame raw data)
    metadata = {"render_modes": ["text", "pygame"], "render_fps": 4}

    def _2d_to_1d(self, value): # Flatten/ravel
        return value[0]*self.inFile.shape[1] + value[1]

    def __init__(self, render_mode=None):
        # Remember: See "Coordinate Systems for `.csv` and `print(numpy)`", above.
        # hard-coded vars (begin)
        inFileStr = 'map1.csv'
        initX = 2
        initY = 2
        goalX = 7
        goalY = 2
        # hard-coded vars (end)

        self.inFile = np.genfromtxt(inFileStr, delimiter=',')
        self.inFile[goalX][goalY] = 3 # The goal (3) is fixed, so we paint it, but the robot (2) moves, so done at render().

        self.nS = self.inFile.shape[0] * self.inFile.shape[1] # nS: number of states
        self.observation_space = spaces.Discrete(self.nS)

        self._action_to_direction = {
            0: np.array([-1,  0]),  # UP
            1: np.array([-1,  1]),  # UP_RIGHT
            2: np.array([ 0,  1]),  # RIGHT
            3: np.array([ 1,  1]),  # DOWN_RIGHT
            4: np.array([ 1,  0]),  # DOWN
            5: np.array([ 1, -1]),  # DOWN_LEFT
            6: np.array([ 0, -1]),  # LEFT
            7: np.array([-1, -1]),  # UP_LEFT
        }
        self.nA = 8 # nA: number of actions
        self.action_space = spaces.Discrete(self.nA)

        self.initial_state = np.array([initX, initY])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._2d_to_1d(self.state)

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random.
        super().reset(seed=seed)

        self.state = self.initial_state

        observation = self._get_obs()
        info = self._get_info()

        self.render()

        return observation, info

    def step(self, action):
        #print('GridWorldEnv.step', action)

        candidate_state = self.state + self._action_to_direction[action]
        candidate_state_tag = self.inFile[candidate_state[0]][candidate_state[1]]

        if candidate_state_tag == 0: # free space
            self.state = candidate_state
            reward = 0
            terminated = False
        elif candidate_state_tag == 1: # wall
            # state preserved
            reward = -0.5
            terminated = True
        elif candidate_state_tag == 3: # goal
            self.state = candidate_state
            reward = 1.0
            terminated = True
        else:
            print('GridWorldEnv.step: something wicked, please review!')
            terminated = True
            quit()

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self):
        #print('GridWorldEnv.render', self.render_mode)
        if self.render_mode == "text":
            return self._render_text()
        if self.render_mode == "pygame": # "human" in tutorial
            return self._render_pygame()
        else: # None
            pass

    def _render_text(self):
        viewer = np.copy(self.inFile) # Force a deep copy for rendering.
        viewer[self.state[0], self.state[1]] = 2
        print(viewer)

    def _render_pygame(self):

        if self.window is None and self.render_mode == "pygame":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        if self.clock is None and self.render_mode == "pygame":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        canvas.fill(COLOR_BACKGROUND)
        for iX in range(self.inFile.shape[0]):
            #print "iX:",iX
            for iY in range(self.inFile.shape[1]):
                #print "* iY:",iY

                pixelX = SCREEN_WIDTH/self.inFile.shape[0]
                pixelY = SCREEN_HEIGHT/self.inFile.shape[1]

                #-- Skip box if map indicates a 0
                if self.inFile[iX][iY] == 0:
                    continue
                if self.inFile[iX][iY] == 1:
                    pygame.draw.rect(canvas,
                                     COLOR_WALL,
                                     pygame.Rect( pixelX*iX, pixelY*iY, pixelX, pixelY ))
                if self.inFile[iX][iY] == 3:
                    pygame.draw.rect(canvas, (0,255,0),
                                     pygame.Rect( pixelX*iX, pixelY*iY, pixelX, pixelY ))
                robot = pygame.draw.rect(canvas, COLOR_ROBOT,
                                         pygame.Rect( pixelX*self.state[0]+pixelX/4.0, pixelY*self.state[1]+pixelY/4.0, pixelX/2.0, pixelY/2.0 ))

        # The following line copies our drawings from `canvas` to the
        # visible window.
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the
        # framerate stable.
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        print('GridWorldEnv.close')
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
