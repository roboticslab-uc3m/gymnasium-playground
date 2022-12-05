import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

# X points down (rows)(v), Y points right (columns)(>), Z would point outwards.
LEFT = 0  # < Decrease Y (column)
DOWN = 1  # v Increase X (row)
RIGHT = 2 # > Increase Y (column)
UP = 3    # ^ Decrease X (row)
CORNER_UP_RIGHT=4
CORNER_UP_LEFT=5
CORNER_DOWN_RIGHT=6
CORNER_DOWN_LEFT=7

SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
COLOR_BACKGROUND = (0, 0, 0)
COLOR_WALL = (255, 255, 255)
COLOR_ROBOT = (255, 0, 0)

class GridWorldEnv(gym.Env):
    # classic "render_modes" use: "human" (pygame window), "rgb_array" (pygame raw data)
    metadata = {"render_modes": ["text", "pygame"], "render_fps": 4}

    def __init__(self, render_mode=None):
        # Remember: X points down, Y points right, thus Z points outwards.
        # hard-coded vars (begin)
        inFileStr = 'map1.csv'
        initX = 2
        initY = 2
        goalX = 7
        goalY = 2
        # hard-coded vars (end)

        self.inFile = np.genfromtxt(inFileStr, delimiter=',')
        self.inFile[goalX][goalY] = 3 # The goal (3) is fixed, so we paint it, but the robot (2) moves, so done at render().

        self.nrow, self.ncol = nrow, ncol = self.inFile.shape
        self.nS = nrow * ncol # nS: number of states
        self.observation_space = spaces.Discrete(self.nS)

        self.nA = 8 # nA: number of actions
        self.action_space = spaces.Discrete(self.nA)

        self.P = {s : {a : [] for a in range(self.nA)} for s in range(self.nS)} # transitions (*), filled in at the for loop below.

        def _to_s(row, col):
            return row*ncol + col

        self.initial_state = _to_s(initX, initY)

        def _inc(row, col, a): # Assures we will not go off limits.
            if a == LEFT:
                col = max(col-1,0)
            elif a == DOWN:
                row = min(row+1,nrow-1)
            elif a == RIGHT:
                col = min(col+1,ncol-1)
            elif a == UP:
                row = max(row-1,0)
            elif a == CORNER_UP_RIGHT:
                row = max(row-1,0)
                col = min(col+1,ncol-1)
            elif a == CORNER_UP_LEFT:
                row = max(row-1,0)
                col = max(col-1,0)
            elif a == CORNER_DOWN_RIGHT:
                row = min(row+1,nrow-1)
                col = min(col+1,ncol-1)
            elif a == CORNER_DOWN_LEFT:
                row = min(row+1,nrow-1)
                col = max(col-1,0)
            return (row, col)

        for row in range(nrow): # Fill in P[s][a] transitions and rewards
            for col in range(ncol):
                s = _to_s(row, col)
                for a in range(self.nA):
                    li = self.P[s][a] # In Python this is not a deep copy, therefore we are appending to actual P[s][a] !!
                    tag = self.inFile[row][col]
                    if tag == 3: # goal
                        li.append((1.0, s, 1.0, True)) # (probability, nextstate, reward, done)
                    elif tag == 1: # wall
                        li.append((1.0, s, -0.5, True)) # (probability, nextstate, reward, done) # Some algorithms fail with reward -float('inf')
                    else: # e.g. tag == 0
                        newrow, newcol = _inc(row, col, a)
                        newstate = _to_s(newrow, newcol)
                        li.append((1.0, newstate, 0.0, False)) # (probability, nextstate, reward, done)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self.s

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random.
        super().reset(seed=seed)

        self.s = self.initial_state

        observation = self._get_obs()
        info = self._get_info()

        self.render()

        return observation, info

    def step(self, action):
       _, s, reward, terminated = self.P[self.s][action][0]
       self.s = s

       observation = self._get_obs()
       info = self._get_info()

       return observation, reward, terminated, False, info

    def render(self):
        #print('CsvEnv.render', self.render_mode)
        if self.render_mode == "text":
            return self._render_text()
        if self.render_mode == "pygame": # "human" in tutorial
            return self._render_pygame()
        else: # None
            pass

    def _render_pygame(self):

        if self.window is None and self.render_mode == "pygame":
            pygame.init()
            pygame.display.init()
            pygame = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        if self.clock is None and self.render_mode == "pygame":
            self.clock = pygame.time.Clock()

        row, col = self.s // self.ncol, self.s % self.ncol # Opposite of ravel().

        canvas = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        canvas.fill(COLOR_BACKGROUND)
        for iX in range(self.nrow):
            #print "iX:",iX
            for iY in range(self.ncol):
                #print "* iY:",iY

                pixelX = SCREEN_WIDTH/self.nrow
                pixelY = SCREEN_HEIGHT/self.ncol

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
                                         pygame.Rect( pixelX*row+pixelX/4.0, pixelY*col+pixelY/4.0, pixelX/2.0, pixelY/2.0 ))

        # The following line copies our drawings from `canvas` to the
        # visible window.
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the
        # framerate stable.
        self.clock.tick(self.metadata["render_fps"])

    def _render_text(self):
        row, col = self.s // self.ncol, self.s % self.ncol # Opposite of ravel().
        viewer = np.copy(self.inFile) # Force a deep copy for rendering.
        viewer[row, col] = 2
        print(viewer)

    def close(self):
        print('CsvEnv.close')
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
