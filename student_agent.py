import gym
from Env import State_Frame
from model import DQNAgent
import random
# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)

        self.frames = State_Frame()
        self.agent = DQNAgent(240,12,"cpu")
        self.agent.load_model("q_best_network.pth","target_best_network.pth")
        print("load")
        self.skip_frames = 5
        self.steps = 0
        self.action = None
    def act(self, observation):
        if self.steps%self.skip_frames == 0:
            self.frames.add_frame(observation)
            self.action = self.agent.get_action(self.frames.get(),True)
    
        #print(observation)
        self.steps += 1
        
        #print(self.action)
        return self.action