import torch
import numpy as np
import gym
from gym_minigrid.register import env_list
import time
import sys

import policy_model as pm

class MinigridRL:
    def __init__(self,env_path,chekpoint,seed,max_steps=120,discount = 0.99):
        self.set_seed(seed)
        self.env_path =env_path
        self.max_steps=max_steps
        self.env = self.reset()
        
        self.num_actions=self.env.action_space.n
        
        self.discount=discount
        self.policy = pm.Policy(self.num_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001)
        self.cp = chekpoint
        self.load_checkpoint()
        
        self.map_actions = ['left','right','forward','pickup','drop','toggle','done']    
    def run_episode(self,actions_list,seed,target_reward=1.0):
        """Interacts with the environment given actions """
        rewards = 0
        done = False
        success = False
        self.env.seed(seed)
        state = self.env.reset()
        for action in actions_list:
            state, reward, done,info = self.env.step(action)
            if done and reward == target_reward:
                success = True
            rewards += reward
        return rewards,success
            
    def train(self):
        pass
    def reset(self):
        env = gym.make(self.env_path)
        env.max_steps =self.max_steps
        return env
    def set_seed(self,seed):
        np.random.seed(seed)
        #torch.manual_seed(seed)
    def load_checkpoint(self,filepath=None):
        if filepath is not None:
            self.cp.load_path = filepath
        self.cp.load(self.policy,self.optimizer)
    def save_checkpoint(self,filepath=None):
        if filepath is not None:
            self.cp.save_path = filepath
        self.cp.save(self.policy,self.optimizer)
    
    def play(self,actions,seed,auto = False):
        renderer = self.env.render('human')
        self.env.seed(seed)
        state = self.env.reset()
        done = False
        total_reward = 0
        it = iter(actions)
        def keyDownCb(keyName):
            nonlocal it,total_reward,done
            if keyName == 'ESCAPE':
                sys.exit(0)
            elif keyName == 'SPACE' and not auto:
                action = it.__next__()
            else:
                print("unknown key %s" % keyName)
                return
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            print_log(action,reward)

        def print_log(action,reward):
            print ('step:{} action: {}, reward: {:.2f}'.format(self.env.step_count,self.map_actions[action],reward))
            if done:
                print('done with total reward: ',total_reward)
        
        #if not auto:
        renderer.window.setKeyDownCb(keyDownCb)
        if auto:
            for action in actions:
                obs, reward, done, info = self.env.step(action)
                self.env.render('human')
                time.sleep(0.1)
                total_reward += reward
                print_log(action,reward)        
        while True:
            self.env.render('human')
            time.sleep(0.1)
            # If the window was closed
            if renderer.window == None or done:
                break


