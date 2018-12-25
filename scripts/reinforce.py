import numpy as np
import copy
import time
import torch
from torch.distributions import Categorical

import scripts.utils as utils
from scripts.minigrid_rl import MinigridRL

class Reinforce(MinigridRL):
    def __init__(self,
                 env_path,
                 chekpoint,
                 seed,
                 max_steps=240,
                 discount=0.99):
        super().__init__(env_path,chekpoint,seed,max_steps,discount)
        
        self.max_steps=max_steps
        self.discount = self.discount
        
        self.saved_log_probs = []
        self.rewards = []
        self.policy.log_softmax = torch.nn.Softmax(dim=-1)
    def select_action(self,state):
        #state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy([state])
        m = Categorical(probs)
        action = m.sample()

        self.saved_log_probs.append(m.log_prob(action))
        return action.item()
    
    def finish_episode(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.discount * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        #rewards = (rewards - rewards.mean())/(rewards.std() + 1.2e-7)
        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]

    def train(self,num_episodes=500,seed = 1234):
        success = 0
        
        for episode in range(num_episodes):
            self.env.seed(seed)
            state = self.env.reset()
            actions=[]
            done=False
            while not done:
                action = self.select_action(state)
                state, reward, done, _ = self.env.step(action)
                """
                if reward==0:
                    reward= -0.1
                """
                self.rewards.append(reward)
                actions.append(action)
            trajectory_reward,suc = self.run_episode(actions,seed)
            success += suc
            #rewards.append(trajectory_reward)
            
            self.finish_episode()
            seed+=1
            print('episode: ',episode)
            print ('reinforce reward: {:.3f},success: {}, length: {}'.format(trajectory_reward,suc,len(actions)))
            if episode % 20 == 1:

                self.save_checkpoint()
        self.save_checkpoint()

    
    def play(self,sample_opt = True,seed=1234,inarow=True,auto=True):

        def resetEnv(seed):
            self.env = self.reset()
            self.env.seed(seed)
            state = self.env.reset()
            done=False
            actions=[]
            
            while not done:
                action = self.select_action(state)
                state, reward, done, _ = self.env.step(action)
                actions.append(action)
            return actions
        while True:
            seed+=1
            actions = resetEnv(seed)
            super().play(actions,seed,auto=auto)
            if not inarow:
                break
