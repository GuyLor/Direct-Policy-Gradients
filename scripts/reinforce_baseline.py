import numpy as np
import copy
import time
import torch
from torch.distributions import Categorical

import scripts.utils as utils
from scripts.minigrid_rl import MinigridRL


class ReinforceBaseline(MinigridRL):
    def __init__(self,
                 env_path,
                 chekpoint,
                 seed,
                 max_steps=240,
                 max_interactions=1000,
                 discount=0.99):
        super().__init__(env_path,chekpoint,seed,max_steps,max_interactions,discount)
        
        self.max_steps=max_steps
        self.discount = self.discount
        
        self.saved_log_probs = []
        self.rewards = []
        self.baseline = []
        self.policy.fc = torch.nn.Linear(64+4, self.num_actions+1)
        self.policy.log_softmax = Identity()
        self.load_checkpoint()
        #self.policy.log_softmax = torch.nn.Softmax(dim=-1)
    def select_action(self,state):
        #state = torch.from_numpy(state).float().unsqueeze(0)
        output = self.policy([state])
        scores,baseline = output[:,:self.num_actions],output[:,self.num_actions]
        probs = torch.nn.functional.softmax(scores,dim=-1)
        m = Categorical(probs)
        action = m.sample()

        self.saved_log_probs.append(m.log_prob(action))
        return action.item(),baseline.item()
    
    def finish_episode(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.discount * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        #rewards = (rewards - rewards.mean())/(rewards.std() + 1.2e-7)
        
        pred = torch.tensor(self.baseline)
        pred.requires_grad = True
        
        #return_base = alpha * rewards + (1-alpha)*return_base

        advantage = (rewards - pred).detach()
        baseline_loss = torch.nn.functional.mse_loss(pred,rewards)
        
        for log_prob, reward in zip(self.saved_log_probs, advantage):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()+baseline_loss
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.baseline[:]
        del self.saved_log_probs[:]

    def train(self,num_episodes=500,seed = 1234):
        success = 0
        to_plot = []
        repeat_map = self.max_interactions//self.max_steps
        episode = 0
        trial = 0
        count_interactions=0
        while count_interactions < self.max_interactions*num_episodes:
            trial+=1
            
            self.env.seed(seed)
            state = self.env.reset()
            actions=[]
            
            done=False
            
            while not done:
                action,b = self.select_action(state)
                state, reward, done, _ = self.env.step(action)
                self.baseline.append(b)
                self.rewards.append(reward)
                actions.append(action)
            
            self.finish_episode()
            count_interactions+=len(actions)
            trajectory_reward,suc = self.run_episode(actions,seed)
            success += suc
            

            print ('reinforce reward train: {:.3f},success: {}, length: {}'.format(trajectory_reward,suc,len(actions)))
            
            to_plot.append((count_interactions,trajectory_reward))
            
            if trial % repeat_map == 0:
                print('--------- new map {} -------------'.format(episode))
                seed+=1
                episode +=1
            if episode % 20 == 1:
                self.save_checkpoint()
        
        self.log['to_plot'] = to_plot
        self.save_checkpoint()
    
    def play(self,sample_opt = True,seed=1234,inarow=True,auto=True):

        def resetEnv(seed):
            self.env = self.reset()
            self.env.seed(seed)
            state = self.env.reset()
            done=False
            actions=[]
            
            while not done:
                action,b = self.select_action(state)
                state, reward, done, _ = self.env.step(action)
                actions.append(action)
            return actions
        while True:
            seed+=1
            actions = resetEnv(seed)
            super().play(actions,seed,auto=auto)
            if not inarow:
                break

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self,x):
        return x
