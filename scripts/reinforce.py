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
                 action_level=True,
                 max_steps=240,
                 max_interactions=1000,
                 discount=0.99):
        super().__init__(env_path,chekpoint,seed,max_steps,max_interactions,discount)
        
        self.max_steps=max_steps
        self.discount = self.discount
        
        self.saved_log_probs = []
        self.rewards = []
        self.policy.log_softmax = torch.nn.Softmax(dim=-1)
        self.alpha = 0.9
        self.return_base=0
        self.action_level = action_level
    def select_action(self,state):
        #state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy([state])
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()
    
    def finish_episode(self):
        
        policy_loss = []
        rewards = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.discount * R
            rewards.insert(0, R)
        
        
        rewards = torch.tensor(rewards)
        """
        rewards = rewards - self.return_base
        if self.trial> 1/(1-self.alpha):
            self.return_base = self.alpha * rewards + (1-self.alpha)*self.return_base
        else:
            self.return_base = self.return_base*(self.trial-1)/self.trial + rewards/self.trial
        """
        
        #rewards = (rewards - rewards.mean())/(rewards.std() + 1.2e-7)       
        
        for log_prob, reward in zip(self.saved_log_probs, rewards):
            if self.action_level:
                policy_loss.append(-log_prob * reward)
            else:
                policy_loss.append(-log_prob * R)
        
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        
        del self.rewards[:]
        del self.saved_log_probs[:]
        return R

    def train(self,num_episodes=500,seed = 1234):
        success = 0
        to_plot = []
        to_plot_avg =[]
        to_plot100 = []
        repeat_map = self.max_interactions//self.max_steps
        self.episode = 1
        self.trial,t = 0,0
        count_interactions=0
        total_interactions = 1e6
        avg_reward,avg_inter,suc_num= 0,0,0
        while count_interactions < self.max_interactions*num_episodes:
            self.trial+=1
            
            self.env.seed(seed)
            state = self.env.reset()
            actions=[]
            
            done=False
            suc = False
            t+=1
            while not done:

                action = self.select_action(state)
                state, reward, done, _ = self.env.step(action)
                self.rewards.append(reward)
                actions.append(action)
                if reward == self.target_reward:
                    suc = True
            
            trajectory_reward = self.finish_episode()
            count_interactions+=len(actions)
            #trajectory_reward,suc = self.run_episode(actions,seed)
            #success += suc
            
            #m = min(100,len(actions))
            #trajectory_reward_test,suc_test = self.run_episode(actions[:m],seed)
            avg_reward += trajectory_reward
            suc_num += suc
            #print ('reinforce reward train: {:.3f},success: {}, length: {}'.format(trajectory_reward,suc,len(actions)))
            #print ('reinforce reward test: {:.3f},success: {}, length: {}'.format(trajectory_reward_test,suc_test,len(actions[:m])))
            to_plot.append((count_interactions,trajectory_reward))
            #to_plot100.append((count_interactions,trajectory_reward_test))
            if self.trial % repeat_map == 0 or suc:
            
                self.optimizer.step()
                self.optimizer.zero_grad()
                print('--------- new map {} -------------'.format(self.episode))
                
                avg_reward/=t
                print ('reinforce reward train: {:.3f}, {} success out of {}'.format(avg_reward,suc_num,t))
                to_plot_avg.append((count_interactions,avg_reward))
                seed+=1
                self.episode +=1
                t=0
                avg_reward,suc_num= 0,0
            if self.episode % 20 == 1:
                self.save_checkpoint()
        
        self.log['to_plot'] = to_plot
        self.log['to_plot_avg'] = to_plot_avg
        #self.log['to_plot100'] = to_plot100
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


