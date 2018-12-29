import numpy as np
import copy
import time
import torch
import heapq

import scripts.utils as utils
from scripts.minigrid_rl import MinigridRL

class Trajectory:
    def __init__(self,actions,states,gumbel,reward,status,node):
        self.actions=actions
        self.states=states
        self.gumbel=gumbel
        self.reward=reward
        self.status=status
        self.node=node

class Node:
    epsilon = 1.0
    discount = 0.99
    def __init__(self,
                 env,
                 states,
                 next_actions,
                 prefix=[],                 
                 reward_so_far=0,
                 done=False,
                 logprob_so_far=0,
                 max_gumbel=utils.sample_gumbel(0),
                 t_opt=True):
        
        self.env = env
        self.prefix = prefix
        self.states = states
        self.reward_so_far = reward_so_far
        self.done = done
        self.logprob_so_far = logprob_so_far
        self.max_gumbel = max_gumbel
        self.next_actions = next_actions
        
        self.t_opt = t_opt #true: opt, false: direct
        self.priority = self.epsilon*self.reward_so_far*self.discount**len(self.prefix)+ self.max_gumbel
        

    def __lt__(self, other):
        if self.t_opt == other.t_opt: #false==false
            return self.priority >  other.priority
        
        elif self.t_opt:
            """
            This is how we sample t_opt, the starting node is with t_opt=True and
            its 'special child' will be also with t_opt=True and because we always returning true
            when we compare it to the 'other childrens' (where t_opt=False) the special path is sampled (speical childs only) 
            """
            return True

class DirectAstar(MinigridRL):
    def __init__(self,
                 env_path,
                 chekpoint,
                 seed,
                 update_wo_improvement=False,
                 keep_searching = False,
                 max_steps=240,
                 discount=0.99,
                 max_search_time=30,
                 eps_grad=1.0,
                 eps_reward=3.0):
        super().__init__(env_path,chekpoint,seed,max_steps,discount)
        
        self.max_search_time=max_search_time
        self.max_steps=max_steps
        self.epsilon = eps_grad
        Node.epsilon = eps_reward
        Node.discount = self.discount
        self.update_wo_improvement = update_wo_improvement # if true, updates even if priority(t_direct)<priority(t_opt)
        self.keep_searching = keep_searching # if true, keep searching for t_direct even if priority(t_direct)>priority(t_opt)
    
    def sample_t_opt_and_search_for_t_direct(self,inference=False,to_print=False):
        """Samples an independent Gumbel(logprob) for each trajectory in top-down-ish order.
        Args:
            policy_model: Pytorch model that gets state and returns actions logits
            environment: Initial environment that initialized with random seed
            num_actions: Number of possible actions, should be equal to the policy output size
            max_trajectory_length: Maximum length of a trajectory to allow.
            max_search_time: Maximum searching time of t_direct
            epsilon: direct optimization parameter
            inference: sampling t_opt without searching for t_direct
        Returns:
            y_opt: trajectory of max gumbel
            y_direct: trajectory of max gumbel + reward*epsilon (target)
            final_trajectories: int, number of all trajectories (can be deleted, just for debugging)
        """        
        root_node = Node(
            env = self.env,
            states = [self.env.reset()],
            max_gumbel=utils.sample_gumbel(0),
            next_actions=range(self.num_actions))
        
        queue = []
        heapq.heappush(queue,root_node)
        max_gumbel_eps_reward ,max_gumbel= -float('Inf'),-float('Inf')
        
        final_trajectories = []
        start_time = float('Inf')
<<<<<<< HEAD
        flag = True
=======
        flag=True
>>>>>>> 860ad91dcc019fe0b514a2a99a57e3f7c2febc28
        while queue:
            if to_print:
                print(10*'-')
                for q in queue:
                    print(q.priority,q.t_opt,'|',q.prefix,q.reward_so_far,q.logprob_so_far,q.max_gumbel,q.next_actions)
<<<<<<< HEAD
        
=======
                    
>>>>>>> 860ad91dcc019fe0b514a2a99a57e3f7c2febc28
            parent = heapq.heappop(queue)
            
            if inference and not parent.t_opt:
                t_direct=None
                break
<<<<<<< HEAD
            if not parent.t_opt and flag:
                start_time = time.time()
                flag = False
=======
                
            if not parent.t_opt and flag:
                start_time = time.time()
                flag = False
                
>>>>>>> 860ad91dcc019fe0b514a2a99a57e3f7c2febc28
            if  parent.done or (not parent.t_opt and parent.priority>t_opt.node.priority):
                t = Trajectory(actions=parent.prefix,
                               states=parent.states,
                               gumbel=parent.max_gumbel,
                               reward=parent.reward_so_far,
                               status=parent.done,
                               node = parent)
                assert len(t.actions) == len(parent.states)-1
                if parent.t_opt:
                    t_opt = t
                else:
                    final_trajectories.append(t)
                    if parent.priority>max_gumbel_eps_reward:
                        max_gumbel_eps_reward = parent.priority
                        t_direct = t
                        if  parent.priority>t_opt.node.priority and not self.keep_searching:
                            if to_print:
                                print('stop!!', parent.done,parent.priority,value_to_stop)
                                print('*'*100)
                            break
                continue
            if time.time()-start_time>self.max_search_time:
                if len(final_trajectories)==0:
                    t_direct = Trajectory(actions=parent.prefix,
                               states=parent.states,
                               gumbel=parent.max_gumbel,
                               reward=parent.reward_so_far,
                               status=parent.done,
                               node = parent)
                    final_trajectories.append(t_direct)
                break

            current_state = parent.states[-1]
            with torch.no_grad():
                self.policy.eval()
                action_logprobs = self.policy([current_state]).numpy().squeeze(0)

            next_action_logprobs = action_logprobs[parent.next_actions]

            maxval,special_action_index = utils.sample_gumbel_argmax(next_action_logprobs)

            special_action = parent.next_actions[special_action_index]
            special_action_logprob = action_logprobs[special_action]

            env_copy = copy.deepcopy(parent.env) # do it here, before the step
            new_state,reward,done,info = parent.env.step(special_action)
            reward = 100*reward
            special_child = Node(
                                 env = parent.env,
                                 prefix=parent.prefix + [special_action],
                                 states=parent.states + [new_state],
                                 reward_so_far=parent.reward_so_far + reward,
                                 done=done,
                                 logprob_so_far=parent.logprob_so_far + special_action_logprob,
                                 max_gumbel=parent.max_gumbel,
                                 next_actions=range(self.num_actions),# All next actions are possible.
                                 t_opt = parent.t_opt)

            heapq.heappush(queue,special_child)
            # Sample the max gumbel for the non-chosen actions and create an "other
            # children" node if there are any alternatives left.

            other_actions = [i for i in parent.next_actions if i != special_action]
            assert len(other_actions) == len(parent.next_actions) - 1

            if other_actions:
                other_max_location = utils.logsumexp(action_logprobs[other_actions])
                other_max_gumbel = utils.sample_truncated_gumbel(parent.logprob_so_far + other_max_location,parent.max_gumbel)
                other_children = Node(
                                    env = env_copy,
                                    prefix=parent.prefix,
                                    states=parent.states,
                                    reward_so_far=parent.reward_so_far,
                                    done=parent.done,
                                    logprob_so_far=parent.logprob_so_far,
                                    max_gumbel=other_max_gumbel,
                                    next_actions=other_actions,
                                    t_opt = False)
            
                heapq.heappush(queue,other_children)
        return t_opt, t_direct,final_trajectories

    def direct_optimization_loss(self,t_opt,t_direct):
        """computes \nabla \phi(a,s) = \sum_{t=1}^T \nabla \phi(a_t, s_t) with direct optimization
        
        Args:
            policy_model: Pytorch model gets state and returns actions logits
            t_opt: trajectory with the max(gumbel)
            t_direct: trajectory with the max(gumbel+epsilon*reward)
            epsilon: for direct optimization, usually between 0.9-1.0
        Returns:
            logits multiplied with vector of (1,0,..,-1) [for example]
            so when we derive it with backward we ends up with \nabla \phi_opt(a,s) - \nabla \phi_direct(a,s)"""
        direct_states  = t_direct.states[:-1]
        opt_states  = t_opt.states[:-1]
        opt_direct_states = (opt_states+direct_states)
        
        direct_actions = torch.LongTensor(t_direct.actions).view(-1,1)
        opt_actions = torch.LongTensor(t_opt.actions).view(-1,1)
        
        phi = self.policy(opt_direct_states) # gets the logits so the network will calculates weights gradients
        y_direct =  -torch.FloatTensor(direct_actions.size(0),phi.size(1)).zero_().scatter_(-1,direct_actions,1.0) # one-hot which points to the best direction
        y_opt = torch.FloatTensor(opt_actions.size(0),phi.size(1)).zero_().scatter_(-1, opt_actions,1.0)
        
        y_opt_direct = torch.cat((y_opt,y_direct))
        y_opt_direct = y_opt_direct*(1.0/self.epsilon)

        policy_loss = torch.sum(y_opt_direct*phi)
        return policy_loss
    
    def direct_optimization_loss_normelized_phi(self,t_direct):

        direct_states  = t_direct.states[:-1]
        direct_actions = torch.LongTensor(t_direct.actions).view(-1,1)
        
        phi = self.policy(direct_states) # gets the logits so the network will calculates weights gradients
        y_direct =  -torch.FloatTensor(direct_actions.size(0),phi.size(1)).zero_().scatter_(-1,direct_actions,1.0) # one-hot which points to the best direction
        y_opt_direct = y_direct*(1.0/self.epsilon)
        policy_loss = torch.sum(y_opt_direct*phi)
        return policy_loss

    def train(self,num_episodes=500,seed = 1234):
        success = 0
        rewards = []
        
        for episode in range(num_episodes):
            self.env.seed(seed)
            print('episode: ',episode)
            t_opt, t_direct,final_trajectories = self.sample_t_opt_and_search_for_t_direct()
            
            if t_direct.node.priority > t_opt.node.priority or self.update_wo_improvement:
                opt,improvement = t_opt,t_direct
            else:
                opt,improvement = t_direct,t_opt
            self.policy.train()
            #policy_loss = self.direct_optimization_loss(opt,improvement)
            policy_loss = self.direct_optimization_loss_normelized_phi(t_direct=improvement)
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
                
            opt_reward,suc = self.run_episode(t_opt.actions,seed)
            success += suc
            print ('opt reward: {:.3f},success: {}, length: {}, priority: {:.3f}'.format(opt_reward,suc,len(t_opt.actions),t_opt.node.priority))
            t_direct_reward,suc = self.run_episode(t_direct.actions,seed)
            print ('direct reward: {:.3f},success: {}, length: {}, priority: {:.3f}'.format(t_direct_reward,suc,len(t_direct.actions),t_direct.node.priority))
            rewards.append(opt_reward)
            seed+=1
            if episode % 20 == 1:
                self.save_checkpoint()
        self.save_checkpoint()
        return success,rewards
        
    def play(self,sample_opt = True,seed=1234,inarow=True,auto=True):

        def resetEnv(seed):
            self.env = self.reset()
            self.env.seed(seed)
            t_opt,t_direct,_ = self.sample_t_opt_and_search_for_t_direct(inference =sample_opt)
            actions = t_opt.actions if sample_opt else t_direct.actions
            return actions
        while True:
            seed+=1
            actions = resetEnv(seed)
            super().play(actions,seed,auto=auto)
            if not inarow:
                break
