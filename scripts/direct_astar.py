import numpy as np
import copy
import time
import torch
import heapq
import inspect
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
    max_reward = 250 #100 for goal, 150= 5 doors (30 each)
    doors_togo = 5
    def __init__(self,
                 env,
                 states,
                 next_actions,
                 prefix=[],                 
                 reward_so_far=0,
                 done=False,
                 logprob_so_far=0,
                 max_gumbel=utils.sample_gumbel(0),
                 t_opt=True,
                 is_door = False,
                 dfs_like=True):
        
        self.env = env
        self.prefix = prefix
        self.t = len(self.prefix)
        
        self.states = states
        #self.discounted_reward_so_far = self.discounted_reward_so_far + reward_so_far[-1]*self.discount**self.t
        
        self.reward_so_far = reward_so_far
        self.done = done
        self.logprob_so_far = logprob_so_far
        self.max_gumbel = max_gumbel
        self.next_actions = next_actions
        
        self.t_opt = t_opt #true: opt, false: direct
        self.dfs_like =dfs_like
        
        if is_door:
            self.doors_togo-=1
        
        
        
        self.bound_reward_togo = self.set_bound_reward_togo()
        self.priority = self.set_priority()

    
    
    
    def __lt__(self, other):
        if self.t_opt == other.t_opt and not self.dfs_like: #false==false
            return self.priority >  other.priority
        elif self.t_opt or self.dfs_like:
            """
            This is how we sample t_opt, the starting node is with t_opt=True and
            its 'special child' will be also with t_opt=True and because we always returning true
            when we compare it to the 'other childrens' (where t_opt=False) the special path is sampled (speical childs only) 
            """
            return True
    def set_priority(self):
        return self.max_gumbel+self.epsilon*(self.reward_so_far + self.bound_reward_togo)
    
    def set_bound_reward_togo(self):
        return 100 * self.discount ** (self.t + self.manhattan_distance()) + (30*self.doors_togo )*self.discount**self.t
    
    def manhattan_distance(self):
        goal_pos = self.env.goal_pos
        agent_pos = self.env.agent_pos
        return np.sum(np.abs(goal_pos-agent_pos))





class DirectAstar(MinigridRL):
    def __init__(self,
                 env_path,
                 chekpoint,
                 seed,
                 update_wo_improvement=False,
                 keep_searching = False,
                 max_steps=240,
                 full_t_direct=False,
                 discount=0.99,
                 max_search_time=30,
                 max_interactions = 3000,
                 mixed_search_strategy = False,
                 dfs_like=False,
                 eps_grad=1.0,
                 eps_reward=3.0):
        super().__init__(env_path,chekpoint,seed,max_steps,max_interactions,discount)
        
        self.max_search_time=max_search_time
        self.max_interactions = max_interactions
        self.dfs_like=dfs_like
        self.mixed_search_strategy = mixed_search_strategy
        self.max_steps=max_steps
        self.epsilon = eps_grad
        Node.epsilon = eps_reward
        Node.discount = self.discount
        Node.doors_togo = len(self.env.rooms)-1
        self.full_t_direct=full_t_direct
        self.update_wo_improvement = update_wo_improvement # if true, updates even if priority(t_direct)<priority(t_opt)
        self.keep_searching = keep_searching # if true, keep searching for t_direct even if priority(t_direct)>priority(t_opt)

        self.log['priority_func'] = inspect.getsource(Node)
        self.log['search_proc'] = inspect.getsource(DirectAstar)

    def sample_t_opt_and_search_for_t_direct(self,inference=False,to_print=False):
    
    
        root_node = Node(
            env = self.env,
            states = [self.env.reset()],
            max_gumbel=utils.sample_gumbel(0),
            next_actions=range(self.num_actions))

        queue = []
        heapq.heappush(queue,root_node)
        best_direct ,max_gumbel= -float('Inf'),-float('Inf')
        
        final_trajectories = []
        start_time = float('Inf')
        start_search_direct = False
        lower_bound = -float('Inf')
        prune_count =0
        num_interactions=0
        dfs_like = self.dfs_like
        while queue:
            if to_print:
                print(10*'-')
                for q in queue:
                    print(q.priority,q.t_opt,q.dfs_like,'|',q.prefix,q.reward_so_far,q.logprob_so_far,q.max_gumbel,q.next_actions)
                    break

            parent = heapq.heappop(queue)
            #Sample t_opt without searching for t_direct
            if inference and not parent.t_opt:
                t_direct=None
                break
            #Start the search time count
            if not parent.t_opt and not start_search_direct:
                start_time = time.time()
                start_search_direct = True
            if not parent.t_opt:
                stop = parent.priority>t_opt.node.priority
                
            if self.mixed_search_strategy and num_interactions > self.max_interactions/2: # - self.max_steps:
                dfs_like = not self.dfs_like
            
            if  parent.done or (not parent.t_opt and stop and not self.full_t_direct):
                t = Trajectory(actions=parent.prefix,
                               states=parent.states,
                               gumbel=parent.max_gumbel,
                               reward=parent.reward_so_far,
                               status=parent.done,
                               node = parent)
                assert len(t.actions) == len(parent.states)-1
                
                if parent.max_gumbel+parent.reward_so_far>lower_bound:
                    lower_bound = parent.max_gumbel+parent.reward_so_far
                
                if parent.t_opt:
                    t_opt = t
            
                else:
                    final_trajectories.append(t)
                    if parent.priority > best_direct:
                        best_direct = parent.priority
                        t_direct = t
                        
                        if stop and not self.keep_searching or t_direct.reward == Node.max_reward:
                            print('*****  priority(direct) > priority(opt)   *****')
                            break
                continue
            
            if time.time()-start_time>self.max_search_time or num_interactions >= self.max_interactions:
                if len(final_trajectories)==0:
                    t_direct = Trajectory(actions=parent.prefix,
                               states=parent.states,
                               gumbel=parent.max_gumbel,
                               reward=parent.reward_so_far,
                               status=parent.done,
                               node = parent)
                    final_trajectories.append(t_direct)
                print("*****  time's-up/max interactions   *****")
                break
            

            current_state = parent.states[-1]
            with torch.no_grad():
                self.policy.eval()
                action_logprobs = self.policy([current_state]).cpu().numpy().squeeze(0)

            next_action_logprobs = action_logprobs[parent.next_actions]

            maxval,special_action_index = utils.sample_gumbel_argmax(next_action_logprobs)

            special_action = parent.next_actions[special_action_index]
            special_action_logprob = action_logprobs[special_action]

            env_copy = copy.deepcopy(parent.env) # do it here, before the step
            new_state,reward,done,info = parent.env.step(special_action)
            num_interactions += 1
            is_door = True if reward == 30 else False

            reward = reward*self.discount**len(parent.prefix)
            special_child = Node(
                                 env = parent.env,
                                 prefix=parent.prefix + [special_action],
                                 states=parent.states + [new_state],
                                 reward_so_far=parent.reward_so_far + reward,
                                 done=done,
                                 logprob_so_far=parent.logprob_so_far + special_action_logprob,
                                 max_gumbel=parent.max_gumbel,
                                 next_actions=range(self.num_actions),# All next actions are possible.
                                 t_opt = parent.t_opt,
                                 is_door = is_door,
                                 dfs_like = dfs_like)
                                 
            if special_child.priority < lower_bound:
                prune_count+=1
                continue
            else:
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
                                    t_opt = False,
                                    dfs_like = False)
                if other_children.priority < lower_bound:
                    prune_count+=1
                    continue
                else:
                    heapq.heappush(queue,other_children)
        if not inference:
            print ('pruned branches: {}, t_direct candidates: {}, nodes left in queue: {}, num interactions: {} '.format(prune_count,len(final_trajectories), len(queue),num_interactions))
        return t_opt, t_direct,num_interactions
        

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
        y_opt_direct = utils.use_gpu(y_opt_direct*(1.0/self.epsilon))

        policy_loss = torch.sum(y_opt_direct*phi)
        return policy_loss
    
    def direct_optimization_loss_normelized_phi(self,t_direct):

        direct_states  = t_direct.states[:-1]
        direct_actions = torch.LongTensor(t_direct.actions).view(-1,1)
        
        phi = self.policy(direct_states) # gets the logits so the network will calculates weights gradients
        y_direct =  -torch.FloatTensor(direct_actions.size(0),phi.size(1)).zero_().scatter_(-1,direct_actions,1.0) # one-hot which points to the best direction
        y_opt_direct = utils.use_gpu(y_direct*(1.0/self.epsilon))
        policy_loss = torch.sum(y_opt_direct*phi)
        return policy_loss

    def train(self,num_episodes=500,seed = 1234):
        success = 0
        count_interactions=0
        rewards_opt_direct = []
        priority_opt_direct = []
        lengths_opt_direct=[]
        interactions = []
        to_plot_opt = []
        to_plot_direct = []
        self.log['start_seed'] = seed
        
        episode=0
        while count_interactions < self.max_interactions*num_episodes:
            self.env.seed(seed)
            episode+=1
            print('--------- new map {} -------------'.format(episode))

            t_opt, t_direct,num_interactions = self.sample_t_opt_and_search_for_t_direct()
            
            if (t_direct.node.priority > t_opt.node.priority or self.update_wo_improvement) and len(t_direct.states)>1:
                opt,improvement = t_opt,t_direct
                #else:
                    #opt,improvement = t_direct,t_opt
                self.policy.train()
                #policy_loss = self.direct_optimization_loss(opt,improvement)
                policy_loss = self.direct_optimization_loss_normelized_phi(t_direct=improvement)
                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()
            interactions.append(num_interactions)
            opt_reward,suc = self.run_episode(t_opt.actions,seed)
            success += suc
            print ('opt reward: {:.3f},success: {}, length: {}, priority: {:.3f}'.format(opt_reward,suc,len(t_opt.actions),t_opt.node.priority))
            direct_reward,suc = self.run_episode(t_direct.actions,seed)
            print ('direct reward: {:.3f},success: {}, length: {}, priority: {:.3f}'.format(direct_reward,suc,len(t_direct.actions),t_direct.node.priority))
            
            to_plot_opt.append((count_interactions+len(t_opt.actions),opt_reward))
            to_plot_direct.append((count_interactions+num_interactions,direct_reward))
            
            rewards_opt_direct.append((opt_reward,direct_reward))
            lengths_opt_direct.append((len(t_opt.actions),len(t_direct.actions)))
            priority_opt_direct.append((t_opt.node.priority,t_direct.node.priority))
            
            count_interactions+=num_interactions
            seed+=1
            if episode % 20 == 1:
                self.save_checkpoint()
        self.save_checkpoint()

        self.log['interactions']=interactions
        self.log['rewards_opt_direct'] = rewards_opt_direct
        self.log['lengths_opt_direct'] = lengths_opt_direct
        self.log['priority_opt_direct'] = priority_opt_direct
        
        self.log['to_plot_opt'] =to_plot_opt
        self.log['to_plot_direct'] =to_plot_direct
        
        return success,rewards_opt_direct
        
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
