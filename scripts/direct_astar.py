import numpy as np
import copy
import time
import torch
import heapq
import inspect
import math,sys
import pdb
DEBUG = False
import scripts.utils as utils
from scripts.minigrid_rl import MinigridRL


def discounted_reward(reward, discount_factor, timestep):
    return reward * (discount_factor ** timestep)

class Trajectory:
    def __init__(self,actions,states,gumbel,reward,status,node):
        self.actions=actions
        self.states=states
        self.gumbel=gumbel
        self.reward=reward
        self.status=status
        self.node=node


class Node:
    goal_reward = 100  # 100 for goal, 150= 5 doors (30 each)
    epsilon = 1.0
    discount = .999
    alpha = 0.2
    def __init__(self,
                 env,
                 states,
                 next_actions,
                 parent_doors_togo=5,
                 prefix=[],
                 parent_reward_so_far=0,
                 undiscounted_reward=0,
                 rewards_list=[],
                 done=False,
                 logprob_so_far=0,
                 max_gumbel=None,
                 t_opt=True,
                 t_opt_objective=None,  # How much total objective t_opt achieved.
                 dfs_like=True):

        if max_gumbel is None:
            max_gumbel = utils.sample_gumbel(0)
    
        self.env = env
        self.prefix = prefix
        self.t = len(self.prefix)
        self.undiscounted_reward = undiscounted_reward

        self.rewards_list = rewards_list
        self.states = states
        self.reward_so_far = parent_reward_so_far + discounted_reward(undiscounted_reward, self.discount, self.t)

        self.t_opt_objective = t_opt_objective
        self.done = done
        self.logprob_so_far = logprob_so_far
        self.max_gumbel = max_gumbel
        self.next_actions = next_actions
        
        self.t_opt = t_opt #true: opt, false: direct
        self.dfs_like =dfs_like
        
        if undiscounted_reward == 30 :
            self.doors_togo = parent_doors_togo - 1
        else:
            self.doors_togo = parent_doors_togo

        self.bound_reward_togo = self.get_bound_reward_togo()
        self.upper_bound = self.get_priority_alpha_upper_bound(1.0)
        self.priority = self.get_priority_alpha_upper_bound(self.alpha)
        self.objective = self.get_objective()

    def __lt__(self, other):
        if self.t_opt == other.t_opt and not self.dfs_like: #false==false
            return self.priority > other.priority
        elif self.t_opt or self.dfs_like:
            """
            This is how we sample t_opt, the starting node is with t_opt=True and
            its 'special child' will be also with t_opt=True and because we always returning true
            when we compare it to the 'other childrens' (where t_opt=False) the special path is sampled (speical childs only)
            """
            return True

    def get_priority(self):
            return self.get_objective()

    def get_priority_max_gumbel(self):
            return self.max_gumbel

    def get_priority_alpha_upper_bound(self,alpha):
        return self.max_gumbel + self.epsilon * (self.reward_so_far + alpha*self.bound_reward_togo)

    def get_upper_bound(self):
        return self.max_gumbel + self.epsilon * (self.reward_so_far + self.bound_reward_togo)
    
    def get_bound_reward_togo(self):
        next_timestep = self.t + 1
        # It takes two actions to get through a door: toggle, then move through.
        upper_bound = discounted_reward(self.goal_reward, self.discount, next_timestep + self.doors_togo + self.manhattan_distance())
        
        for k in range(self.doors_togo):
            upper_bound += discounted_reward(30, self.discount, next_timestep + k)
        
        return upper_bound

    def get_objective(self):
        """Computes the objective of the trajectory.
        Only used if a node is terminal.
        """
        return self.max_gumbel + self.epsilon * self.reward_so_far
    
    def manhattan_distance(self):
        goal_pos = self.env.goal_pos
        agent_pos = self.env.agent_pos
        return np.sum(np.abs(goal_pos-agent_pos))

    def bound_info(self):
        """A tuple with the components that go into computing the bound."""
        return (self.max_gumbel, self.epsilon * self.reward_so_far, self.epsilon * self.bound_reward_togo, self.t, self.manhattan_distance(), self.doors_togo)


class DirectAstar(MinigridRL):
    def __init__(self,
                 env_path,
                 chekpoint,
                 seed,
                 independent_sampling = False,
                 keep_searching = False,
                 max_steps=240,
                 discount=0.99,
                 alpha=0.2,
                 max_search_time=30,
                 max_interactions = 3000,
                 mixed_search_strategy = False,
                 optimization_method = 'direct',
                 dfs_like=False,
                 eps_grad=1.0,
                 eps_reward=3.0):
        super().__init__(env_path,chekpoint,seed,max_steps,max_interactions,discount)
        
        self.max_search_time=max_search_time
        self.max_interactions = max_interactions
        self.independent_samplimg = independent_sampling
        self.dfs_like=dfs_like
        self.mixed_search_strategy = mixed_search_strategy
        self.max_steps=max_steps
        self.alpha=alpha,
        self.eps_grad = eps_grad
        self.eps_reward = eps_reward
        Node.epsilon = eps_reward
        Node.discount = self.discount
        Node.alpha = self.alpha[0]
        self.break_on_goal = True
        # Node.doors_togo = len(self.env.rooms)-1
        self.max_interactions = self.max_steps*30
        self.keep_searching = keep_searching # if true, keep searching for t_direct even if priority(t_direct)>priority(t_opt)
        self.optimization_method = optimization_method
        self.log['priority_func'] = inspect.getsource(Node)
        self.log['search_proc'] = inspect.getsource(DirectAstar)
    
    def sample_trajectories(self, share_gumbels = True):
        num_interactions=0
        final_trajectories = []
        reached_the_goal = 0
        max_gumbel = utils.sample_gumbel(0)
        while num_interactions < self.max_interactions:
            self.env.seed(self.seed)
            if not share_gumbels:
                max_gumbel = utils.sample_gumbel(0)
            root_node = Node(
                env = self.env,
                states = [self.env.reset()],
                max_gumbel=max_gumbel,
                next_actions=range(self.num_actions),
                t_opt=True)
            queue = [root_node]
            
            while queue:
                parent = queue.pop()
                if parent.done:
                    status = True if parent.undiscounted_reward == Node.goal_reward else False
                    t = Trajectory(actions=parent.prefix,
                                   states=parent.states,
                                   gumbel=parent.max_gumbel,
                                   reward=parent.reward_so_far,
                                   status=status,
                                   node = parent)
                    final_trajectories.append(t)

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

                special_child = Node(
                                     env = parent.env,
                                     prefix=parent.prefix + [special_action],
                                     states=parent.states + [new_state],
                                     parent_reward_so_far=parent.reward_so_far,
                                     undiscounted_reward=reward,
                                     rewards_list=parent.rewards_list + [reward],
                                     done=done,
                                     logprob_so_far=parent.logprob_so_far + special_action_logprob,
                                     max_gumbel=parent.max_gumbel,
                                     next_actions=range(self.num_actions),# All next actions are possible.
                                     parent_doors_togo=parent.doors_togo,
                                     t_opt = parent.t_opt)
                                     
                queue.append(special_child)
                
            if final_trajectories[-1].status:
                '******* GOAL BREAK *******'
                break
    
        t_opt = max(final_trajectories,key=lambda x: x.gumbel)
        t_direct = max(final_trajectories,key=lambda x: x.gumbel+self.eps_reward*x.reward)
        print ('num candidates: {}, num interactions: {} '.format(len(final_trajectories),num_interactions))
        return t_opt,t_direct,final_trajectories,num_interactions
        
    def sample_t_opt_and_search_for_t_direct(self,inference=False,to_print=False):
        self.env.seed(self.seed)
        root_node = Node(
            env = self.env,
            states = [self.env.reset()],
            max_gumbel=utils.sample_gumbel(0),
            next_actions=range(self.num_actions),
            t_opt=True)

        queue = []
        heapq.heappush(queue,root_node)
        
        final_trajectories = []
        start_time = float('Inf')
        start_search_direct = False
        prune_count =0
        num_interactions=0
        dfs_like = self.dfs_like
        t_opt = None  # Will get set when we find t_opt.
        t_direct = None  # Will get set when we find t_opt.
        longest_prefix_with_reward = 0
        num_popped_nodes_with_reward = 0

        while queue:
            if to_print:
                print(10*'-')
                for q in queue:
                    print(q.priority,q.t_opt,q.dfs_like,'|',q.prefix,q.reward_so_far,q.logprob_so_far,q.max_gumbel,q.next_actions)
                    break

            parent = heapq.heappop(queue)
            t_opt_objective = t_opt.node.objective if t_opt is not None else None
            lower_bound_objective = -float('Inf') if t_direct is None else t_direct.node.objective
            if DEBUG: print(t_opt_objective, lower_bound_objective, parent.priority, parent.bound_info())

            if parent.reward_so_far > 0:
                longest_prefix_with_reward = max([longest_prefix_with_reward, len(parent.prefix)])
                num_popped_nodes_with_reward += 1

            if lower_bound_objective > parent.upper_bound:
                prune_count += 1
                continue

            #Sample t_opt without searching for t_direct
            if inference and not parent.t_opt:
                t_direct=None
                break

            #Start the search time count
            if not parent.t_opt and not start_search_direct:
                start_time = time.time()
                start_search_direct = True
            
            if self.mixed_search_strategy and num_interactions > self.max_interactions- self.max_steps:
                dfs_like = not self.dfs_like
            
            if parent.done:
                status = True if parent.undiscounted_reward == Node.goal_reward else False
                t = Trajectory(actions=parent.prefix,
                               states=parent.states,
                               gumbel=parent.max_gumbel,
                               reward=parent.reward_so_far,
                               status=status,
                               node = parent)

                assert len(t.actions) == len(parent.states)-1
                final_trajectories.append(t)
                if parent.t_opt:
                    t_opt = t
                    t_direct = t
                else:
                    if DEBUG and t.reward > 0:
                        print(t_opt.node.objective, t.node.objective)
                        print(parent.bound_info())
                        pdb.set_trace()
                
                    if t.node.objective > t_direct.node.objective:
                        t_direct = t
                        if not self.keep_searching:
                            print('*****  priority(direct) > priority(opt)   *****')
                            break
                if t.status and self.break_on_goal:
                    print('*****  GOAL BREAK   *****')
                    break
                continue
            
            if time.time()-start_time>self.max_search_time or num_interactions >= self.max_interactions:
                #print("*****  time's-up/max interactions   *****")
                
                # Change to True if you want to consider prefix. This can be much better in some cases.
                if False and len(final_trajectories)==1 :
                    print('take the best prefix')
                    t_direct = Trajectory(actions=parent.prefix,
                               states= parent.states,
                               gumbel=parent.max_gumbel,
                               reward=parent.reward_so_far,
                               status=parent.done,
                               node = parent)
                    final_trajectories.append(t_direct)
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

            if DEBUG and parent.reward_so_far > 0:
                pdb.set_trace()

            special_child = Node(
                                 env = parent.env,
                                 prefix=parent.prefix + [special_action],
                                 states=parent.states + [new_state],
                                 parent_reward_so_far=parent.reward_so_far,
                                 undiscounted_reward=reward,
                                 rewards_list=parent.rewards_list + [reward],
                                 done=done,
                                 logprob_so_far=parent.logprob_so_far + special_action_logprob,
                                 max_gumbel=parent.max_gumbel,
                                 next_actions=range(self.num_actions),# All next actions are possible.
                                 parent_doors_togo=parent.doors_togo,
                                 t_opt = parent.t_opt,
                                 t_opt_objective=t_opt_objective,
                                 dfs_like = dfs_like)
                                 
            if special_child.upper_bound < lower_bound_objective:
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
                                    parent_reward_so_far=parent.reward_so_far,
                                    rewards_list=parent.rewards_list,
                                    done=parent.done,
                                    logprob_so_far=parent.logprob_so_far,
                                    max_gumbel=other_max_gumbel,
                                    next_actions=other_actions,
                                    parent_doors_togo=parent.doors_togo,
                                    t_opt = False,
                                    t_opt_objective=t_opt_objective,
                                    dfs_like = False)
                if DEBUG and parent.reward_so_far > 0:
                    for node in [parent, special_child, other_children]:
                        print(node.upper_bound, node.bound_info())
                    pdb.set_trace()

                if other_children.upper_bound < lower_bound_objective:
                    prune_count+=1
                    continue
                else:
                    heapq.heappush(queue,other_children)
        if not inference:
            print ('pruned branches: {}, t_direct candidates: {}, nodes left in queue: {}, num interactions: {} '.format(prune_count,len(final_trajectories), len(queue),num_interactions))

        return t_opt, t_direct,final_trajectories,num_interactions
        
    def get_one_side_loss(self,t,is_direct):
        states  = t.states[:-1]
        actions = torch.LongTensor(t.actions).view(-1,1)
        
        phi = self.policy(states) # gets the logits so the network will calculates weights gradients
        
        if is_direct:
            y = -torch.FloatTensor(actions.size(0),phi.size(1)).zero_().scatter_(-1,actions,1.0)
        else:
            y =  torch.FloatTensor(actions.size(0),phi.size(1)).zero_().scatter_(-1,actions,1.0)
        
        y_opt_direct = utils.use_gpu(y)
        policy_loss = torch.sum(y_opt_direct*phi)/self.eps_grad
        return policy_loss
        
    def direct_optimization_loss(self,t_opt,t_direct):
        """computes \nabla \phi(a,s) = \sum_{t=1}^T \nabla \phi(a_t, s_t) with direct optimization
        Args:
            policy_model: Pytorch model gets state and returns actions logits
            t_opt: trajectory with the max(gumbel)
            t_direct: trajectory with the max(gumbel+epsilon*reward)
            normelized_phi: if True, uses only the 'direct' side
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
        y_opt_direct = utils.use_gpu(y_opt_direct) *(1.0/self.eps_grad)
        policy_loss = torch.sum(y_opt_direct*phi)
        return policy_loss

    def cross_entropy_loss(self,final_trajectories, elite_frac = 0.05):
        final_trajectories.sort(key=lambda x: x.reward, reverse=True)
        end = math.ceil(len(final_trajectories)*elite_frac)
        print ('len(final_trajectories): ',len(final_trajectories[:end]))
        ce_loss = 0
        for t in final_trajectories[:end]:
            ce_loss += self.get_one_side_loss(t,is_direct=True)
        return ce_loss

    def train(self,num_episodes=500,seed = 1234):
        
        self.seed =seed
        rewards_opt_direct = []
        priority_opt_direct = []
        lengths_opt_direct=[]
        interactions = []
        candidates = []
        to_plot_opt = []
        to_plot_direct = []
        self.log['start_seed'] = seed
        total_interactions = 6e6
        count_interactions=0
        episode=0
        
        sampling = self.sample_trajectories if self.independent_samplimg else self.sample_t_opt_and_search_for_t_direct
        while count_interactions < self.max_interactions*num_episodes: #total_interactions:
            self.env.seed(self.seed)
            episode+=1
            
            print('--------- new map {} -------------'.format(episode))
            t_opt, t_direct,final_trajectories,num_interactions = sampling()

            for i in range(1):
                self.policy.train()
                if self.optimization_method == 'direct':
                    policy_loss = self.direct_optimization_loss(t_opt, t_direct)
                elif self.optimization_method == 'CE':
                    policy_loss = self.cross_entropy_loss(final_trajectories, elite_frac = 0.05)
                    
                print(i,policy_loss)
                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()

            interactions.append(num_interactions)

            opt_reward = t_opt.reward
            direct_reward=t_direct.reward
            print ('opt reward: {:.3f}, success: {}, length: {}, priority: {:.3f}, objective: {:.3f}'.format(opt_reward, t_opt.status, len(t_opt.actions),t_opt.node.priority,t_opt.node.objective))
            print ('direct reward: {:.3f}, success: {}, length: {}, priority: {:.3f}, objective: {:.3f}'.format(direct_reward,t_direct.status, len(t_direct.actions),t_direct.node.priority,t_direct.node.objective))
            
            to_plot_opt.append((count_interactions+len(t_opt.actions),opt_reward))
            to_plot_direct.append((count_interactions+num_interactions,direct_reward))
            candidates.append(len(final_trajectories))
            rewards_opt_direct.append((opt_reward,direct_reward))
            lengths_opt_direct.append((len(t_opt.actions),len(t_direct.actions)))
            priority_opt_direct.append((t_opt.node.priority,t_direct.node.priority))
            sys.stdout.flush()
            count_interactions+=num_interactions
            self.seed+=1
            if episode % 20 == 1:
                self.save_checkpoint()
            
        self.save_checkpoint()
        self.log['interactions']=interactions
        self.log['num_candidates'] = candidates
        self.log['rewards_opt_direct'] = rewards_opt_direct
        self.log['lengths_opt_direct'] = lengths_opt_direct
        self.log['priority_opt_direct'] = priority_opt_direct
        
        self.log['to_plot_opt'] =to_plot_opt
        self.log['to_plot_direct'] =to_plot_direct
        
        return rewards_opt_direct
        
    def collect_data_candidates_direct_returns(self,exps=40,alphas=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]):
        sampling = self.sample_trajectories if self.independent_samplimg else self.sample_t_opt_and_search_for_t_direct
        
        returns_of_candidates_alphas,direct_obj_of_candidates_alphas,direct_priority_of_candidates_alphas = {},{},{}
        self.break_on_goal = False
        for alpha in alphas:
            direct_returns_of_all_candidates = []
            direct_obj = []
            direct_priority =[]
            Node.alpha = alpha
            seed = self.seed
            seed += 100000
            print ('-'*100,'\n',alpha,'\n','-'*100)
            for exp in range(exps):
                self.env.seed(seed)
                #print('--------- new map {} -------------'.format(seed))
                t_opt, t_direct,final_trajectories,num_interactions = sampling()
                seed+=1
                direct_returns_of_all_candidates += [t.reward for t in final_trajectories]
                direct_obj += [t.node.objective for t in final_trajectories]
                direct_priority += [t.node.priority for t in final_trajectories]
            
            returns_of_candidates_alphas[alpha] = direct_returns_of_all_candidates
            direct_obj_of_candidates_alphas[alpha] = direct_obj
            direct_priority_of_candidates_alphas[alpha] = direct_priority
        
        self.log['returns_of_candidates']=returns_of_candidates_alphas
        self.log['direct_obj_of_candidates']=direct_obj_of_candidates_alphas
        self.log['direct_priority_of_candidates']=direct_priority_of_candidates_alphas
        self.break_on_goal = True
        
    def play(self,sample_opt = True,seed=1234,inarow=True,auto=True):
        self.seed = seed
        def resetEnv(seed):
            self.env = self.reset()
            self.env.seed(seed)
            t_opt,t_direct,_,_= self.sample_t_opt_and_search_for_t_direct(inference =sample_opt)
            actions = t_opt.actions if sample_opt else t_direct.actions
            return actions
        while True:
            
            self.seed+=1
            actions = resetEnv(self.seed)
            super().play(actions,self.seed,auto=auto)
            if not inarow:
                break
