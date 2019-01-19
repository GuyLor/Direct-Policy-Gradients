from scripts.direct_astar import DirectAstar
from scripts.reinforce import Reinforce
from scripts.reinforce_baseline import ReinforceBaseline
from scripts.policy_model import Checkpoint
import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--env_path', type=str, default='MiniGrid-MultiRoom-N6-v0',
                    help='path of minigrid enviornment, should start with "MiniGrid"')

parser.add_argument('--method', type=str, default='direct_astar',
                    help='rl method- direct_astar, reinforce or reinforce_baseline ')

parser.add_argument('--model_dir', type=str, default='saved_models',
                    help='path of pytorch state dicts for saving and loading')

parser.add_argument('--save_path', type=str, default=None,
                    help='path of pytorch state dicts for saving after training') #policy_state_dicts_multiroom.pkl

parser.add_argument('--load_path', type=str, default=None,
                    help='path of pytorch state dicts for loading before training or playing')

parser.add_argument('--log_path', type=str, default='log.pkl',
                    help='path of log (dictionary)')

parser.add_argument('--train', action='store_true',
                    help='train the model (continue from last checkpoint if load_path exist )')

parser.add_argument('--episodes', type=int, default=3000,
                    help='number of training episodes')

parser.add_argument('--eps_reward', type=float, default=1.0,
                    help='priority = gumbel + eps_reward * reward')

parser.add_argument('--eps_grad', type=float, default=1.0,
                    help='gradient = 1/eps_grad * [grad(t_opt) - grad(t_direct)]')

parser.add_argument('--discount', type=float, default=0.99,
                    help='it is better to leave the discount factor 1 because t_direct is shorter then t_opt')

parser.add_argument('--max_steps', type=float, default=100,
                    help='max trajectory length')

parser.add_argument('--max_interactions', type=float, default=3000,
                    help='max interactions allowed with the enviornment')

parser.add_argument('--max_search_time', type=float, default=30,
                    help='max time in seconds of one t_direct search (priority queue) ')

parser.add_argument('--update_wo_improvement',action='store_true',
                    help='update parameters even if priority(t_direct)<priority(t_opt)')

parser.add_argument('--keep_searching',action='store_true',
                    help='keep searching for t_direct even if priority(t_direct)>priority(t_opt)')

parser.add_argument('--mixed_search_strategy',action='store_true',
                    help='for half interactions search in a BFS manner and the rest in a DFS (if dfs_like is true, the order is changing) ')

parser.add_argument('--dfs_like',action='store_true',
                    help='sample a full trajectory before popping the next node from the queue')

parser.add_argument('--seed', type=int, default=0,
                    help='random seed')

parser.add_argument('--play',action='store_true',
                    help='let the policy-model play games in a row')

args = parser.parse_args()

    
def main():
    cp = Checkpoint(folder_path=args.model_dir,
                    load_path=args.load_path,
                    save_path=args.save_path)
    
    if args.method == 'direct_astar':
        m = DirectAstar(env_path=args.env_path,
                        chekpoint=cp,
                        seed=args.seed,
                        discount=args.discount,
                        eps_grad=args.eps_grad,
                        eps_reward=args.eps_reward,
                        max_steps=args.max_steps,                        
                        max_interactions=args.max_interactions,
                        mixed_search_strategy = args.mixed_search_strategy,
                        dfs_like=args.dfs_like,
                        max_search_time=args.max_search_time,
                        update_wo_improvement=args.update_wo_improvement,
                        keep_searching=args.keep_searching)
    elif args.method == 'reinforce':
        m = Reinforce(env_path=args.env_path,
                        chekpoint=cp,
                        seed=args.seed,
                        discount=args.discount,
                        max_interactions=args.max_interactions,
                        max_steps=args.max_steps)
    elif args.method == 'reinforce_baseline':
        m = ReinforceBaseline(env_path=args.env_path,
                            chekpoint=cp,
                            seed=args.seed,
                            discount=args.discount,
                            max_interactions=args.max_interactions,
                            max_steps=args.max_steps)

    if args.train:
        m.train(num_episodes=args.episodes,seed=args.seed)
        m.log['args'] = args
        torch.save(m.log, args.log_path)
    if args.play:
        m.play(seed = args.seed+9999,inarow=True,auto=True)

if __name__ == '__main__':
    main()
