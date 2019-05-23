import torch
import numpy as np
import matplotlib as matplotlib  # for plotting
import matplotlib.pyplot as plt

def plot_ours_vs_reinforce(plots,labels,filename):
    opt,direct,reinforce=plots
    #opt,direct=plots
    plt.figure(figsize=(12,6))
    plt.plot(opt[0] ,opt[1], 'b.',alpha=.5,markersize=1,label=labels[0])
    plt.plot(direct[0],direct[1], 'g.',alpha=.5,markersize=1,label=labels[1])
    plt.plot(reinforce[0],reinforce[1], 'r.',alpha=.5,markersize=1,label=labels[2])
    
    plt.legend(markerscale=15)

    plt.xlabel('Interactions')
    plt.ylabel('Reward')
    plt.savefig(filename)

def plot_ours_vs_ours(plots,labels,filename):
    plt.figure(figsize=(12,6))
    colors =['b','g','r','k','c','m','y','y']
    def plot_each(p,l):
        opt,direct=p
        print (len(opt),len(direct))
        
        plt.plot(opt[0] ,opt[1], colors[l]+'.', alpha=.5, markersize=1,label=labels[l])
        plt.plot(direct[0],direct[1], colors[l+1]+'.', alpha=.5,markersize=1,label=labels[l+1])
    
    for i in range(0,len(plots),2):
        p=plots[i],plots[i+1]
        plot_each(p,i)
    
    plt.legend(markerscale=15)
    plt.xlabel('Interactions')
    plt.ylabel('Reward')
    plt.savefig(filename)

def plot_ours_vs_cross_entropy(plots,labels,filename):
    plt.figure(figsize=(12,6))
    colors =['b','g','r','k','c','m','y','y']
    def plot_each(p,l):
        plt.plot(p[0] ,p[1], colors[l]+'.', alpha=.5, markersize=1,label=labels[l])
    
    for i,p in enumerate(plots):
        print (i)
        
        plot_each(p,i)
    
    plt.legend(markerscale=15)
    plt.xlabel('Interactions')
    plt.ylabel('Reward')
    #plt.title('independent sampling')
    plt.savefig(filename)


def sliding_window_rewards(x,window_size=1000):
    return np.convolve(x,np.ones(window_size)/window_size,'valid')

def aligned(x,y):
    if len(x) != len(y):
        print (len(x), len(y))
        m=min(len(x),len(y))
        x = x[:m]
        y=y[:m]
    return x,y

def prepare_plots_ours(ours,slide = 600,truncate=None):
    opt_x,opt_y = zip(*ours['to_plot_opt'])
    direct_x,direct_y = zip(*ours['to_plot_direct'])
    
    opt_y = sliding_window_rewards(opt_y,slide)
    direct_y = sliding_window_rewards(direct_y,slide)
    
    opt_x,opt_y = aligned(opt_x,opt_y)
    direct_x,direct_y = aligned(direct_x,direct_y)
    print(opt_x[-1])
    if truncate is not None:
        opt_x,opt_y = opt_x[:truncate],opt_y[:truncate]
        direct_x,direct_y = direct_x[:truncate],direct_y[:truncate]
    
    return (opt_x,opt_y),(direct_x,direct_y)

def prepare_plots_cross_entropy(ours,slide = 500,truncate=None):
    opt_x,opt_y = zip(*ours['to_plot_opt'])
    
    opt_y = sliding_window_rewards(opt_y,slide)
    
    opt_x,opt_y = aligned(opt_x,opt_y)
    print(opt_x[-1])
    if truncate is not None:
        opt_x,opt_y = opt_x[:truncate],opt_y[:truncate]
    return (opt_x,opt_y)

def prepare_plots_reinforce(reinforce,truncate=None):
    reinforce_x, reinforce_y = zip(*reinforce['to_plot'])
    reinforce_y = sliding_window_rewards(reinforce_y,30)
    reinforce_y = sliding_window_rewards(reinforce_y,3000)
    reinforce_x, reinforce_y=aligned(reinforce_x, reinforce_y)
    if truncate is not None:
        reinforce_x,reinforce_y = reinforce_x[:truncate],reinforce_y[:truncate]
    return reinforce_x, reinforce_y

"""
ours_1 =torch.load('../logs/priority4.pkl')
ours_12 =torch.load('../logs/priority4_eps1000_objective.pkl')
ours_2 =torch.load('../logs/cross_entropy_priority4_elite02.pkl')
ours_3 =torch.load('../logs/cross_entropy_priority4_elite005.pkl')
ours_4 =torch.load('../logs/cross_entropy_priority4_elite0001.pkl')

opt1,direct1=prepare_plots_ours(ours_1)
opt12,direct12=prepare_plots_ours(ours_12,slide = 300)
opt2=prepare_plots_cross_entropy(ours_2,slide = 300)
opt3=prepare_plots_cross_entropy(ours_3,slide = 300)
opt4=prepare_plots_cross_entropy(ours_4)
"""

ours_1 =torch.load('../logs/direct_search_sampling_constant_sampling.pkl')
ours_12 =torch.load('../logs/direct_search_sampling_per_action.pkl')
ours_13 =torch.load('../logs/direct_search_sampling_eps_05.pkl')
ours_14 =torch.load('../logs/direct_search_sampling_eps_annealing2.pkl')
ours_15 =torch.load('../logs/direct_search_sampling_direct_prefix.pkl')
ours_152 =torch.load('../logs/direct_search_sampling_priority2_mixed.pkl')
ours_16 =torch.load('../logs/direct_search_sampling_uniform_sampling.pkl')
ours_17 =torch.load('../logs/direct_search_sampling_geometric_sampling.pkl')

ours_2 =torch.load('../logs/direct_independent_sampling.pkl')
#ours_21 =torch.load('../logs/direct_independent_sampling_shared.pkl')

ours_3 =torch.load('../logs/ce_search_sampling_top_5.pkl')
ours_4 =torch.load('../logs/ce_independent_sampling_top_5.pkl')
#ours_41 =torch.load('../logs/ce_independent_sampling_shared.pkl')

#ours_5 =torch.load('../logs/direct_ce_search_sampling.pkl')
#ours_6 =torch.load('../logs/direct_ce_independent_sampling.pkl')
#ours_61 =torch.load('../logs/direct_ce_independent_sampling_shared.pkl')

opt1,direct1=prepare_plots_ours(ours_1)
opt12,direct12=prepare_plots_ours(ours_12)
opt13,direct13=prepare_plots_ours(ours_13)
opt14,direct14=prepare_plots_ours(ours_14)
opt15,direct15=prepare_plots_ours(ours_15)
opt152,direct152=prepare_plots_ours(ours_152)

opt16,direct16=prepare_plots_ours(ours_16)
opt17,direct17=prepare_plots_ours(ours_17)

opt2,direct2=prepare_plots_ours(ours_2)

opt3,direct3=prepare_plots_ours(ours_3)
opt4,direct4=prepare_plots_ours(ours_4)

#opt5,direct5=prepare_plots_ours(ours_5)
#opt6,direct6=prepare_plots_ours(ours_6)


reinforce_actions = torch.load('../logs/reinforce_action_level.pkl')
reinforce_traj = torch.load('../logs/reinforce_traj_level.pkl')



reinforce_actions = prepare_plots_reinforce(reinforce_actions)
reinforce_traj = prepare_plots_reinforce(reinforce_traj)
"""
to_plot = [opt1,opt16,opt17]#,reinforce_actions,reinforce_traj]

plot_ours_vs_cross_entropy(to_plot,
['constant length - 100','uniform sampling','geometric sampling'],
'../logs/uniform_vs_geomtric_vs_constant_direct_search_2.png')
"""
to_plot = [opt1,opt15,opt152,reinforce_actions,reinforce_traj]

plot_ours_vs_cross_entropy(to_plot,
['t_opt','t_opt prefix search (priority2)','t_opt prefix search complete from policy (priority2)','reinforce action level','reinforce trajectory level'],
'../logs/reinforce_actions_vs_traj_vs_ours_prefix_complete_from_policy.png')

"""
to_plot=[opt1,opt2,opt3,opt4,reinforce_traj,reinforce_actions]

plot_ours_vs_cross_entropy(to_plot,
['direct search','direct independent','cross entropy search - top5','cross entropy independent - top5','reinforce trajectory level','reinforce action level'],
'../logs/direct_vs_top5_ce_vs_reinforce.png')

to_plot=[opt1,opt13,opt14,opt12]

plot_ours_vs_cross_entropy(to_plot,
['global epsilon 1.0','global epsilon 0.5','global annealing epsilon 2.0-0.5','per action epsilon'],
'../logs/direct_search_global1.0_vs0.5_vs2-05_vs_per_action_epsilon.png')
"""

