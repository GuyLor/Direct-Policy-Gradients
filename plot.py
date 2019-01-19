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



def sliding_window_rewards(x,window_size=500):
    return np.convolve(x,np.ones(window_size)/window_size,'valid')

def aligned(x,y):
    if len(x) != len(y):
        print (len(x), len(y))
        m=min(len(x),len(y))
        x = x[:m]
        y=y[:m]
    return x,y

def prepare_plots_ours(ours,truncate=None):
    opt_x,opt_y = zip(*ours['to_plot_opt'])
    direct_x,direct_y = zip(*ours['to_plot_direct'])
    
    opt_y = sliding_window_rewards(opt_y)
    direct_y = sliding_window_rewards(direct_y)
    
    opt_x,opt_y = aligned(opt_x,opt_y)
    direct_x,direct_y = aligned(direct_x,direct_y)
    print(opt_x[-1])
    if truncate is not None:
        opt_x,opt_y = opt_x[:truncate],opt_y[:truncate]
        direct_x,direct_y = direct_x[:truncate],direct_y[:truncate]
    
    return (opt_x,opt_y),(direct_x,direct_y)

def prepare_plots_reinforce(reinforce,truncate=None):
    reinforce_x, reinforce_y = zip(*reinforce['to_plot100'])
    reinforce_y = sliding_window_rewards(reinforce_y)
    reinforce_x, reinforce_y=aligned(reinforce_x, reinforce_y)
    if truncate is not None:
        reinforce_x,reinforce_y = reinforce_x[:truncate],reinforce_y[:truncate]
    return reinforce_x, reinforce_y

ours_1 =torch.load('ours_log1.pkl')
ours_2 =torch.load('ours_log2.pkl')
ours_3 =torch.load('ours_log3.pkl')


log_reinforce = torch.load('reinforce_log.pkl')
#log_reinforce = torch.load('new_baseline_repeat10.pkl')

opt1,direct1=prepare_plots_ours(ours_1)
opt2,direct2=prepare_plots_ours(ours_2)

opt3,direct3=prepare_plots_ours(ours_3)

"""

reinforce = prepare_plots_reinforce(log_reinforce)

to_plot = [opt3,direct3,reinforce]

plot_ours_vs_reinforce(to_plot,
['opt','direct','reinforce'],
'ours_vs_renforce.png')

"""
to_plot=[opt1,direct1,opt2,direct2,opt3,direct3]

plot_ours_vs_ours(to_plot,
['opt_1','direct_1','opt_2','direct_2','opt_3','direct_3'],
'ours_vs_ours.png')



