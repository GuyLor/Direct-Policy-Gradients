import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
from matplotlib import colors as mcolors

matplotlib.rc('lines', linewidth=3)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('axes', labelsize=24)


def plot_hist_candidates_direct(logs, alphas_idx=None, filename='new_hist.png'):
    all_alphas = np.array([x / 10 for x in range(0, 11)])  # np.linspace(0,1,11)
    alphas = all_alphas[alphas_idx]
    print("alphas: ", alphas)
    str_alph = ['_{:.1f}'.format(i) for i in alphas]
    name = ''
    for s in str_alph:
        name += s

    type = 'returns'
    #filename = filename + '/alphas_{}'.format(type) + name + '.png'

    # colors =np.array(['b','g','r','k','c','m','sandybrown','salmon','powderblue','fuchsia','y'])
    colors = np.array(['k', 'k', 'k', 'k', 'k', 'k', 'k', 'sienna', 'm', 'k', 'k'])
    # lg = np.array(['alpha = 0.0','alpha = 0.1','alpha = 0.2','alpha = 0.3','alpha = 0.4','alpha = 0.5 ','alpha = 1.0'])
    plt.figure(figsize=(10, 8))
    lg = np.array(['alpha = {:.1f}'.format(i) for i in all_alphas])
    # alp =np.array([.8,.8,.67,.54,.41,.28,.15,.07,.8,.8,.8])
    alp = np.array([.8, .67, .54, .41, .28, .17, .15, .08, .8, .8, .1])
    colors_dict = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    colors = [colors_dict[c] for c in colors]
    color_a = []
    for c, a in zip(colors, alp):
        # print (c,a)
        color_a.append(mcolors.to_rgba(c, a))
        """
        c=np.array(c)
        c[:3]=a
        print (c,a)

        color_a.append(c)
        """
    colors = np.array(color_a)
    print(colors)
    to_plot = []
    weights = []
    seeds = len(logs)
    bins = 9
    for idx, alp in enumerate(all_alphas):
        p = []
        w = []
        for i, log in enumerate(logs):
            if type == 'returns':
                candi = log['returns_of_candidates']
            elif type == 'objective':
                candi = log['direct_obj_of_candidates']
            else:
                candi = log['direct_priority_of_candidates']

            p += candi[alp]
            # to_plot.append(candi[alp])
            w += [1.0 / (seeds)] * len(candi[alp])

            # w = [1.0/(seeds*20)]*len(candi[alp])
            # weights.append(w)
            # print(alp,i)
            # print (np.histogram(candi[alp], bins=bins))
        to_plot.append(p)
        weights.append(w)
    print(len(to_plot))
    to_plot = np.array(to_plot)
    weights = np.array(weights)
    plt.hist(to_plot[alphas_idx], histtype='bar', log=True, bins=bins, weights=weights[alphas_idx],
             color=colors[alphas_idx], label=lg[alphas_idx])

    # plt.hist(to_plot[alphas_idx], histtype='bar', alpha=alp[alphas_idx],log=True, bins=bins,weights=weights[alphas_idx],color='k',label=lg[alphas_idx])
    plt.xlabel('direct {}'.format(type))  # ,fontsize=12)
    plt.ylabel('number of candidates')  # ,fontsize=12)
    plt.legend(markerscale=15, fontsize=18)
    plt.savefig(filename)
"""

path = os.path.join('train_with_0.2')  # ,'400 episodes')
alpha_upper_bound = [torch.load(path + '/400 episodes prev UB/seed_{}.pkl'.format(i)) for i in range(1, 6)]

alphas = [0, 1, 2, 3, 4, 5, -1]
plot_hist_candidates_direct(alpha_upper_bound, alphas, filename=path)
"""


def plot_ours_vs_ours(plots, labels, color=None, plot_direct=None, filename='new_def_plot.png'):
    """ direct = None: opt & direct, direct = True: only direct, direct = False: only opt """
    # plt.figure(figsize=(12,6))
    plt.figure(figsize=(10, 8))
    colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', 'y']

    def plot_each(p, l, c):
        opt, direct = p
        lo, ld = l
        if plot_direct is None:
            plt.plot(opt[0], opt[1], c + '-', alpha=.8, linewidth=3, label=lo)
            plt.plot(direct[0], direct[1], c + '--', alpha=.8, linewidth=3, label=ld)
        elif plot_direct:
            plt.plot(direct[0], direct[1], c + '--', alpha=.8, linewidth=3, label=ld)
        else:
            plt.plot(opt[0], opt[1], c + '-', alpha=.8, linewidth=3, label=lo)

    for ind, i in enumerate(range(0, len(plots), 2)):
        p = plots[i], plots[i + 1]
        # print (i,labels[i])
        if plot_direct is None:
            l = labels[ind] + ' (opt)', labels[ind] + ' (direct)'
        else:
            l = labels[ind], labels[ind]

        if color is None:
            c = colors[ind]
        else:
            c = color
        plot_each(p, l, c)

    plt.legend(markerscale=15, fontsize=18)  # markerscale=15
    # plt.title('max gumbel + return + alpha*upper bound')
    plt.xlabel('interactions')  # ,fontsize=12
    plt.ylabel('return')  # ,fontsize=12
    plt.savefig(filename)
"""
to_plot = [opt3,direct3]
lg = ['alpha = 0.2']
plot_ours_vs_ours(to_plot,
lg,
color='r',
plot_direct=None,
filename='plots/avg_alpha02_direct&opt.png')

"""