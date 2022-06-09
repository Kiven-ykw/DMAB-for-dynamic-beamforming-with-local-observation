""" visualize the simulation results """

import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


def data_visualization():
    window = 500

    filename = 'ee/DRQN_commonR_PL=5_TS=30000.json'
    r = []
    with open(filename, 'r') as f:
        data = np.array(json.load(f))
        for i in range(len(data) - window + 1):
            r.append(np.mean(data[i:i + window]))
    r = np.array(r)
    #sio.savemat('ee/dqn_commonPL10ts.mat', {'dqn_commonPL10ts': r})
    plt.plot(r, label='DMAB-cooperative', marker="v", markevery=3000, markersize=6, markerfacecolor='none', linewidth=1)

    filename = 'ee/DRQN_competitiveR_PL=5_TS=30000.json'
    r = []
    with open(filename, 'r') as f:
        data = np.array(json.load(f))
        for i in range(len(data) - window + 1):
            r.append(np.mean(data[i:i + window]))
    r = np.array(r)
    # sio.savemat('ee/drqn_competitivePL10ts.mat', {'drqn_competitivePL10ts': r})
    plt.plot(r, label='DMAB-competitive', marker="o", markevery=3000, markersize=6,
             markerfacecolor='none', linewidth=1)

    filename = 'ee/DQN_commonR_PL=5_TS=30000.json'
    r = []
    with open(filename, 'r') as f:
        data = np.array(json.load(f))
        for i in range(len(data) - window + 1):
            r.append(np.mean(data[i:i + window]))
    r = np.array(r)
    #sio.savemat('ee/dqn_commonPL10ts.mat', {'dqn_commonPL10ts': r})
    plt.plot(r, label='DQN-cooperative', marker="x", markevery=3000, markersize=6, markerfacecolor='none', linewidth=1)

    filename = 'ee/DQN_competitiveR_PL=5_TS=30000.json'
    r = []
    with open(filename, 'r') as f:
        data = np.array(json.load(f))
        for i in range(len(data) - window + 1):
            r.append(np.mean(data[i:i + window]))
    r = np.array(r)
    #sio.savemat('ee/dqn_competitivePL10ts.mat', {'dqn_competitivePL10ts': r})
    plt.plot(r, label='DQN-competitive', marker="+", markevery=3000, markersize=8, markerfacecolor='none', linewidth=1)

    filename = 'ee/random_performance.json'
    r = []
    with open(filename, 'r') as f:
        data = np.array(json.load(f))
        for i in range(len(data) - window + 1):
            r.append(np.mean(data[i:i + window]))
    r = np.array(r)
    #sio.savemat('ee/random.mat', {'random': r})
    plt.plot(r, label='Random', marker="p", markevery=3000, markersize=8, markerfacecolor='none', linewidth=1)

    filename = 'ee/greedy_performance.json'
    r = []
    with open(filename, 'r') as f:
        data = np.array(json.load(f))
        for i in range(len(data) - window + 1):
            r.append(np.mean(data[i:i + window]))
    r = np.array(r)
    #sio.savemat('ee/greedy.mat', {'greedy': r})
    plt.plot(r, label='Greedy', marker="d", markevery=3000, markersize=6, markerfacecolor='none', linewidth=1)

    plt.xlabel('The number of time slots')
    plt.ylabel('EE (bps/Hz/J)')
    y_tick = np.arange(150, 376, 25)
    plt.yticks(y_tick, fontsize=10)
    plt.rcParams.update({'font.size': 10})
    plt.xticks(np.arange(0, 30001, 5000), fontsize=10)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()


data_visualization()
