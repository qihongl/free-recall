import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import numpy as np
import torch
import os

from collections import Counter
from tasks import FreeRecall
# from models import CRPLSTM as Agent
from models import GRU as Agent
from models import compute_a2c_loss, compute_returns
from utils import to_sqpth, to_pth, to_np, to_sqnp, make_log_fig_dir, rm_dup
from stats import compute_stats, compute_recall_order, lag2index
from vis import plot_learning_curve

sns.set(style='white', palette='colorblind', context='talk')
subj_id = 0
np.random.seed(subj_id)
torch.manual_seed(subj_id)

# init task
n = 18
n_std = 6
reward = 1
penalty = -1
penalize_repeat = True
task = FreeRecall(
    n_std=n_std, n=n, reward=reward, penalty=penalty,
    penalize_repeat=penalize_repeat
)
# init model
lr = 1e-3
dim_hidden = 512
dim_input = task.x_dim
dim_output = task.x_dim

# make log dirs
epoch_trained = 90000
exp_name = f'n-{n}-n_std-{n_std}/h-{dim_hidden}/sub-{subj_id}'
log_path, fig_path = make_log_fig_dir(exp_name, makedirs=False)

# reload the weights
fname = f'wts-{epoch_trained}.pth'
agent = Agent(dim_input, dim_hidden, dim_output)
agent.load_state_dict(torch.load(os.path.join(log_path, fname)))

# testing
n_test = 2000
log_r = np.zeros((n_test, n_std))
log_a = np.zeros((n_test, n_std))
log_std_items = np.zeros((n_test, n_std))
for i in range(n_test):
    # re-sample studied items
    X = task.sample(to_pytorch=True)
    log_std_items[i] = task.studied_item_ids
    # study phase
    hc_t = agent.get_zero_states()
    for t, x_t in enumerate(X):
        [_, _, _, hc_t], _ = agent.forward(x_t, hc_t)

    # recall phase
    empty_input = torch.zeros(task.x_dim)
    for t in range(n_std):
        [a_t, pi_a_t, v_t, hc_t], _ = agent.forward(empty_input, hc_t)
        r_t = task.get_reward(a_t)
        # log
        log_r[i, t] = r_t
        log_a[i, t] = a_t


'''figures'''

# mask = np.logical_not(np.mean(log_r, axis=1) == 1)
# n_test = np.sum(mask)
# select the last n_test trials to analyze
targ = log_std_items
resp = log_a


# compute the recall order given target and responses
order = compute_recall_order(targ, resp)

# compute the tally for actual and possible responses
tally = np.zeros((n_test, (n_std - 1)* 2))
tally_poss = np.zeros((n_test, (n_std - 1)* 2))
lags = []
for i in range(n_test):
    order_i = order[i]
    # order_i_rmnan = order_i[~np.isnan(order_i)]
    order_i_rmnan = order_i[~np.isnan(order_i)]
    order_i_rmnan = rm_dup(order_i_rmnan)
    for j in range(len(order_i_rmnan) - 1):
        lag = int(order_i_rmnan[j+1] -  order_i_rmnan[j])
        if lag != 0:
            lag_index = lag2index(lag, n_std)
            # increment the count of the corresponding lag index
            tally[i, lag_index] +=1
            lags.append(lag)

    for j in range(len(order_i_rmnan) - 1):
        for o in range(n_std):
            lag_poss = int(o - order_i_rmnan[j])
            if lag_poss != 0:
                lag_index = lag2index(lag_poss, n_std)
                tally_poss[i, lag_index] +=1

assert np.all(tally_poss >= tally), 'possible count must >= actual count'
# compute conditional response probability
crp = np.divide(tally, tally_poss, out=np.zeros_like(tally), where=tally_poss!=0)
# crp = tally / tally.sum(axis=1)[:,None]

p_mu, p_se = compute_stats(crp, axis=0)
xticklabels = np.concatenate((np.arange(-(n_std - 1), 0), np.arange(1, n_std)))

f, ax = plt.subplots(1,1, figsize=(6,4))
ax.errorbar(x=np.arange(n_std - 1), y=p_mu[:n_std - 1], yerr = p_se[:n_std - 1],color='k')
ax.errorbar(x=np.arange(n_std - 1) + (n_std - 1), y=p_mu[-(n_std - 1):], yerr = p_se[-(n_std - 1):], color='k')
ax.set_ylim([0, None])
ax.set_xticks(np.arange((n_std - 1)*2))
ax.set_xticklabels(xticklabels)
ax.set_xlabel('Lag')
ax.set_ylabel('p')
ax.set_title('Conditional response probability')
sns.despine()

# order = order[:,0]
# plot the serial position curve
unique_recalls = np.concatenate([np.unique(order[i]) for i in range(n_test)])
counter = Counter(unique_recalls)
recalls = np.array([counter[i] for i in range(n_std)])

spc_x = range(n_std)
spc_y = recalls / n_test
f, ax = plt.subplots(1,1, figsize=(6, 4))
ax.plot(spc_y)
# ax.set_ylim([None, 1])
ax.set_xlabel('Position')
ax.set_ylabel('p')
ax.set_title('Recall probability')
sns.despine()



n_items_recalled = np.empty(n_test, )
for i in range(n_test):
    order_i = np.unique(order[i])
    n_items_recalled[i] = len(order_i[~np.isnan(order_i)])
prop_item_recalled = n_items_recalled / n_std

pir_mu, pir_se = compute_stats(prop_item_recalled)
print(f'%% items recalled = %.2f, se = %.2f' % (pir_mu, pir_se))

mean_r_all_trials = np.mean(log_r, axis=1)
r_mu, r_se = compute_stats(mean_r_all_trials)
print(f'Average reward = %.2f, se = %.2f' % (r_mu, r_se))


p_lure = np.sum(np.isnan(order)) / len(order.reshape(-1))
print(f'Probability of lure recall is {p_lure}')

counter = Counter(order[:,0])
recalls_1st = np.array([counter[i] for i in range(n_std)])

f, ax = plt.subplots(1,1, figsize=(6, 4))
ax.plot(recalls_1st / np.sum(recalls_1st))
# ax.set_ylim([None, 1])
ax.set_xlabel('Position')
ax.set_ylabel('p')
ax.set_title('Recall probability for the 1st item')
sns.despine()
