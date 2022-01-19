import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import numpy as np
import torch
import os

from collections import Counter
from tasks import FreeRecall
from models import CRPLSTM, A2C_linear
from models import compute_a2c_loss, compute_returns
from utils import to_sqpth, to_pth, to_np, to_sqnp, make_log_fig_dir
from vis import plot_learning_curve
from stats import compute_stats

sns.set(style='white', palette='colorblind', context='talk')
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
log_root = '../log'
fig_root = '../figs'

# init task
n = 8
reward = 1
penalty = -.5
tmax = n // 2
task = FreeRecall(n, reward=reward, penalty=penalty)

# init model
lr = 1e-3
dim_input = task.x_dim
dim_output = task.x_dim
dim_hidden = 512

# for dim_hidden in [2 ** k for k in np.arange(4, 10)]:
print(dim_hidden)

# make log dirs
epoch_trained = 100000
exp_name = f'n-{n}-h-{dim_hidden}'
log_path, fig_path = make_log_fig_dir(exp_name)

# reload the weights
fname = f'wts-{epoch_trained}.pth'
agent = CRPLSTM(dim_input, dim_hidden, dim_output, 0, use_ctx=False)
agent.load_state_dict(torch.load(os.path.join(log_path, fname)))

# testing
n_test = 1000
log_r = np.zeros((n_test, tmax))
log_a = np.zeros((n_test, tmax))
log_std_items = np.zeros((n_test, task.n_std))
for i in range(n_test):
    # re-sample studied items
    X = task.sample(to_pytorch=True)
    log_std_items[i] = task.studied_item_ids

    # reset init state
    h_0 = torch.zeros(1, 1, dim_hidden)
    c_0 = torch.zeros(1, 1, dim_hidden)
    h_t, c_t = h_0, c_0

    # study phase
    for t, x_t in enumerate(X):
        [_, _, _, h_t, c_t], _ = agent.forward(x_t.view(1, 1, -1), h_t, c_t)

    # recall phase
    empty_input = torch.zeros(task.x_dim).view(1, 1, -1)
    for t in range(tmax):
        [a_t, pi_a_t, v_t, h_t, c_t], _ = agent.forward(empty_input, h_t, c_t)
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
order = np.full((n_test, tmax), np.nan)
# loop over all trials
for i in range(n_test):
    # for each trial, loop over time
    for j in range(tmax):
        # if recall a targ
        if resp[i][j] in targ[i]:
            # figure out the true order
            order_resp_j = np.where(targ[i] == resp[i][j])[0]
            order[i, j] = int(order_resp_j)

# from collections import Counter
# Counter(order.reshape(-1))

def lag2index(lag, n_std_items):
    '''map lag to lag_index
    e.g.
    if n stud items is 4, then max lag is 3 (item 1 -> item 4),
    so all lags are -3, -2, -1, +1, +2, +3
    and lag_index are 0, 1, 2, 3, 4, 5
    '''
    if lag == 0:
        return None
    if lag > 0:
        lag_index = lag + n_std_items - 1
    else:
        lag_index = lag + n_std_items
    return lag_index - 1

# compute the tally for actual and possible responses
tally = np.zeros((n_test, (task.n_std - 1)* 2))
tally_poss = np.zeros((n_test, (task.n_std - 1)* 2))
lags = []
for i in range(n_test):
    order_i = order[i]
    order_i_rmnan = order_i[~np.isnan(order_i)]
    for j in range(len(order_i_rmnan) - 1):
        lag = int(order_i_rmnan[j+1] -  order_i_rmnan[j])
        if lag != 0:
            lag_index = lag2index(lag, task.n_std)
            # increment the count of the corresponding lag index
            tally[i, lag_index] +=1
            lags.append(lag)

    for j in range(len(order_i_rmnan) - 1):
        for o in range(task.n_std):
            lag_poss = int(o - order_i_rmnan[j])
            if lag_poss != 0:
                lag_index = lag2index(lag_poss, task.n_std)
                tally_poss[i, lag_index] +=1

assert np.all(tally_poss >= tally), 'possible count must >= actual count'
# compute conditional response probability
crp = np.divide(tally, tally_poss, out=np.zeros_like(tally), where=tally_poss!=0)
# crp = tally / tally.sum(axis=1)[:,None]

# for n_std_items in [4]:
# # n_std_items = 6
#     temp = [- i-1 for i in np.arange(n_std_items-1)][::-1] + [i+1 for i in range(n_std_items-1)]
#     print([lag2index(i, n_std_items) for i in temp])
#     print(lag2index(0, n_std_items))


p_mu, p_se = compute_stats(crp, axis=0)
xticklabels = np.concatenate((np.arange(-(task.n_std - 1), 0), np.arange(1, task.n_std)))

f, ax = plt.subplots(1,1, figsize=(6,4))
ax.errorbar(x=np.arange(task.n_std - 1), y=p_mu[:task.n_std - 1], yerr = p_se[:task.n_std - 1],color='k')
ax.errorbar(x=np.arange(task.n_std - 1) + (task.n_std - 1), y=p_mu[-(task.n_std - 1):], yerr = p_se[-(task.n_std - 1):], color='k')
ax.set_ylim([0, None])
ax.set_xticks(np.arange((task.n_std - 1)*2))
ax.set_xticklabels(xticklabels)
ax.set_xlabel('Lag')
ax.set_ylabel('p')
ax.set_title('Conditional response probability')
sns.despine()


# from collections import Counter
# counter = Counter(lags)
# sorted(counter.items())
