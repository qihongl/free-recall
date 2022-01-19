import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import numpy as np
# import itertools
import torch
import os

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
n_std = 4
n = 10
reward = 1
penalty = -.5
tmax = n // 2
task = FreeRecall(n_std=n_std, n=n, reward=reward, penalty=penalty)

# init model
dim_input = task.x_dim
dim_output = task.x_dim
lr = 1e-3
dim_hidden = 128

# for dim_hidden in [2 ** k for k in np.arange(4, 10)]:

# make log dirs
exp_name = f'n-{n}-n_std-{n_std}/-h-{dim_hidden}'
log_path, fig_path = make_log_fig_dir(exp_name)

# testing
agent = CRPLSTM(dim_input, dim_hidden, dim_output, 0, use_ctx=False)
optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

n_epochs = 10001
log_r = np.zeros((n_epochs, tmax))
log_a = np.zeros((n_epochs, tmax))
log_std_items = np.zeros((n_epochs, n_std))
log_loss_actor = np.zeros((n_epochs, ))
log_loss_critic = np.zeros((n_epochs, ))
for i in range(n_epochs):

    # re-sample studied items
    X = task.sample(to_pytorch=True)
    log_std_items[i] = task.studied_item_ids
    # task.recalled_item_id
    # reset init state
    h_0 = torch.zeros(1, 1, dim_hidden)
    c_0 = torch.zeros(1, 1, dim_hidden)
    h_t, c_t = h_0, c_0
    # prealloc
    probs, rewards, values = [], [], []

    # study phase
    for t, x_t in enumerate(X):
        [_, _, _, h_t, c_t], _ = agent.forward(x_t.view(1, 1, -1), h_t, c_t)

    # recall phase
    empty_input = torch.zeros(task.x_dim).view(1, 1, -1)
    for t in range(tmax):
        [a_t, pi_a_t, v_t, h_t, c_t], _ = agent.forward(empty_input, h_t, c_t)
        r_t = task.get_reward(a_t)
        # compute loss
        rewards.append(r_t)
        values.append(v_t)
        probs.append(pi_a_t)
        # log
        log_r[i, t] = r_t
        log_a[i, t] = a_t

    returns = compute_returns(rewards, normalize=False)
    loss_actor, loss_critic = compute_a2c_loss(probs, values, returns)
    loss = loss_actor + loss_critic
    log_loss_actor[i] = to_np(loss_actor)
    log_loss_critic[i] = to_np(loss_critic)

    # update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print('%3d | r = %.4f' % (i, np.mean(log_r[i])))
    if i % 1000 == 0:
        '''save weights'''
        fname = f'wts-{i}.pth'
        torch.save(agent.state_dict(), os.path.join(log_path, fname))

'''figures'''

# plot the learning curves
w_size = 20
f, ax = plot_learning_curve(np.mean(log_r,axis=1), window_size=w_size, ylabel='avg reward')
f.savefig(os.path.join(fig_path, 'r.png'))
f, ax = plot_learning_curve(log_loss_actor, window_size=w_size, ylabel='loss actor')
f, ax = plot_learning_curve(log_loss_critic, window_size=w_size, ylabel='loss critic')
