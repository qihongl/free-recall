import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import os

from tasks import FreeRecall
from models import CRPLSTM, A2C_linear
from models import compute_a2c_loss, compute_returns
from utils import to_sqpth, to_pth, to_np, to_sqnp, make_log_fig_dir
from vis import plot_learning_curve
from stats import compute_stats

sns.set(style='white', palette='colorblind', context='talk')

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='free-recall', type=str)
parser.add_argument('--subj_id', default=0, type=int)
parser.add_argument('--n', default=21, type=int)
parser.add_argument('--n_std', default=6, type=int)
parser.add_argument('--dim_hidden', default=512, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--n_epochs', default=200001, type=int)
parser.add_argument('--reward', default=1, type=int)
parser.add_argument('--penalty', default=-.5, type=float)
parser.add_argument('--penalize_repeat', default=1, type=int)
p = parser.parse_args()
np.random.seed(p.subj_id)
torch.manual_seed(p.subj_id)

# make log dirs
exp_name = f'n-{p.n}-n_std-{p.n_std}/h-{p.dim_hidden}/sub-{p.subj_id}'
log_path, fig_path = make_log_fig_dir(exp_name)

# init task
task = FreeRecall(
    n_std=p.n_std, n=p.n, reward=p.reward, penalty=p.penalty,
    penalize_repeat=bool(p.penalize_repeat)
)

# init model
dim_input = task.x_dim
dim_output = task.x_dim
agent = CRPLSTM(dim_input, p.dim_hidden, dim_output)
optimizer = torch.optim.Adam(agent.parameters(), lr=p.lr)

'''start training'''
log_r = np.zeros((p.n_epochs, p.n_std))
log_a = np.zeros((p.n_epochs, p.n_std))
log_std_items = np.zeros((p.n_epochs, p.n_std))
log_loss_actor = np.zeros((p.n_epochs, ))
log_loss_critic = np.zeros((p.n_epochs, ))
for i in range(p.n_epochs):
    time_s = time.time()
    # re-sample studied items
    X = task.sample(to_pytorch=True)
    log_std_items[i] = task.studied_item_ids
    # reset init state
    h_t, c_t = agent.get_zero_states()
    # prealloc
    probs, rewards, values = [], [], []
    # study phase
    for t, x_t in enumerate(X):
        [_, _, _, h_t, c_t], _ = agent.forward(x_t.view(1, 1, -1), h_t, c_t)

    # recall phase
    empty_input = torch.zeros(task.x_dim).view(1, 1, -1)
    for t in range(p.n_std):
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

    # save weights
    if i % 10000 == 0:
        '''save weights'''
        fname = f'wts-{i}.pth'
        torch.save(agent.state_dict(), os.path.join(log_path, fname))
    time_took = time.time() - time_s
    if i % 1000 == 0:
        print('%3d | r = %.4f, loss-a = %.4f, loss-c = %.4f, time = %.2f sec' % (
        i, np.mean(log_r[i]), log_loss_actor[i], log_loss_critic[i], time_took))

'''figures'''
# plot the learning curves
w_size = 20
f, ax = plot_learning_curve(np.mean(log_r,axis=1), window_size=w_size, ylabel='avg reward')
f.savefig(os.path.join(fig_path, 'r.png'))
f, ax = plot_learning_curve(log_loss_actor, window_size=w_size, ylabel='loss actor')
f.savefig(os.path.join(fig_path, 'loss-a.png'))
f, ax = plot_learning_curve(log_loss_critic, window_size=w_size, ylabel='loss critic')
f.savefig(os.path.join(fig_path, 'loss-c.png'))
