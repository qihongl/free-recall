'''dup from  train-fr, add the stop unit'''
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import os

from tasks import FreeRecall
from models import GRU as Agent
from models import compute_a2c_loss, compute_returns
from utils import to_sqpth, to_pth, to_np, to_sqnp, make_log_fig_dir, estimated_run_time, int2onehot
from vis import plot_learning_curve
from stats import compute_stats

sns.set(style='white', palette='colorblind', context='talk')

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='free-recall-stp', type=str)
parser.add_argument('--subj_id', default=0, type=int)
parser.add_argument('--n', default=40, type=int)
parser.add_argument('--n_std', default=8, type=int)
parser.add_argument('--dim_hidden', default=256, type=int)
parser.add_argument('--lr', default=3e-3, type=float)
parser.add_argument('--n_epochs', default=100001, type=int)
parser.add_argument('--reward', default=1, type=int)
parser.add_argument('--penalty', default=-.2, type=float)
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
dim_input = task.x_dim * 2 + 1
dim_output = task.x_dim + 1
agent = Agent(dim_input, p.dim_hidden, dim_output)
optimizer = torch.optim.Adam(agent.parameters(), lr=p.lr)

'''start training'''
len_test_phase = p.n_std + int(p.n_std * .5)
log_r = [0 for i in range(p.n_epochs)]
# np.full((p.n_epochs, len_test_phase), np.nan)
log_a = np.full((p.n_epochs, len_test_phase), np.nan)
log_std_items = np.full((p.n_epochs, p.n_std), np.nan)
log_loss_actor = np.full((p.n_epochs, ), np.nan)
log_loss_critic = np.full((p.n_epochs, ), np.nan)
for i in range(p.n_epochs):
    time_s = time.time()
    # re-sample studied items
    X = task.sample(to_pytorch=True)
    log_std_items[i] = task.studied_item_ids
    # reset init state
    hc_t = agent.get_zero_states()
    # study phase
    for t, x_t_std in enumerate(X):
        # x_t = torch.cat([x_t_std, torch.zeros(task.x_dim)])
        x_t = torch.cat([x_t_std, torch.zeros(task.x_dim), torch.zeros(1)])
        [_, _, _, hc_t], _ = agent.forward(x_t, hc_t)

    # recall phase
    log_r_i = []
    probs, rewards, values = [], [], []
    x_t_tst = torch.zeros(task.x_dim)
    r_t = torch.zeros(1)
    for t in range(len_test_phase):
        # feed _, prev_recall_item, prev_reward
        x_t = torch.cat([torch.zeros(task.x_dim), x_t_tst, r_t.view(1)])
        [a_t, pi_a_t, v_t, hc_t], _ = agent.forward(x_t, hc_t)
        # compute reward
        r_t = task.get_reward(a_t)
        # compute loss
        rewards.append(r_t)
        values.append(v_t)
        probs.append(pi_a_t)
        # log
        log_r_i.append(r_t)
        log_a[i, t] = a_t
        # if the model choose the stop, break
        if int(to_np(a_t)) == p.n:
            break
        # make a onehot that represent the recalled item at time t
        x_t_tst = int2onehot(a_t, p.n)

    if t > 0:
        returns = compute_returns(rewards, normalize=False)
        loss_actor, loss_critic = compute_a2c_loss(probs, values, returns)
        loss = loss_actor + loss_critic
        log_r[i] = log_r_i
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
    # compute ert
    time_took = time.time() - time_s
    if i == 0:
        ert = estimated_run_time(time_took, p.n_epochs)
        print('Estimated run time = %.2f hours' % (ert))

    pct_rcld = len(task.recalled_item_id) / p.n_std
    if i % 100 == 0:
        print('%3d | r = %.4f, %% item recalled: %.2f loss a = %.4f, c = %.4f, time = %.2f sec' % (
        i, np.mean(log_r[i]), pct_rcld, log_loss_actor[i], log_loss_critic[i], time_took))

'''figures'''
# plot the learning curves
w_size = 20
f, ax = plot_learning_curve(np.mean(log_r,axis=1), window_size=w_size, ylabel='avg reward')
f.savefig(os.path.join(fig_path, 'r.png'))
f, ax = plot_learning_curve(log_loss_actor, window_size=w_size, ylabel='loss actor')
f.savefig(os.path.join(fig_path, 'loss-a.png'))
f, ax = plot_learning_curve(log_loss_critic, window_size=w_size, ylabel='loss critic')
f.savefig(os.path.join(fig_path, 'loss-c.png'))
