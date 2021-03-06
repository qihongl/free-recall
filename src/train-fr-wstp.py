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

# plotting param
sns.set(style='white', palette='colorblind', context='talk')
plt_win_size = 20

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='free-recall-stp', type=str)
parser.add_argument('--subj_id', default=0, type=int)
parser.add_argument('--n', default=40, type=int)
parser.add_argument('--n_std', default=10, type=int)
parser.add_argument('--v', default=3, type=int)
parser.add_argument('--dim_hidden', default=128, type=int)
# parser.add_argument('--noise_level', default=.02, type=int)
parser.add_argument('--noise_level', default=0, type=int)
parser.add_argument('--lr', default=3e-3, type=float)
parser.add_argument('--beta', default=1, type=float)
parser.add_argument('--beta_decay', default=1, type=float)
parser.add_argument('--beta_decay_freq', default=None, type=float)
parser.add_argument('--n_epochs', default=100001, type=int)
parser.add_argument('--reward', default=1, type=int)
parser.add_argument('--penalty', default=-1, type=float)
parser.add_argument('--penalize_early_stop', default=True, type=bool)
p = parser.parse_args()
np.random.seed(p.subj_id)
torch.manual_seed(p.subj_id)

# make log dirs
exp_name = f'n-{p.n}-n_std-{p.n_std}-v-{p.v}/h-{p.dim_hidden}/sub-{p.subj_id}'
log_path, fig_path = make_log_fig_dir(exp_name)

# init task
task = FreeRecall(n_std=p.n_std, n=p.n, v=p.v, reward=p.reward, penalty=p.penalty)

# init model
dim_input = task.x_dim * 2 + 1
dim_output = task.x_dim + 1
agent = Agent(dim_input, p.dim_hidden, dim_output, p.beta)
optimizer = torch.optim.Adam(agent.parameters(), lr=p.lr)

'''start training'''
len_test_phase = p.n_std + p.v + 1
log_r = [0 for i in range(p.n_epochs)]
log_p_rcl = np.zeros(p.n_epochs, )
log_a = np.full((p.n_epochs, len_test_phase), np.nan)
log_loss_actor = np.full((p.n_epochs, ), np.nan)
log_loss_critic = np.full((p.n_epochs, ), np.nan)
for i in range(p.n_epochs):
    time_s = time.time()
    # re-sample studied items
    X = task.sample(to_pytorch=True)
    # reset init state
    hc_t = agent.get_zero_states()
    # study phase
    for t, x_t_std in enumerate(X):
        hc_t = agent.add_normal_noise(hc_t, scale=p.noise_level)
        x_t = torch.cat([x_t_std, torch.zeros(task.x_dim), torch.zeros(1)])
        [_, _, _, hc_t], _ = agent.forward(x_t, hc_t)

    # recall phase
    log_r_i = []
    probs, rewards, values = [], [], []
    x_t_tst = torch.zeros(task.x_dim)
    r_t = torch.zeros(1)
    for t in range(len_test_phase):
        # feed _, prev_recall_item, prev_reward
        # hc_t = agent.add_normal_noise(hc_t, scale=p.noise_level)
        x_t = torch.cat([torch.zeros(task.x_dim), x_t_tst, r_t.view(1)])
        [a_t, pi_a_t, v_t, hc_t], _ = agent.forward(x_t, hc_t)
        # compute reward
        r_t = task.get_reward(a_t, penalize_early_stop=p.penalize_early_stop)
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

    # update beta
    if i > 0 and p.beta_decay_freq and i % p.beta_decay_freq == 0:
        p.beta *= p.beta_decay
        agent.update_beta(p.beta)

    # save weights
    if i % 10000 == 0:
        '''save weights'''
        fname = f'wts-{i}.pth'
        torch.save(agent.state_dict(), os.path.join(log_path, fname))

    if i > plt_win_size and i % 10000 == 0:
        # plot the learning curve
        # print(np.shape(log_r[:i]))
        tmp_ = np.array([np.mean(log_r_i) for log_r_i in log_r[:i]])
        f, ax = plot_learning_curve(tmp_, window_size=plt_win_size, ylabel='avg reward')
        f.savefig(os.path.join(fig_path, f'r-ep_{i}.png'))

    # compute ert
    time_took = time.time() - time_s
    if i == 0:
        ert = estimated_run_time(time_took, p.n_epochs)
        print('Estimated run time = %.2f hours' % (ert))

    log_p_rcl[i] = len(task.recalled_item_id) / len(task.studied_item_ids)
    if i % 100 == 0:
        print('%3d | r = %.4f, %% item recalled: %.2f loss a = %.4f, c = %.4f, beta = %.4f, time = %.2f sec' % (
        i, np.mean(log_r[i]), log_p_rcl[i], log_loss_actor[i], log_loss_critic[i], p.beta, time_took)
    )

'''figures'''
# plot the learning curves
f, ax = plot_learning_curve(log_loss_actor, window_size=plt_win_size, ylabel='loss actor')
f.savefig(os.path.join(fig_path, 'loss-a.png'))
f, ax = plot_learning_curve(log_loss_critic, window_size=plt_win_size, ylabel='loss critic')
f.savefig(os.path.join(fig_path, 'loss-c.png'))
