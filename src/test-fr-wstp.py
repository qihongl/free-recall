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
from utils import to_sqpth, to_pth, to_np, to_sqnp, make_log_fig_dir, rm_dup, int2onehot
from stats import compute_stats, compute_recall_order, lag2index
from vis import plot_learning_curve
from sklearn.linear_model import RidgeClassifier

sns.set(style='white', palette='colorblind', context='talk')

subj_id = 0
# for subj_id in range(6):
np.random.seed(subj_id)
torch.manual_seed(subj_id)
print(subj_id)

# init task
n = 40
n_std_tr = 10
n_std_te = 10
v_tr = 3
v_te = 0
reward = 1
penalty = -.5
dim_hidden = 128
beta = 1
# noise_level = 0.025
noise_level = 0

task = FreeRecall(n_std=n_std_te, n=n, v=v_te, reward=reward, penalty=penalty)
dim_input = task.x_dim * 2 + 1
dim_output = task.x_dim + 1

# make log dirs
epoch_trained = 100000
exp_name = f'n-{n}-n_std-{n_std_tr}-v-{v_tr}/h-{dim_hidden}/sub-{subj_id}'
log_path, fig_path = make_log_fig_dir(exp_name, makedirs=False)

# reload the weights
fname = f'wts-{epoch_trained}.pth'
agent = Agent(dim_input, dim_hidden, dim_output, beta)
agent.load_state_dict(torch.load(os.path.join(log_path, fname)))

# testing
n_test = 2000
len_test_phase = task.n_std + v_tr + 1
# log_r = np.zeros((n_test, n_std_te))
log_r = [0 for i in range(n_test)]
log_a = np.full((n_test, len_test_phase), np.nan)
log_std_items = np.zeros((n_test, n_std_te))
log_h_std = np.full((n_test, task.n_std, dim_hidden), np.nan)
log_h_tst = np.full((n_test, len_test_phase, dim_hidden), np.nan)
i=0
for i in range(n_test):
    # re-sample studied items
    X = task.sample(to_pytorch=True)
    log_std_items[i] = task.studied_item_ids
    # study phase
    r_t = torch.zeros(1)
    hc_t = agent.get_zero_states()
    for t, x_t_std in enumerate(X):
        hc_t = agent.add_normal_noise(hc_t, scale=noise_level)
        x_t = torch.cat([x_t_std, torch.zeros(task.x_dim), r_t])
        [_, _, _, hc_t], _ = agent.forward(x_t, hc_t)
        log_h_std[i, t] = to_sqnp(hc_t)
    # recall phase
    log_r_i = []
    x_t_tst = torch.zeros(task.x_dim)
    for t in range(len_test_phase):
        x_t = torch.cat([torch.zeros(task.x_dim), x_t_tst, r_t.view(1)])
        [a_t, pi_a_t, v_t, hc_t], _ = agent.forward(x_t, hc_t)

        r_t = task.get_reward(a_t)
        log_r_i.append(to_np(r_t))
        # log
        log_a[i, t] = a_t
        log_h_tst[i, t] = to_sqnp(hc_t)
        # if the model choose the stop, break
        if int(to_np(a_t)) == task.n:
            break
        # make a onehot that represent the recalled item at time t
        x_t_tst = int2onehot(a_t, task.n)
    # collect the reward
    log_r[i] = log_r_i

'''figures'''

# select the last n_test trials to analyze
targ = log_std_items
resp = log_a

'''compute error info'''
# compute number of lures for each trials
n_lures = np.zeros(n_test)
stop_times = np.full(n_test, np.nan)
for i in range(n_test):
    stop_time = np.where(resp[i] == n)
    # trim the trial at the stop point
    if len(stop_time[0]) > 0:
        stop_times[i] = int(stop_time[0])
        resp_i = resp[i][:int(stop_time[0])]
    else:
        # stop_times[i] = len_test_phase
        resp_i = resp[i]
    # count the number of items that are not in the targets
    for resp_it in resp_i:
        if resp_it not in targ[i]:
            n_lures[i] +=1

n_lures_mu, n_lures_se = compute_stats(n_lures)
stop_times_mu = np.nanmean(stop_times)


# compute number of repeats for each trials
repeats = np.zeros(n_test)
for i in range(n_test):
    recalled_items_i = log_a[i][~np.isnan(log_a[i])]
    # count the number of redundent items
    repeats[i] = len(recalled_items_i) - len(np.unique(recalled_items_i))
repeats_mu, repeats_se = compute_stats(repeats)

print(f'Lures   : mu = %.2f, se = %.2f' % (n_lures_mu, n_lures_se))
print(f'Repeats : mu = %.2f, se = %.2f' % (repeats_mu, repeats_se))
print(f'Stop time : mu = %.2f' % (stop_times_mu))


'''compute recall order '''

# compute the recall order given target and responses
order = compute_recall_order(targ, resp)

# compute the tally for actual and possible responses
tally = np.zeros((n_test, (n_std_te - 1)* 2))
tally_poss = np.zeros((n_test, (n_std_te - 1)* 2))
lags = []
for i in range(n_test):
    order_i = order[i]
    # order_i_rmnan = order_i[~np.isnan(order_i)]
    order_i_rmnan = order_i[~np.isnan(order_i)]
    order_i_rmnan = rm_dup(order_i_rmnan)
    # order_i_rmnan = list(dict.fromkeys(order_i_rmnan))
    for j in range(len(order_i_rmnan) - 1):
        lag = int(order_i_rmnan[j+1] -  order_i_rmnan[j])
        if lag != 0:
            lag_index = lag2index(lag, n_std_te)
            # increment the count of the corresponding lag index
            tally[i, lag_index] +=1
            lags.append(lag)

    for j in range(len(order_i_rmnan) - 1):
        for o in range(n_std_te):
            lag_poss = int(o - order_i_rmnan[j])
            if lag_poss != 0:
                lag_index = lag2index(lag_poss, n_std_te)
                tally_poss[i, lag_index] +=1

assert np.all(tally_poss >= tally), 'possible count must >= actual count'
# compute conditional response probability
crp = np.divide(tally, tally_poss, out=np.zeros_like(tally), where=tally_poss!=0)
# crp = tally / tally.sum(axis=1)[:,None]

p_mu, p_se = compute_stats(crp, axis=0)
xticklabels = np.concatenate((np.arange(-(n_std_te - 1), 0), np.arange(1, n_std_te)))

f, ax = plt.subplots(1,1, figsize=(6,4))
ax.errorbar(x=np.arange(n_std_te - 1), y=p_mu[:n_std_te - 1], yerr = p_se[:n_std_te - 1],color='k')
ax.errorbar(x=np.arange(n_std_te - 1) + (n_std_te - 1), y=p_mu[-(n_std_te - 1):], yerr = p_se[-(n_std_te - 1):], color='k')
ax.set_ylim([0, None])
ax.set_xticks(np.arange((n_std_te - 1)*2))
ax.set_xticklabels(xticklabels)
ax.set_xlabel('Lag')
ax.set_ylabel('p')
ax.set_title('Conditional response probability')
sns.despine()

''' compute recall probabiliy arranged according to the studied order'''
# order = order[:,0]
# plot the serial position curve
unique_recalls = np.concatenate([np.unique(order[i]) for i in range(n_test)])
counter = Counter(unique_recalls)
recalls = np.array([counter[i] for i in range(n_std_te)])

spc_x = range(n_std_te)
spc_y = recalls / n_test
f, ax = plt.subplots(1,1, figsize=(6, 4))
ax.plot(spc_y)
ax.set_ylim([0, 1])
ax.set_xlabel('Position')
ax.set_ylabel('p')
ax.set_title('Recall probability')
sns.despine()

f, ax = plt.subplots(1,1, figsize=(6, 4))
ax.plot(spc_y)
ax.set_xlabel('Position')
ax.set_ylabel('p')
ax.set_title('Recall probability')
sns.despine()

''' compute p(1st recall) '''
counter = Counter(order[:,0])
recalls_1st = np.array([counter[i] for i in range(n_std_te)])
f, ax = plt.subplots(1,1, figsize=(6, 4))
ax.plot(recalls_1st / np.sum(recalls_1st))
# ax.set_ylim([None, 1])
ax.set_ylim([0, None])
ax.set_xlabel('Position')
ax.set_ylabel('p')
ax.set_title('The distribution of the 1st recall')
sns.despine()

f, ax = plt.subplots(1,1, figsize=(6, 4))
ax.plot(recalls_1st / np.sum(recalls_1st))
ax.set_xlabel('Position')
ax.set_ylabel('p')
ax.set_title('The distribution of the 1st recall')
sns.despine()


''' compute mean performance '''
n_items_recalled = np.empty(n_test, )
for i in range(n_test):
    order_i = np.unique(order[i])
    n_items_recalled[i] = len(order_i[~np.isnan(order_i)])
prop_item_recalled = n_items_recalled / n_std_te

pir_mu, pir_se = compute_stats(prop_item_recalled)
print(f'%% items recalled = %.2f, se = %.2f' % (pir_mu, pir_se))

mean_r_all_trials = [np.mean(log_r_) for log_r_ in log_r]
r_mu, r_se = compute_stats(mean_r_all_trials)
print(f'Average reward = %.2f, se = %.2f' % (r_mu, r_se))




'''MVPA decoding'''
n_tr = 1000
n_te = n_test - n_tr

y_te_std_hat = np.zeros((n, n_tr, n_std_te))
y_te_std = np.zeros((n, n_tr, n_std_te))

dec_acc = np.zeros(n)

# reshape X
log_h_std_rs = np.reshape(log_h_std, (-1, dim_hidden))
ridge_clfs = [None for _ in range(n)]

# train/test the clf on the study phase
for std_item_id in range(n):
    # make y - whether item i has been presented in the past
    pres_time_item_i = log_std_items == std_item_id
    inwm_time_item_i = np.zeros_like(pres_time_item_i)
    for i in range(n_test):
        if np.any(pres_time_item_i[i, :]):
            t = int(np.where(pres_time_item_i[i, :])[0])
            inwm_time_item_i[i, t:] = 1

    # reshape y for the classifier for the study phase
    inwm_time_item_i_rs = np.reshape(inwm_time_item_i, (-1,))

    # split the data
    x_tr = log_h_std_rs[:n_tr* n_std_te]
    y_tr = inwm_time_item_i_rs[:n_tr* n_std_te]
    x_te = log_h_std_rs[n_tr* n_std_te:]

    # fit model
    ridge_clfs[std_item_id] = RidgeClassifier().fit(x_tr, y_tr)
    # test the clf on the study phase test set
    y_te_std_hat_rs = ridge_clfs[std_item_id].predict(x_te)
    y_te_std_hat[std_item_id] = np.reshape(y_te_std_hat_rs, (-1, n_std_te))
    y_te_std[std_item_id] = inwm_time_item_i[n_tr:]

# generalize the clf to the test phase
y_te_tst_hat = np.full((n, n_tr, len_test_phase), np.nan)
# test phase
for std_item_id in range(n):
    for i in range(n_tr):
        for t in range(len_test_phase):
            if not np.isnan(log_h_tst[n_tr + i, t]).any():
                y_te_tst_hat[std_item_id, i, t] = ridge_clfs[std_item_id].predict(
                    log_h_tst[n_tr + i, t].reshape(1,-1)
                )




'''helper func - decoding accuracy over time for different order info'''
def compute_hit_by_order(ridge_hits, order_info):
    dec_acc_dict = {i:[] for i in range(n_std_te)}
    # loop over all test set examples
    for i in np.arange(n_te):
        # for each test set trial, loop over all studied items
        for std_o, item_id in enumerate(order_info[n_tr + i]):
            if std_o >= n_std_te: # if std_o > n_std_te -> test phase buffer time
                break
            # whether the o-th studied item was a hit
            if np.isnan(item_id) or item_id == n:
                continue
            dec_acc_dict[std_o].append(ridge_hits[int(item_id), i])
    # compute average for all studied order
    dec_acc_mu = np.zeros((n_std_te, n_std_te))
    dec_acc_se = np.zeros((n_std_te, n_std_te))
    for o in range(n_std_te):
        dec_acc_mu[o], dec_acc_se[o] = compute_stats(np.stack(dec_acc_dict[o]))
    return dec_acc_mu, dec_acc_se

def plot_decoding_curves(dec_acc_mu, dec_acc_se, study_phase=True):
    cpal = sns.color_palette("Spectral", n_colors = n_std_te)
    f, ax = plt.subplots(1,1, figsize=(6, 4))
    for o in range(n_std_te):
        ax.errorbar(
            x=np.arange(o, n_std_te), y=dec_acc_mu[o, o:], yerr=dec_acc_se[o, o:],
            color = cpal[o], marker='.'
        )
    sns.despine()
    if study_phase:
        ax.set_xlabel(f'Time (study phase)')
    else:
        ax.set_xlabel(f'Time (test phase)')
    ax.set_ylabel('Decoding accuracy')
    return f, ax


# compute accuracy
ridge_hits_std = y_te_std_hat == y_te_std
ridge_hits_by_item_id = np.mean(ridge_hits_std, axis=0)
dec_acc_mu, dec_acc_se = compute_stats(np.mean(ridge_hits_by_item_id,axis=0))
print('the overall decoding accuracy is %.2f, se = %.2f' % (dec_acc_mu, dec_acc_se))


'''compute the decoding accuracy over time for different study order index'''
dec_acc_mu, dec_acc_se = compute_hit_by_order(ridge_hits_std, log_std_items)
f, ax = plot_decoding_curves(dec_acc_mu, dec_acc_se, True)

'''compute the same thing during the test phase'''
# compare prediction to 1 since all items were presented
ridge_hits_tst = y_te_tst_hat == 1
dec_acc_mu, dec_acc_se = compute_hit_by_order(ridge_hits_tst[:,:,:n_std_te], log_a)
f, ax = plot_decoding_curves(dec_acc_mu, dec_acc_se, False)
