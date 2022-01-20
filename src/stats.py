import numpy as np
from scipy.stats import sem


def compute_stats(arr, axis=0, n_se=2):
    """compute mean and errorbar w.r.t to SE
    Parameters
    ----------
    arr : nd array
        data
    axis : int
        the axis to do stats along with
    n_se : int
        number of SEs
    Returns
    -------
    (n-1)d array, (n-1)d array
        mean and se
    """
    mu_ = np.mean(arr, axis=axis)
    er_ = sem(arr, axis=axis) * n_se
    return mu_, er_


def moving_average(x, winsize):
    return np.convolve(x, np.ones(winsize), 'valid') / winsize


def compute_recall_order(targ, resp):
    '''compute the recall order
    e.g.
    if target is [3, 1, 4, 2]
    and resp is [1, 1, 4, 5]
    then order is [1, 1, 2, np.nan]
    '''
    assert np.shape(targ) == np.shape(resp)
    (n_test, tmax) = np.shape(targ)
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
    return order


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


if __name__ == "__main__":

    for n_std_items in [4]:
    # n_std_items = 6
        temp = [- i-1 for i in np.arange(n_std_items-1)][::-1] + [i+1 for i in range(n_std_items-1)]
        print([lag2index(i, n_std_items) for i in temp])
        print(lag2index(0, n_std_items))
