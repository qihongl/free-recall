import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from stats import compute_stats, moving_average
sns.set(style='white', palette='colorblind', context='talk')
cpal = sns.color_palette()


def plot_learning_curve(
    data, ylabel='MSE', window_size=1, alpha=.3, final_error_winsize=100, figsize=(8, 4)
):
    """plot learning curve, input needs to be 1D
    Parameters
    ----------
    data : type
        Description of parameter `data`.
    ylabel : type
        Description of parameter `ylabel`.
    window_size : type
        Description of parameter `window_size`.
    alpha : type
        Description of parameter `alpha`.
    final_error_winsize : type
        Description of parameter `final_error_winsize`.
    figsize : type
        Description of parameter `figsize`.
    Returns
    -------
    type
        Description of returned object.
    """
    f, ax = plt.subplots(1, 1, figsize=figsize)
    if window_size > 1:
        smoothed_data = moving_average(data, window_size)
        smoothed_data_lab = f'smoothed (window size = {window_size})'
        ax.plot(smoothed_data, color=cpal[0], label=smoothed_data_lab)
        ax.plot(data, alpha=alpha, color=cpal[0], label='raw data')
        ax.legend()
    else:
        ax.plot(data, color=cpal[0])
    ax.set_xlabel('Epochs')
    ax.set_ylabel(ylabel)
    # compute average final error from the last 100 epochs
    final_ymean = np.mean(data[-min(final_error_winsize, len(data)):])
    ax.set_title('Final %s = %.4f' % (ylabel, final_ymean))
    sns.despine()
    f.tight_layout()
    return f, ax


def make_line_plot_wer(d_mu, d_se, figsize, xlabel, ylabel, ylim=[None, None]):
    '''make a line plot with error bar'''
    f, ax = plt.subplots(1, 1, figsize=figsize)
    ax.errorbar(x=range(len(d_mu)), y=d_mu, yerr=d_se)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(ylim)
    f.tight_layout()
    sns.despine()
    return f, ax
