import torch
import numpy as np
from utils import to_pth, to_np


class FreeRecall():
    '''
    learn n_std (out of n) items during study, and then recall during test
    '''

    def __init__(self, n_std=8, n=50, v=0, reward=1, penalty=-1):
        """init the free recall task.

        Parameters
        ----------
        n_std : int
            the (mean) number of studied items
        n : int
            the number of possible words/tokens
        v : int
            the variation of number of studied items
            the actual studied items for a particular trial is uniformly
            sampled from U(n_std-v, n_std+v)
        reward : float
            the size of the reward when recalling a studied item (the 1st time)
            and stop the trial after recalling all items
        penalty : float
            the size of the reward when recalling a lure or recalling a
            previously recalled item

        """
        assert n > n_std + v >= n_std > 1, 'n words > n study items > 1'
        assert reward > 0 and penalty <= 0, 'reward/penalty must be pos/neg'
        self.n = n
        self.v = v
        self.n_std = n_std
        self.reward = reward
        self.penalty = penalty
        # init helpers
        self._init_stimuli()

    def _init_stimuli(self, option='onehot'):
        if option == 'onehot':
            self.stimuli = np.eye(self.n)
            self.x_dim = self.n
        else:
            raise NotImplementedError()
        # prealloc studied ids
        self.studied_item_ids = None

    def sample(self, to_pytorch=False):
        '''
        randomly set the ids for the studied items, and return their one hots
        '''
        # reinit the list of recalled items
        self.recalled_item_id = []
        # decide the number of study items
        if self.v > 0:
            n_std = self.n_std + np.random.randint(low=-self.v, high=+self.v)
        else:
            n_std = self.n_std
        # sample the studied items
        self.studied_item_ids = np.random.choice(
            range(self.n), size=n_std, replace=False
        )
        # construct the input / onehot reps
        stimuli = np.zeros((n_std, self.x_dim))
        for i, id in enumerate(self.studied_item_ids):
            stimuli[i,:] = self.stimuli[id]
        # to pytorch format
        if to_pytorch:
            stimuli = to_pth(stimuli)
        return stimuli

    def get_reward(self, recalled_id, penalize_early_stop=True):
        '''
        return reward/penalty if the model recalled some studied item / lure

        repeated recall of studied item will be penalized

        stop when all std items were recal will be rewarded,
        but early stopping will be penalized
        - this function should work if the model doesn't have a stop unit
        '''
        if self.studied_item_ids is None:
            raise ValueError('studied_item_ids is none, call sample() first' )
        # convert action to numpy type is necessary
        if torch.is_tensor(recalled_id):
            recalled_id = to_np(recalled_id)
        # if the action is the stop unit
        if recalled_id == self.n:
            if len(self.recalled_item_id) == len(self.studied_item_ids):
                return to_pth(self.reward)
            else:
                if penalize_early_stop:
                    return to_pth(self.penalty)
                else:
                    return to_pth(0)
        # if recalled item is a lure, penalty
        if recalled_id not in self.studied_item_ids:
            return to_pth(self.penalty * 1.5)
        # here, recalled item is studied
        # if recalled item has been recalled
        if recalled_id in self.recalled_item_id:
            # penalize repeat
            return to_pth(self.penalty * 1.5)
        # if is studied but hasn't been recalled, reward the model
        else:
            self.recalled_item_id.append(recalled_id)
            return to_pth(self.reward)





if __name__ == "__main__":
    '''how to use'''
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='white', palette='colorblind', context='poster')
    np.random.seed()

    '''simulation - fixed the number of study items '''
    n_std = 6
    n = 10
    task = FreeRecall(n_std, n)
    X = task.sample()
    # print(task.stimuli)
    print(task.studied_item_ids)
    print(X)
    plt.imshow(X)

    for a_t in range(n):
        print(a_t, task.get_reward(a_t))

    print(task.recalled_item_id)
    a_t = 0
    print(a_t, task.get_reward(a_t))


    '''testing - optimal policy'''
    test_epochs = 10
    log_r = np.zeros(test_epochs)
    for i in range(test_epochs):
        _ = task.sample()
        r = 0
        for t in range(task.n_std):
            a_t = task.studied_item_ids[t]
            r += task.get_reward(a_t)
        log_r[i] = r
    print(log_r)

    '''simulation - VARY the number of study items '''
    n_std = 6
    v = 3
    n = 10
    task = FreeRecall(n_std=n_std, n=n, v=v)
    X = task.sample()
    # print(task.stimuli)
    print(task.studied_item_ids)
    # print(X)
    # plt.imshow(X)

    log_r = []
    for t, a_t in enumerate(task.studied_item_ids):
        r_t = task.get_reward(a_t)
        log_r.append(to_np(r_t))
    r_t = task.get_reward(task.n)

    log_r.append(to_np(r_t))
    print(np.array(log_r))
