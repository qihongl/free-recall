import numpy as np
from utils import to_pth

# import pdb


class FreeRecall():
    '''
    learn n/2 (out of n) items during study, and then recall during test
    '''

    def __init__(self, n_std=8, n=50, reward=1, penalty=-1, penalize_repeat=True):
        assert n > n_std > 1, 'n words > n study items > 1'
        assert reward > 0 and penalty <= 0, 'reward/penalty must be pos/neg'
        self.n = n
        self.n_std = n_std
        self.reward = reward
        self.penalty = penalty
        self.penalize_repeat = penalize_repeat
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

    def set_penalize_repeat(self, penalize_repeat):
        self.penalize_repeat = penalize_repeat

    def sample(self, to_pytorch=False):
        '''
        randomly set the ids for the studied items, and return their one hots
        '''
        # sample the studied items
        self.studied_item_ids = np.random.choice(
            range(self.n), size=self.n_std, replace=False
        )
        self.recalled_item_id = []
        X = np.zeros((self.n_std, self.x_dim))
        # construct the input / onehot reps
        for i, id in enumerate(self.studied_item_ids):
            X[i,:] = self.stimuli[id]
        # to pytorch format
        if to_pytorch:
            X = to_pth(X)
        return X

    def get_reward(self, recalled_id):
        '''
        return reward/penalty if the model recalled some studied item / lure
        '''
        if self.studied_item_ids is None:
            raise ValueError('studied_item_ids is none, call sample() first' )

        if recalled_id in to_pth(self.studied_item_ids):
            if recalled_id in self.recalled_item_id:
                if self.penalize_repeat:
                    return to_pth(self.penalty)
                else:
                    return to_pth(0)
            else:
                self.recalled_item_id.append(recalled_id)
                return to_pth(self.reward)
        return to_pth(self.penalty)



if __name__ == "__main__":
    '''how to use'''
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='white', palette='colorblind', context='poster')
    np.random.seed()
    n_std = 5
    n = 30
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
