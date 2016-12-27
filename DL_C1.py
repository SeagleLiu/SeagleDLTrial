"""Softmax."""
#%%
scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    tExp = np.exp(x)
    logit = tExp/np.sum(tExp, axis=0)
    return logit;

#%%
print(softmax(scores))

#%%
# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()

#%%
scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])
tl = softmax(scores)
print(tl)
np.sum(tl, axis=0)