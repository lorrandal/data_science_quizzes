# -*- coding: utf-8 -*-
"""
DATA SCIENCE QUIZZES 

QUIZ 1
"""
import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 1000, 10.0 # mean and standard deviation
s = np.random.normal(mu, sigma, 100000) # creat normal distribution

#abs(mu - np.mean(s)) < 0.01
#abs(sigma - np.std(s, ddof=1)) < 0.01

print(np.mean(s))
print(np.std(s))

distinct_values = len(set(s))
print(distinct_values)
#%% PLOT
count, bins, ignored = plt.hist(s, 30, normed=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
          linewidth=2, color='r')
plt.show()

