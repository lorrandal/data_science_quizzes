# -*- coding: utf-8 -*-
"""
DATA SCIENCE QUIZZES 

QUIZ 1
"""
import numpy as np

mu, sigma = 1000, 10.0 # mean and standard deviation
s = np.random.normal(mu, sigma, 100000) # creat normal distribution

print('mean = %f' %np.mean(s))
print('std = %f' %np.std(s))
print('distinct_values = %i' %len(set(s)))
