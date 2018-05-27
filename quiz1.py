# -*- coding: utf-8 -*-
"""
DATA SCIENCE QUIZZES 

QUIZ 1

Create a list of numbers with the following properties: 1) Minimum 100 distinct values, 2) the Mean of all values is 1000 (+/- 0.5), 3) the standard Deviation is 10 (+/- 0.1) 

"""
import numpy as np

mu, sigma = 1000, 10.0 # mean and standard deviation
s = np.random.normal(mu, sigma, 100000) # create normal distribution

print('distinct_values = %i' %len(set(s)))
print('mean = %f' %np.mean(s))
print('std = %f' %np.std(s))
