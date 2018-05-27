# -*- coding: utf-8 -*-
"""
DATA SCIENCE QUIZZES 

QUIZ 2
"""
import random

def decision(probability):
    return random.random() < probability

#%%
counter_boys = 0
counter_girls = 0
counter_families = 1000

for i in range(counter_families):
    while decision(0.5) == False: #girl
        counter_girls = counter_girls + 1
    counter_boys = counter_boys +1        
    
        

total_children = counter_boys + counter_girls
percentage_boys = counter_boys/total_children*100

print('boys percentage = %g' %percentage_boys)
print('total children = %i' %total_children)

