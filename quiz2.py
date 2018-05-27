# -*- coding: utf-8 -*-
"""
DATA SCIENCE QUIZZES 

QUIZ 2
"""
import random

# Function to simulate the birth of a Boy or Girl
def decision(probability):
    return random.random() < probability

counter_boys = 0
counter_girls = 0
counter_families = 100000

for i in range(counter_families):
    while decision(0.5) == False: #girl
        counter_girls = counter_girls + 1
    counter_boys = counter_boys + 1        
    
        

total_children = counter_boys + counter_girls
percentage_boys = counter_boys / total_children * 100
percentage_girls = counter_girls / total_children * 100
ratio_of_boys_to_girls = counter_boys / counter_girls

print('boys percentage = %g' %percentage_boys)
print('girls percentage = %g' %percentage_girls)
print('total children = %i' %total_children)
print('Ratio of boys to girls = %g' %ratio_of_boys_to_girls)