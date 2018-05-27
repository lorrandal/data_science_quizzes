# -*- coding: utf-8 -*-
"""
DATA SCIENCE QUIZZES 

QUIZ 2

Implement a solution to the following problem

Imagine there is a country in which couples only want boys.
Couples continue to have children until they have their first boy.
If they get a boy, they stop getting children. 
What is the long-term ratio of boys to girls in the country?

"""
import random

# Function to simulate the birth of a Boy or a Girl
def decision(probability):
    return random.random() < probability

counter_boys = 0
counter_girls = 0
counter_families = 100000

for i in range(counter_families):
    while decision(0.5) == False: #birth of a Girl, continue until a Boy
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

# The long-term ratio (counter_families tends to infinity) of boys to girls is 1 