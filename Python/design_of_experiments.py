# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Design of Experiments (DoE)

# Outputs are 'all_experiments.csv' and 'selected_experiments.csv'
# 'all_experiments.csv' includes all possible experiments and
# 'selected_experiments.csv' includes experiments selected from them.

# In settings, you can change the contents of 'variable1', 'variable2' and 'variable3',
# delete 'variable3', and add 'variable4', 'variable5', ... after 'variable3'

import numpy as np
import pandas as pd
from numpy import matlib

# settings
number_of_experiments = 30
variables = {'variable1': [1, 2, 3, 4, 5],
             'variable2': [-10, 0, 10, 20],
             'variable3': [0.2, 0.6, 0.8, 1, 1.2]
             }
# you can add 'variable4', 'variable5', ... after 'variable3' as well 

optimal_type = 'D' # D, I
number_of_random_searches = 1000
# make all possible experiments
all_experiments = np.array(variables['variable1'])
all_experiments = np.reshape(all_experiments, (all_experiments.shape[0], 1))
for variable_number in range(2, len(variables) + 1):
    grid_seed = variables['variable{0}'.format(variable_number)]
    grid_seed_tmp = matlib.repmat(grid_seed, all_experiments.shape[0], 1)
    all_experiments = np.c_[matlib.repmat(all_experiments, len(grid_seed), 1),
                            np.reshape(grid_seed_tmp.T, (np.prod(grid_seed_tmp.shape), 1))]

all_experiments_df = pd.DataFrame(all_experiments)
all_experiments_df.to_csv('all_experiments.csv', header=False, index=False)

# select experiments
np.random.seed(100) # fix random number seed
experiment_indexes = np.arange(0, all_experiments.shape[0])
autoscaled_all_experiments = (all_experiments - all_experiments.mean(axis=0)) / all_experiments.std(axis=0, ddof=1)
for random_search_number in range(number_of_random_searches):
    new_selected_experiment_numbers = np.random.randint(0, all_experiments.shape[0], number_of_experiments)
    new_selected_experiments = all_experiments[new_selected_experiment_numbers, :]
    autoscaled_new_selected_experiments = (new_selected_experiments - new_selected_experiments.mean(axis=0)) / new_selected_experiments.std(axis=0, ddof=1)
    if optimal_type == 'D':
        optimal_value = np.linalg.det(np.dot(autoscaled_new_selected_experiments.T, autoscaled_new_selected_experiments))
    elif optimal_type == 'I':
        optimal_value = sum(np.diag(autoscaled_new_selected_experiments.dot(np.linalg.inv(np.dot(autoscaled_new_selected_experiments.T, autoscaled_new_selected_experiments) / autoscaled_new_selected_experiments.shape[0])).dot(autoscaled_new_selected_experiments.T)))
        
    if random_search_number == 0:
        best_optimal_value = optimal_value
        selected_experiment_indexes = new_selected_experiment_numbers.copy()
    else:
        if optimal_type == 'D':
            if best_optimal_value < optimal_value:
                print(best_optimal_value)
                selected_experiment_indexes = new_selected_experiment_numbers.copy()
                best_optimal_value = optimal_value
        elif optimal_type == 'I':
            if best_optimal_value > optimal_value:
                print(best_optimal_value)
                selected_experiment_indexes = new_selected_experiment_numbers.copy()
                best_optimal_value = optimal_value

selected_experiments_df = pd.DataFrame(all_experiments[selected_experiment_indexes, :])
selected_experiments_df.to_csv('selected_experiments.csv', header=False, index=False)
