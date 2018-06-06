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
autoscaled_all_experiments = (all_experiments - all_experiments.mean(axis=0)) / all_experiments.std(axis=0, ddof=1)
for experiment_number in range(all_experiments.shape[0] - number_of_experiments):
    determinants = []
    autoscaled_all_experiments_tmp = autoscaled_all_experiments.copy()
    for calc_determinant_number in range(all_experiments.shape[0]):
        autoscaled_all_experiments_tmp = np.delete(autoscaled_all_experiments, calc_determinant_number, 0)
        determinants.append(np.linalg.det(np.dot(autoscaled_all_experiments_tmp.T, autoscaled_all_experiments_tmp)))
    selected_number = np.where(determinants == max(determinants))[0][0]
    all_experiments = np.delete(all_experiments, selected_number, 0)
    autoscaled_all_experiments = np.delete(autoscaled_all_experiments, selected_number, 0)

selected_experiments_df = pd.DataFrame(all_experiments)
selected_experiments_df.to_csv('selected_experiments.csv', header=False, index=False)
