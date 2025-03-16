'''Runs benchmark of hyperparameter optimization rate for gradient boosting models.

Independent variables:

1. Gradient boosting model library
2. Hyperparameter search space size

Dependent variables:

1. Hyperparameter search rate (parameter sets/minute)

See results.ipynb
'''

from functions.data import prep_data


if __name__ == '__main__':
    

    # Get the data ready
    prep_data()

