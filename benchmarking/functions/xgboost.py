'''XGBoost utility functions'''

import time
import itertools

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import configuration as config

def run(results:dict) -> None:
    '''Runs XGBoost hyperparameter optimization benchmark.'''

    hyperparameters={
        'eta': config.HYPERPARAMETERS['learning_rate'],
        'subsample': config.HYPERPARAMETERS['subsample'],
        'objective': ['binary:logistic']
    }

    # Make search space combinations
    parameter_sets=search_space_samples(**hyperparameters)

    # Load the data
    training_df=pd.read_parquet(config.TRAINING_DATA)

    for i in range(config.REPLICATES):
        print(f'Running XGBoost replicate: {i+1}')

        start_time=time.time()

        # Loop on hyperparameter samples
        for parameter_set in parameter_sets:

            # Cross-validate with the hyperparameters
            _=xgb_cross_val(
                parameter_set,
                training_df
            )

        results['Library'].append('XGBoost')
        results['Search time (sec.)'].append(time.time()-start_time)

    return results


def search_space_samples(**search_space):
    '''Takes a dictionary of hyperparameters, where key is string name
    and value is a list of values. Returns individual dictionaries 
    containing the cartesian product of all hyperparameter values'''
    
    parameters=search_space.keys()

    for values in itertools.product(*search_space.values()):
        yield dict(zip(parameters, values))


def xgb_cross_val(
    parameter_set: dict,
    training_df: pd.DataFrame,
    label: str='Outcome'
):
    
    '''Cross-validates an XGBoost model.'''

    # Cross-validation splitter
    k_fold=KFold(n_splits=config.CROSS_VAL_FOLDS)

    # Collector for scores
    scores=[]

    # Loop on cross-validation folds
    for _, (training_idx, validation_idx) in enumerate(k_fold.split(training_df)):

        # Get the split for this fold
        split_training_df=training_df.iloc[training_idx]
        split_validation_df=training_df.iloc[validation_idx]

        # Convert to DMaxtrix for XGBoost training
        dtraining=xgb.DMatrix(split_training_df.drop(label, axis=1), label=split_training_df[label])
        dvalidation=xgb.DMatrix(split_validation_df.drop(label, axis=1), label=split_validation_df[label])

        # Train the model
        model=xgb.train(
            parameter_set,
            dtraining,
            num_boost_round=config.BOOSTING_ROUNDS,
            evals=[(dvalidation, 'validation')],
            verbose_eval=0
        )

        # Get validation RMSE for this fold
        predictions=model.predict(dvalidation)
        predictions=[round(value) for value in predictions]
        scores.append(accuracy_score(split_validation_df[label], predictions))

    return scores