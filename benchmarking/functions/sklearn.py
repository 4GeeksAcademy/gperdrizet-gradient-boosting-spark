'''Scikit-learn utility functions.'''

import time
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

import configuration as config


def run(results:dict) -> None:
    '''Runs Scikit-learn hyperparameter optimization benchmark.'''

    training_df=pd.read_parquet(config.TRAINING_DATA)

    for i in range(config.REPLICATES):
        print(f'Running Scikit-learn replicate: {i+1}')

        start_time=time.time()

        model=GradientBoostingClassifier(
            max_depth=config.MAX_DEPTH,
            n_estimators=config.BOOSTING_ROUNDS
        )

        hyperparameters={
            'learning_rate': config.HYPERPARAMETERS['learning_rate'],
            'subsample': config.HYPERPARAMETERS['subsample']
        }

        search=GridSearchCV(
            model,
            hyperparameters,
            cv=config.CROSS_VAL_FOLDS,
            n_jobs=-1
        )

        _=search.fit(training_df.drop('Outcome', axis=1), training_df['Outcome'])

        results['Library'].append('Scikit-learn')
        results['Search time (sec.)'].append(time.time()-start_time)

    return results