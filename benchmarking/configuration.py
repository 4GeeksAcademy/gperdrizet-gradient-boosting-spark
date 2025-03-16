'''Global configuration for hyperparameter optimization benchmark.'''

##################################
# Data files #####################
##################################

RAW_DATA_URL='https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv'
RAW_DATA='./data/diabetes_data_raw.parquet'
TRAINING_DATA='./data/diabetes_training_data.parquet'
TESTING_DATA='./data/diabetes_testing_data.parquet'
BENCHMARK_RESULTS='./data/benchmark_results.parquet'


##################################
# Hyperparameter search setup#####
##################################

REPLICATES=3
CROSS_VAL_FOLDS=3
MAX_DEPTH=5
BOOSTING_ROUNDS=50

HYPERPARAMETERS={
    'learning_rate':[0.05, 0.1, 0.2],
    'subsample':[0.4, 0.5, 0.6, 0.7]
}