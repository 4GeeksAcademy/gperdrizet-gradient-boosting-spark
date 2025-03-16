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

CROSS_VAL_FOLDS=3

HYPERPARAMETERS={
    'max_depth_range':[2,20],
    'subsample_range':[0.4,0.9]
}