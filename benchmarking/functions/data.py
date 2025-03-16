'''Data preparation utility functions.'''


from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

import configuration as config

def prep_data() -> None:
    '''Downloads and prepares diabetes dataset for modeling. Saves output
    as pickled dictionary object with 'training' and 'testing' keys 
    containing Pandas dataframes.'''

    if not Path(config.RAW_DATA).is_file():
        data_df=pd.read_csv(config.RAW_DATA_URL)
        data_df.to_parquet(config.RAW_DATA)

    if not Path(config.TRAINING_DATA).is_file():
        data_df=pd.read_parquet(config.RAW_DATA)
        data_df.drop_duplicates().reset_index(drop=True, inplace=True)

        training_df, testing_df=train_test_split(
            data_df,
            test_size=0.25,
            random_state=315
        )

        imputed_features=['Insulin','SkinThickness','BloodPressure','BMI','Glucose']
        knn_imputer=KNNImputer(missing_values=0.0, weights='distance')
        knn_imputer.fit(training_df[imputed_features])
        training_df[imputed_features]=knn_imputer.transform(training_df[imputed_features])
        testing_df[imputed_features]=knn_imputer.transform(testing_df[imputed_features])

        training_df.to_parquet(config.TRAINING_DATA)
        testing_df.to_parquet(config.TESTING_DATA)

    return