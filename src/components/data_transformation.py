import pandas as pd
import numpy as np
from src.exception import CustomException
from sklearn.preprocessing import LabelEncoder
import sys
from src.logger import logging
from dataclasses import dataclass
import pickle
import os


@dataclass
class DataTransformationConfig:
    transformer :str = os.path.join('models','transformer.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            model = LabelEncoder() # Create the model for the conversion of sentiments
            return model
        except Exception as e:
            raise CustomException(e,sys)
        
    def intiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path) # Read the test and train data
            test_df = pd.read_csv(test_path)
            logging.info("Reading the train and test object.")
            transformer = self.get_data_transformation_object()
            print(train_df.sentiment.head())
            train_df['sentiment'] = transformer.fit_transform(train_df['sentiment']) ## Transforming the sentiments into the model expected format
            test_df['sentiment'] = transformer.transform(test_df['sentiment'])
            with open(self.data_transformation_config.transformer,'wb') as f:
                pickle.dump(transformer,f,protocol=5)
                logging.info("Transformation object saved.")
            return train_df, test_df
    
        except Exception as e:
            raise CustomException(e,sys)


            
