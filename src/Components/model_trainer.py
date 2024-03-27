from src.logger import logging
from src.exception import CustomException
from src.Utils.utils import read_yaml, evaluate_model
import sys

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class ModelTrainerConfig:

    yaml_config = read_yaml(
        "D:\Machine_Learning_Projects\Sensor_App_Fault_Detection\Config\config.yaml"
    )


class ModelTrainer:

    def __init__(self):

        try:

            self.config = ModelTrainerConfig()
            self.train_data = self.config.yaml_config['preprocessed_train_data']
            self.test_data = self.config.yaml_config['preprocessed_test_data']


        except Exception as e:
            logging.error(f"The yaml file not found: {e}")
            raise FileNotFoundError(f"Yaml file not found")
    

    def initiate_model_trainer(self):

        try:
            
            pass
        
        except Exception as e:
            logging.error(f"Error occured at initiate_model_trainer method with: {e}")
            raise CustomException(e, sys)