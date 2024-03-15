# Internel packages
from src.logger import logging
from src.exception import CustomException
from src.Utils.utils import convert_columns_to_float, read_yaml, write_yaml_file

# Third-party libraries
import pandas as pd
import numpy as np
import os, sys
from scipy.stats import ks_2samp


class DataPreprocessingConfig:

    yaml_config = read_yaml(
        "D:\Machine_Learning_Projects\Sensor_App_Fault_Detection\Config\config.yaml"
    )


class DataPreprocessing:

    def __init__(self):
        logging.info("=================Data Preprocessing starts here=======================")
        try:

            self.data_transformation_config = DataPreprocessingConfig()
            self.validation_error=dict()
            self.validation_dir = self.data_transformation_config.yaml_config['data_validation']['validation_dir']
            self.missing_threshold = self.data_transformation_config.yaml_config['data_validation']['missing_threshold']
            self.train_data = self.data_transformation_config.yaml_config['data_ingestion']['train_data_path']
            self.base_data = self.data_transformation_config.yaml_config['data_ingestion']['raw_data_path']
            self.validation_report = self.data_transformation_config.yaml_config['data_validation']['report_file_path']

        except Exception as e:
            logging.info(f"Error occured with message: {e}")
            raise CustomException(e, sys)
        
    
    def drop_missing_columns(self, data:pd.DataFrame, report_key:str) -> pd.DataFrame:

        try:
            null_report = data.isna().sum() / data.shape[0]
            drop_column_names = null_report[null_report > self.missing_threshold].index

            # logging.info(f"the {drop_column_names} have more null values than {self.missing_threshold}")
            self.validation_error[report_key] = list(drop_column_names)

            new_data = data.drop(drop_column_names, axis= 1)

            return new_data

        except Exception as e:
            logging.info(f"Error occured in drop_missing_columns method in DataTransformation class with message: {e}")
            raise CustomException(e, sys)


    def data_distribution(self, raw_data:pd.DataFrame, train_data : pd.DataFrame, report_key : str) -> pd.DataFrame:

        try:

            distribution_report = dict()

            raw_data_columns = raw_data.columns
                    
            for col in raw_data_columns:
                raw_df, train_df = raw_data[col], train_data[col]
                # logging.info(f"Hypothesis {raw_data_columns} : {raw_df.dtype}, {train_df.dtype}")

                distribution = ks_2samp(raw_df, train_df)

                if distribution.pvalue > 0.05:
                    distribution_report[col] = {
                        "pvalue" : float(distribution.pvalue),
                        "Same distribution" : True
                    }

                else:
                    distribution_report[col] = {
                        "pvalue" : float(distribution.pvalue),
                        "Same distribution" : False
                    }


            self.validation_error[report_key] = distribution_report

            logging.info(f"The distribution report is: {distribution_report}")

            return distribution_report

        except Exception as e:
            logging.info(f"Error occured with message: {e}")
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self):
        
        try:

            raw_data = pd.read_csv(self.base_data)
            train_data = pd.read_csv(self.train_data)

            logging.info(f"The train and raw data read successfully")

            raw_data = raw_data.replace("na", np.NAN)
            train_data = train_data.replace("na", np.NAN)

            logging.info(f"The na values are replaces by NAN values")

            raw_data_new = self.drop_missing_columns(data=raw_data, report_key="missing_values_in_raw_data")
            train_data_new = self.drop_missing_columns(data=train_data, report_key="missing_values_in_train_data")

            logging.info(f"The columns which contains missing values are dropped")

            exclude_columns = ["class"]
            raw_data = convert_columns_to_float(df=raw_data_new, exclude_column=exclude_columns)
            train_data = convert_columns_to_float(df=train_data_new, exclude_column=exclude_columns)

            logging.info(f"The columns are converted into float")

            distribution = self.data_distribution(raw_data=raw_data, train_data=train_data, report_key="data_distribution")

            logging.info(f"The distribution of train and raw data is: \n{distribution}")

            os.makedirs(self.validation_dir, exist_ok=True)
            write_yaml_file(file_path=self.validation_report, data=self.validation_error)

            logging.info(f"The yaml file is saved at {os.path.split(self.validation_dir)[0]}")

            return distribution

        except Exception as e:
            logging.info(f"Error occured with message: {e}")

            raise CustomException(e, sys)
        
if __name__ == "__main__":
    config = DataPreprocessing()
    config.initiate_data_transformation()