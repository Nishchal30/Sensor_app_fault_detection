# Internel packages
from src.logger import logging
from src.exception import CustomException
from src.Utils.utils import read_yaml, data_dump_to_mongodb

# Third-party libraries
import pandas as pd
import numpy as np
import os, sys
from sklearn.model_selection import train_test_split


# Config class to create a artifact folder and store the data in that
class DataIngestionConfig:

    yaml_config = read_yaml(
        "D:\Machine_Learning_Projects\Sensor_App_Fault_Detection\Config\config.yaml"
    )


# Actual Data ingestion class
class DataIngestion:

    # Constructor of data ingestion class to define the config class
    def __init__(self):

        self.data_ingestion_config = DataIngestionConfig()
        self.data_url = self.data_ingestion_config.yaml_config["data_ingestion"][
            "data_url"
        ]
        self.db_name = self.data_ingestion_config.yaml_config["data_ingestion"][
            "database_name"
        ]
        self.collection_name = self.data_ingestion_config.yaml_config["data_ingestion"][
            "collection_name"
        ]
        self.raw_data_path = self.data_ingestion_config.yaml_config["data_ingestion"][
            "raw_data_path"
        ]
        self.train_data_path = self.data_ingestion_config.yaml_config["data_ingestion"][
            "train_data_path"
        ]
        self.test_data_path = self.data_ingestion_config.yaml_config["data_ingestion"][
            "test_data_path"
        ]
        self.root_dir = self.data_ingestion_config.yaml_config["data_ingestion"][
            "root_dir"
        ]

    # Method to actually start data ingestion in artifact directory
    def initiate_data_ingestion(self):
        logging.info(
            "=============================== Data ingestion process started ============================"
        )

        try:
            data_dump_to_mongodb(self.data_url, self.db_name, self.collection_name)
            logging.info(
                f"Data is dumped into Mongodb with db name {self.db_name} and collection {self.collection_name}"
            )

            data = pd.read_csv(self.data_url)
            data = data.replace(to_replace="na", value=np.NAN)
            logging.info(
                f"The original data is read successfully with shape: {data.shape}"
            )

            os.makedirs(
                os.path.join(self.root_dir),
                exist_ok=True,
            )
            data.to_csv(
                self.raw_data_path,
                index=False,
            )
            logging.info(
                "The artifacts directory is created and raw data is saved in that directory"
            )

            train_data, test_data = train_test_split(data, test_size=0.3)
            train_data.to_csv(
                self.train_data_path,
                index=False,
            )
            test_data.to_csv(
                self.test_data_path,
                index=False,
            )

            logging.info(
                f"The train data with size: {train_data.shape} & test data with size: {test_data.shape} saved in artifacts directory"
            )
            logging.info(
                "========================== Data ingestion process completed ============================"
            )

            return (
                self.data_ingestion_config.yaml_config["data_ingestion"][
                    "train_data_path"
                ],
                self.data_ingestion_config.yaml_config["data_ingestion"][
                    "test_data_path"
                ],
            )

        except Exception as e:
            logging.info(f"Error occured with message: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    config = DataIngestion()
    config.initiate_data_ingestion()
