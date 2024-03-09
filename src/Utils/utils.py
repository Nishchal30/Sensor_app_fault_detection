import pymongo, json, os, sys
import pandas as pd
import yaml
from pathlib import Path

from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException


def read_yaml(yaml_path : Path):
    try:
        with open(yaml_path, "r+") as f:
            content = yaml.safe_load(f)
            logging.info("yaml file read successfully using read_yaml method in utils.py")
            return content
    
    except Exception as e:
        logging.info(f"Error occured with message: {e}")
        raise CustomException(e, sys)
    

def data_dump_to_mongodb(data_url : str, db_name : str, collection_name : str):
    
    try:
        client = pymongo.MongoClient("mongodb+srv://Nishchal30:Nishchal30@cluster0.9omin78.mongodb.net/")
        data = pd.read_csv(data_url)
        # data.to_csv(os.path.join(data_save_location, "sensor_data.csv"), index=False)

        # print(f"Shape of data is {data.shape}")
        # print(f"Data saved at {data_save_location}")

        json_data = list(json.loads(data.T.to_json()).values())
        # print(json_data[0])

        client[db_name][collection_name].insert_many(json_data)

        # logging.info(f"The data has been stored in MongoDB with db name {db_name} and collection {collection_name}")

    except Exception as e:
        logging.info(f"Error occured with message: {e}")
        raise CustomException(e, sys)


