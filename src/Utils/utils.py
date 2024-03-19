import pymongo, json, os, sys
import pandas as pd
import yaml
from pathlib import Path
import pickle

from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from dotenv import load_dotenv


load_dotenv()


def read_yaml(yaml_path : Path) -> str:
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
        mongo_client = os.environ.get("mongo_conn")
        client = pymongo.MongoClient(mongo_client)
        data = pd.read_csv(data_url)
        # data.to_csv(os.path.join(data_save_location, "sensor_data.csv"), index=False)

        # print(f"Shape of data is {data.shape}")
        # print(f"Data saved at {data_save_location}")

        json_data = list(json.loads(data.T.to_json()).values())
        # print(json_data[0])
        
        if (db_name not in client.list_database_names()):
            client[db_name][collection_name].insert_many(json_data)
        else:
            logging.info(f"Database already exists in Mongo DB with name {db_name}")

        # logging.info(f"The data has been stored in MongoDB with db name {db_name} and collection {collection_name}")

    except Exception as e:
        logging.info(f"Error occured with message: {e}")
        raise CustomException(e, sys)
    


def convert_columns_to_float(df:pd.DataFrame, exclude_column:list) -> pd.DataFrame:
    try:

        for column in df.columns:
            if column not in exclude_column:
                df[column] = df[column].astype("float")
            
        return df

    except Exception as e:
        logging.info(f"Error occured with message: {e}")
        raise CustomException(e, sys)
    

def write_yaml_file(file_path : Path, data:dict):
    try:

        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,"w") as file:
            yaml.dump(data,file)

    except Exception as e:
        logging.info(f"The error occured in write_yaml method in utils.py as: {e}")
        raise CustomException(e, sys)


def save_object(filepath : Path, object):
    try:
        filedir = os.path.dirname(filepath)
        os.makedirs(filedir, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(object, f)
    
    except Exception as e:
        logging.info(f"The error occured in save_object method in utils.py as: {e}")
        raise CustomException(e, sys)

