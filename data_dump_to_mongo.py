import pymongo, json, os
import pandas as pd

#Mongo DB connection string
client = pymongo.MongoClient("mongodb+srv://Nishchal30:Nishchal30@cluster0.9omin78.mongodb.net/")

data_url = "https://raw.githubusercontent.com/avnyadav/sensor-fault-detection/main/aps_failure_training_set1.csv"
data_save_location = "D:\Machine_Learning_Projects\Sensor_App_Fault_Detection\Data"

DB_name = "Sensor_Fault"
Collection_name = "Apps"

if __name__ == "__main__":

    data = pd.read_csv(data_url)
    data.to_csv(os.path.join(data_save_location, "sensor_data.csv"), index=False)

    print(f"Shape of data is {data.shape}")
    print(f"Data saved at {data_save_location}")

    json_data = list(json.loads(data.T.to_json()).values())
    # print(json_data[0])

    client[DB_name][Collection_name].insert_many(json_data)

    print("The data has been stored in MongoDB")