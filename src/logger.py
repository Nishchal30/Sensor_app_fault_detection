import os, logging
from pathlib import Path
from datetime import datetime as dt

log_file = f"{dt.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"

log_dir_path = "D:\Machine_Learning_Projects\Sensor_App_Fault_Detection\logs"

logfile_path = os.path.join(log_dir_path, log_file)

os.makedirs(log_dir_path, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=logfile_path, 
    format="[%(asctime)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s)]"
)