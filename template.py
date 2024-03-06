import os
from pathlib import Path

list_of_files = [
    "requirements.txt",
    "README.md",
    "setup.py",
    "notebooks/research.ipynb",
    "Data/.gitkeep",
    "src/__init__.py",
    "src/logger.py",
    "src/exception.py",
    "src/Components/__init__.py",
    "src/Components/data_ingestion.py",
    "src/Components/data_transformation.py",
    "src/Components/model_trainer.py",
    "src/Pipelines/__init__.py",
    "src/Pipelines/training_pipeline.py",
    "src/Pipelines/prediction_pipeline.py",
    "src/Utils/__init__.py",
    "src/Utils/utils.py",
]


for file in list_of_files:
    filepath = Path(file)
    filedir, filename = os.path.split(filepath)


    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    
    if not(os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    
    else:
        print("file already exists")
