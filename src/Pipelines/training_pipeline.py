from src.Components.data_ingestion import DataIngestion
from src.Components.data_preprocessing import DataPreprocessing
from src.Components.data_transformation import DataTransformation
from src.Components.model_trainer import ModelTrainer



data_ingest = DataIngestion()
data_ingest.initiate_data_ingestion()

data_preprocess = DataPreprocessing()
data_preprocess.initiate_data_transformation()

data_transform = DataTransformation()
data_transform.initiate_data_transformation()

model_trainer = ModelTrainer()
X, y, X_train, y_train, X_test, y_test = model_trainer.read_and_split_data()
best_params = model_trainer.fine_tune_model(X_train, y_train, X_test, y_test)
best_classifier = model_trainer.train_final_model(X, y, best_params)

print(best_classifier)