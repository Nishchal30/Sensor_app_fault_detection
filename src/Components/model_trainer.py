from src.logger import logging
from src.exception import CustomException
from src.Utils.utils import read_yaml, evaluate_model
import sys
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
import optuna
from sklearn.metrics import accuracy_score


class ModelTrainerConfig:

    yaml_config = read_yaml(
        "D:\Machine_Learning_Projects\Sensor_App_Fault_Detection\Config\config.yaml"
    )


class ModelTrainer:

    def __init__(self):

        try:

            self.config = ModelTrainerConfig()
            self.train_data = self.config.yaml_config["model_trainer"][
                "preprocessed_train_data"
            ]
            self.test_data = self.config.yaml_config["model_trainer"][
                "preprocessed_test_data"
            ]

        except Exception as e:
            logging.error(f"The yaml file not found: {e}")
            raise FileNotFoundError(f"Yaml file not found")

    def read_and_split_data(self):

        try:
            logging.info(f"{'-'*30}Model training will start here {'-'*30}")
            train_data = pd.read_csv(self.train_data)
            test_data = pd.read_csv(self.test_data)
            logging.info("Train and test data read successfully")

            X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
            X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

            logging.info("train & test split completed")

            return train_data, test_data, X_train, y_train, X_test, y_test

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):

        try:
            logging.info("model training started without tuning")
            models = {
                "LogisticRegression": LogisticRegression(),
                "RandomForestClassifier": RandomForestClassifier(),
                "XGBClassifier": XGBClassifier(),
            }

            model_report: dict = evaluate_model(
                X_train, y_train, X_test, y_test, models
            )

            logging.info(f"The accuracy report for model trainig: \n{model_report}")

            return model_report

        except Exception as e:
            logging.error(f"Error occured at initiate_model_trainer method with: {e}")
            raise CustomException(e, sys)

    def objective(self, trial, X_train, y_train, X_test, y_test):

        try:
            logging.info("---------The fine tuning model with optuna method--------")
            n_estimators = trial.suggest_int("n_estimators", 10, 100)
            max_depth = trial.suggest_int("max_depth", 2, 32, log=True)

            classifier = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )

            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"The accuracy of the {classifier} is {accuracy}")

            return accuracy

        except Exception as e:
            logging.error(f"Error occured at objective method with: {e}")
            raise CustomException(e, sys)

    def fine_tune_model(self, X_train, X_test, y_train, y_test):

        try:
            logging.info("The optuna study will create here")
            study = optuna.create_study(direction="maximize")  # maximize accuracy
            study.optimize(
                lambda trial: self.objective(trial, X_train, X_test, y_train, y_test),
                n_trials=100,
            )  
            best_study = study.best_trial
            logging.info(f"The best study with parameters is: {best_study.value, best_study.params}")
            return study.best_params

        except Exception as e:
            logging.error(f"Error occured at fine_tune_model method with: {e}")
            raise CustomException(e, sys)

    def train_final_model(self, X, y, best_params):

        try:

            logging.info(f"Train the model finally with the tunned parameters")
            best_classifier = RandomForestClassifier(**best_params, random_state=42)
            best_classifier.fit(X, y) 
            logging.info(f"The training of {best_classifier} is done with {best_params}")
            return best_classifier

        except Exception as e:
            logging.error(f"Error occured at train_final_model method with: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":

    config = ModelTrainer()
    X, y, X_train, y_train, X_test, y_test = config.read_and_split_data()
    best_params = config.fine_tune_model(X_train, y_train, X_test, y_test)
    best_classifier = config.train_final_model(X, y, best_params)
    print(best_classifier)
