# Internel packages
from src.logger import logging
from src.exception import CustomException
from src.Utils.utils import read_yaml, save_object

# Third-party libraries
import pandas as pd
import sys, os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek


class DataTransformationConfig:

    yaml_config = read_yaml(
        "D:\Machine_Learning_Projects\Sensor_App_Fault_Detection\Config\config.yaml"
    )


class DataTransformation:

    def __init__(self):
        logging.info(
            "================= Data transformation starts here ======================="
        )
        try:
            self.data_transformation_config = DataTransformationConfig()
            self.transformation_dir = self.data_transformation_config.yaml_config[
                "data_transformation"
            ]["transformation_dir"]
            self.train_data = self.data_transformation_config.yaml_config[
                "data_ingestion"
            ]["train_data_path"]
            self.test_data = self.data_transformation_config.yaml_config[
                "data_ingestion"
            ]["test_data_path"]
            self.preprocessor_path = self.data_transformation_config.yaml_config[
                "data_transformation"
            ]["preprocessor_file"]

        except Exception as e:
            logging.info(f"Error occured with message: {e}")
            raise CustomException(e, sys)

    def get_data_transformer_pipeline(self):

        try:
            logging.info("Pipeline started here")

            feature_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ("Scaler", StandardScaler()),
                ]
            )

            target_pipeline = Pipeline(steps=[("lable_encoder", LabelEncoder())])

            return feature_pipeline, target_pipeline

        except Exception as e:
            logging.info(f"Error occured with message: {e}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self):

        try:
            train_data = pd.read_csv(self.train_data)
            test_data = pd.read_csv(self.test_data)

            target_column = "class"
            train_feature_data = train_data.drop(target_column, axis=1)
            test_feature_data = test_data.drop(target_column, axis=1)

            logging.info(
                f"The train & test fetures data size: {train_feature_data.shape, test_feature_data.shape}"
            )

            train_target_data = train_data[target_column]
            test_target_data = test_data[target_column]

            logging.info(
                f"The train & test target data size: {train_target_data.shape, test_target_data.shape}"
            )

            feature_pipline_obj, target_pipeline_obj = (
                DataTransformation.get_data_transformer_pipeline(self)
            )
            train_target_transformed_data = pd.DataFrame(
                target_pipeline_obj.named_steps["lable_encoder"].fit_transform(
                    train_target_data
                ), columns = ["class"]
            )
            test_target_tarnsformed_data = pd.DataFrame(
                target_pipeline_obj.named_steps["lable_encoder"].transform(
                    test_target_data
                ), columns = ["class"]
            )

            train_feature_transformed_data = pd.DataFrame(
                feature_pipline_obj.fit_transform(train_feature_data),
                columns=feature_pipline_obj.get_feature_names_out(
                    train_feature_data.columns
                ),
            )

            test_feature_transformed_data = pd.DataFrame(
                feature_pipline_obj.transform(test_feature_data),
                columns=feature_pipline_obj.get_feature_names_out(
                    test_feature_data.columns
                ),
            )

            logging.info(f"Before resampling {'-'*100}")
            logging.info(
                f"The target feature is encoded with shape: {train_target_transformed_data.shape, test_target_tarnsformed_data.shape}"
            )
            logging.info(
                f"The train & test feature data size after applying transformation: {train_feature_transformed_data.shape, test_feature_transformed_data.shape}"
            )

            smt = SMOTETomek(sampling_strategy="minority")
            train_feature_transformed_data, train_target_transformed_data = (
                smt.fit_resample(
                    train_feature_transformed_data, train_target_transformed_data
                )
            )
            test_feature_transformed_data, test_target_tarnsformed_data = (
                smt.fit_resample(
                    test_feature_transformed_data, test_target_tarnsformed_data
                )
            )

            logging.info(f"After resampling f{'-'*100}")
            logging.info(
                f"The target feature is encoded with shape: {train_target_transformed_data.shape, test_target_tarnsformed_data.shape}"
            )
            logging.info(
                f"The train & test feature data size after applying transformation: {train_feature_transformed_data.shape, test_feature_transformed_data.shape}"
            )

            final_train_data = pd.concat(
                [train_feature_transformed_data, train_target_transformed_data], axis=1
            )
            final_test_data = pd.concat(
                [test_feature_transformed_data, test_target_tarnsformed_data], axis=1
            )

            
            os.makedirs(self.transformation_dir, exist_ok=True)
            final_train_data.to_csv(
                os.path.join(self.transformation_dir, "resampled_train_data.csv"),
                index=False,
            )
            final_test_data.to_csv(
                os.path.join(self.transformation_dir, "resampled_test_data.csv"),
                index=False,
            )

            logging.info("Resampled data is saved in .csv files")

            save_object(
                filepath=os.path.join(self.transformation_dir, "feature_pipeline.pkl"),
                object=feature_pipline_obj,
            )
            save_object(
                filepath=os.path.join(self.transformation_dir, "target_pipeline.pkl"),
                object=target_pipeline_obj,
            )

            logging.info("pipeline objects are saved successfully")

            return final_train_data, final_test_data

        except Exception as e:
            logging.info(f"Error occured with message: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":

    config = DataTransformation()
    config.initiate_data_transformation()
