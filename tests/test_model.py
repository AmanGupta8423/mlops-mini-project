import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from dotenv import load_dotenv
import numpy as np
from mlflow.tracking import MlflowClient

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        load_dotenv()
        dagshub_token = os.getenv('DAGSHUB_PAT')
        if not dagshub_token:
            raise EnvironmentError('DAGSHUB_PAT environment variable is not set')

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "AmanGupta8423"
        repo_name = "mlops-mini-project"

        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
        cls.client = MlflowClient()
        cls.new_model_name = 'my_model2'
        cls.model_alias = 'champion'

        model_version = cls.client.get_model_version_by_alias(name = cls.new_model_name, alias = cls.model_alias)

        cls.new_model_version = model_version.version
        cls.new_model_uri = f'models:/{cls.new_model_name}@{cls.model_alias}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        cls.vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))

        cls.holdout_data = pd.read_csv('data/processed/test_tfidf.csv')
        cls.X_holdout = cls.holdout_data.iloc[:,:-1]
        cls.y_holdout = cls.holdout_data.iloc[:,-1]
        cls.X_holdout.columns = cls.X_holdout.columns.astype(str)

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model, 'Model failed to load from Mlfloew Registry')
        self.asssertGreater(int(self.new_model_version),0,'Invalid model version')

    def test_model_signature(self):
        signature = self.new_model.metadata.signature
        self.assertIsNotNone(signature, 'Model signature is missing')

        self.assertIsNotNone(signature.inputs, 'Model input signature is missing')
        self.assertIsNotNone(signature.output, 'Model output signature is missing')

        input_df = self.X_holdout.head(1)
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_features_names_out()),'Feature count does not match vectorizer')

        prediction = self.new_model.predict(input_df)

        self.assertEqual(len(prediction), 1, 'Unexpected prediction output size')
        self.assertEqual(np.asarray(prediction).ndim, 1, 'Expected a single dimensional prediction output')

    def test_model_performance(self):
        y_pred_new = self.new_model.predict(self.X_holdout)

        accuracy_new = accuracy_score(self.y_holdout,y_pred_new)
        precision_new = precision_score(self.y_holdout,y_pred_new)
        recall_new = recall_score(self.y_holdout, y_pred_new)
        f1_new = f1_score(self.y_holdout, y_pred_new, zero_division= 0)

        print("\n========== MODEL PERFORMANCE ==========")
        print(f"Model Version : {self.new_model_version}")
        print(f"Accuracy      : {accuracy_new:.4f}")
        print(f"Precision     : {precision_new:.4f}")
        print(f"Recall        : {recall_new:.4f}")
        print(f"F1 Score      : {f1_new:.4f}")
        print("========================================\n")

        expected_accuracy = 0.40
        expected_precision = 0.4
        expected_recall = 0.4
        expected_f1 = 0.4

        self.assertGreaterEqual(accuracy_new,expected_accuracy,f'Accuracy should be atleast {expected_accuracy}')
        self.assertGreaterEqual(precision_new,expected_precision,f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(recall_new,expected_recall,f'Recall should be atleast {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1,f'F1 score should be atleast {expected_f1}')

if __name__ == '__main__':
    unittest.main(verbosity=2)