
# Promote Model from Staging to Production using MLflow Aliases

import os
import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv

def promote_model():

    # 1. Set up DagsHub credentials
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

    client = MlflowClient()

    model_name = "my_model2"

    # 3. Get model version using staging alias
    try:
        staging_model = client.get_model_version_by_alias(
            name=model_name,
            alias="staging"
        )

    except Exception as e:
        raise ValueError(
            f"No model found with 'staging' alias: {e}"
        ) from e

    staging_version = staging_model.version

    print(f"Staging Model Version: {staging_version}")

    # 4. Promote staging model to production
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=staging_version
    )

    print(
        f"Model Version {staging_version} "
        f"successfully promoted to Production (champion)"
    )

    # 5. Verify production alias
    production_model = client.get_model_version_by_alias(
        name=model_name,
        alias="champion"
    )

    print(
        f"Production Model Version: {production_model.version}"
    )


if __name__ == "__main__":
    promote_model()
