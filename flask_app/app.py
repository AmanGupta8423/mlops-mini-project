
from flask import Flask, render_template, request
import mlflow
from flask_app.preprocessing_utility import normalize_text
import os
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

dagshub_token = os.getenv("DAGSHUB_PAT")

if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

# DagsHub authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# MLflow tracking URI
dagshub_url = "https://dagshub.com"
repo_owner = "AmanGupta8423"
repo_name = "mlops-mini-project"

mlflow.set_tracking_uri(
    f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
)

app = Flask(__name__)


# Get latest model version
def get_latest_model_version(model_name):

    client = mlflow.MlflowClient()

    try:
        # First preference: Staging alias
        model_version = client.get_model_version_by_alias(
            name=model_name,
            alias="staging"
        )

        return model_version.version

    except Exception as e:

        print(f"Staging alias not found: {e}")

        # Fallback: Latest registered version
        versions = client.search_model_versions(
            filter_string=f"name='{model_name}'"
        )

        if not versions:
            return None

        latest_version = max(
            versions,
            key=lambda x: int(x.version)
        )

        return latest_version.version


# Load registered model
model_name = "my_model2"

model_version = get_latest_model_version(model_name)

if model_version is None:
    raise Exception(f"No registered model found: {model_name}")

model_uri = f"models:/{model_name}/{model_version}"

model = mlflow.pyfunc.load_model(model_uri)

print(f"Loaded Model: {model_name}")
print(f"Model Version: {model_version}")


# Load TF-IDF vectorizer
with open("models/vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)


# Home page
@app.route('/')
def home():

    return render_template(
        'index.html',
        model_version=model_version
    )


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    text = request.form.get('text', '').strip()

    # Empty input validation
    if not text:

        return render_template(
            'index.html',
            error="Please enter some text before predicting.",
            model_version=model_version
        )

    # Text preprocessing
    cleaned_text = normalize_text(text)

    # TF-IDF transformation
    features = vec.transform([cleaned_text])

    # Model prediction
    result = model.predict(features)

    prediction = int(result[0])

    # Map numerical prediction to sentiment
    if prediction == 0:
        sentiment = "Sadness"
        emoji = "😢"
        css_class = "sadness"

    elif prediction == 1:
        sentiment = "Happiness"
        emoji = "😊"
        css_class = "happiness"

    else:
        sentiment = "Unknown"
        emoji = "❓"
        css_class = "unknown"

    # Display result
    return render_template(
        'index.html',
        original_text=text,
        cleaned_text=cleaned_text,
        prediction=prediction,
        sentiment=sentiment,
        emoji=emoji,
        css_class=css_class,
        model_version=model_version
    )


if __name__ == '__main__':
    app.run(debug=True)
