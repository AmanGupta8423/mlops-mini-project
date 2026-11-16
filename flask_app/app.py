from flask import Flask, render_template, request
import mlflow
from preprocessing_utility import normalize_text
import dagshub
import os
from dotenv import load_dotenv
import pickle

load_dotenv()
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "AmanGupta8423"
repo_name = "mlops-mini-project"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


app = Flask(__name__)

def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    try:
        # hona production chahiye allias but thoda error aa rha tha in update
        model_version = client.get_model_version_by_alias( model_name = model_name, alias="staging")
        return model_version.version
    except Exception:
        versions = client.search_model_versions(filter_string=f"name='{model_name}'")
        return max(versions, key=lambda x: int(x.version)).version if versions else None

#load model
model_name = 'my_model2'
model_version = get_latest_model_version(model_name)
model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

vec = pickle.load(open('models/vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods = ['POST'])
def predict():
    text = request.form['text']

    #clean
    text = normalize_text(text)
    #tfidf
    features = vec.transform([text])
    #predict
    result= model.predict(features)
    #show
    return str(result[0])

app.run(debug=True)