FROM python:3.12-slim

WORKDIR /app

# Copy Flask app requirements
COPY flask_app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Flask application while preserving folder structure
COPY flask_app/ /app/flask_app/

# Copy vectorizer model
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["gunicorn", "flask_app.app:app", "--bind", "0.0.0.0:5000"]
