FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow and other required Python packages
RUN pip3 install --no-cache-dir mlflow boto3 psycopg2-binary

WORKDIR /srv
# Expose the default MLflow port
EXPOSE 5000

# Command to run the MLflow server
CMD bash -lc "\
    mlflow server \
    --host 0.0.0.0 --port 5000 \
    --artifacts-destination ./mlartifacts \  
    --serve-artifacts \
"

# Note: For production, consider using a more robust database like PostgreSQL for the backend store
# and a cloud storage solution (like S3 or GCS) for artifact storage.
    