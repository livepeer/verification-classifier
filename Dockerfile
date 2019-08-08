# Compile
FROM python:3.6.7
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY /scripts/asset_processor /scripts
COPY /verifier /scripts