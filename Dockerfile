FROM python:3.6.7
RUN apt-get update
RUN apt-get install libev-dev -y
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY /scripts/asset_processor /scripts
COPY /verifier /scripts