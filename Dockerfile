# Compile
FROM python:3
COPY /cli /src
COPY /scripts /src
COPY requirements.txt .
RUN pip install -r requirements.txt