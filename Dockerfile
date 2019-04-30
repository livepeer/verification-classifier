# Compile
FROM python:3
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY /scripts /scripts
COPY /cli /scripts
CMD /bin/bash
