# Compile
FROM python:3
COPY /scripts /src
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY /cli /src
CMD /bin/bash
