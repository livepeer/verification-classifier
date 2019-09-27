FROM python:3.7.4
COPY requirements.txt .
COPY credentials.json .
RUN pip install -r requirements.txt
RUN rm requirements.txt
ENV GOOGLE_APPLICATION_CREDENTIALS="/credentials.json"
COPY /imports /imports
COPY main.py .
ENTRYPOINT ["bash"]