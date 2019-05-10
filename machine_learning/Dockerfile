# Compile
FROM jupyter/datascience-notebook
COPY requirements.txt .
COPY ./scripts /scripts
RUN pip install -r requirements.txt
USER root
# Make jupyter available from external sources
RUN sed -i "s/#c.NotebookApp.ip = 'localhost'/c.NotebookApp.ip = '0.0.0.0'/g" /home/jovyan/.jupyter/jupyter_notebook_config.py
RUN sed -i "s/#c.NotebookApp.open_browser = True/c.NotebookApp.open_browser = False/g" /home/jovyan/.jupyter/jupyter_notebook_config.py