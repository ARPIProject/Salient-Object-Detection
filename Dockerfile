FROM python:3.8.3-buster

EXPOSE 8501

RUN pip install pipenv
COPY Pipfile* ./
RUN pipenv install --deploy

COPY model /model
WORKDIR /model

ENTRYPOINT ["pipenv", "run", "streamlit", "run", "run.py"]
