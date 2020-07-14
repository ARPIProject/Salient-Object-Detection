FROM python:3.8.3-buster

RUN mkdir /output && mkdir /input
VOLUME /input
VOLUME /output

RUN pip install pipenv
COPY Pipfile* ./
RUN pipenv install --deploy

COPY inference ./inference

ENTRYPOINT ["pipenv", "run", "python", "./inference/inference.py"]
