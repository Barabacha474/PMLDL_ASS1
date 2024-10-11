FROM python:3.9-slim-buster

WORKDIR Code/Deployment/app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY Code/Deployment/app .

ENTRYPOINT ["streamlit", "run", "main.py"]