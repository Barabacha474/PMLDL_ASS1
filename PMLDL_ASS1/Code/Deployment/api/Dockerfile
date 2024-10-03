FROM python:3.10-slim

WORKDIR Code/Deployment/api

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--reload"]