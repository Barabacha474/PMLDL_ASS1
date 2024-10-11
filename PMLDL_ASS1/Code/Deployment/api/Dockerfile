FROM python:3.9

WORKDIR /api

RUN pip install --upgrade pip
RUN pip install --no-cache-dir Pillow tqdm matplotlib pybase64
RUN pip install --no-cache-dir fastapi uvicorn pydantic torch torchvision

COPY . .

#RUN py main.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]