FROM python:3.13-slim

COPY ./Assignment-4/requirements.txt .
COPY Assignment-1 /Assignment-1/
COPY Assignment-2 /Assignment-2/
COPY Assignment-3 /Assignment-3/

RUN pip install -r requirements.txt

WORKDIR /Assignment-3

CMD ["python", "app.py"]
