FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./requirements.txt /app


RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

EXPOSE 80

COPY ./app /app/app

