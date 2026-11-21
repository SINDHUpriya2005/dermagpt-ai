FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod -R 777 /app/data
ENV PORT=7860
CMD gunicorn -b 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 app:app
