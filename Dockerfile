FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["gunicorn","server:app","--bind","0.0.0.0:7860","--timeout","120","--workers","3"]
