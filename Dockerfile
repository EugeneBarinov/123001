FROM pytorch/pytorch:2.2.0-cpu
WORKDIR /app
ENV PYTHONPATH=/app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000 8501
CMD ["uvicorn", "AI.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
