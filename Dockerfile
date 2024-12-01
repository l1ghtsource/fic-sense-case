FROM python:3.10-slim
WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000
EXPOSE 8501

CMD uvicorn service.backend:app --host 0.0.0.0 --port 8000 & streamlit run service/frontend.py