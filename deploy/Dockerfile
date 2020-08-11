FROM python:3.7-slim-stretch
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt
COPY . /app
COPY lgbm_20200720.pkl.z /app
RUN mkdir /root/.streamlit && \
    mv credentials.toml /root/.streamlit && \
    apt-get update && apt-get install libgomp1

CMD sh setup.sh && streamlit run recommender.py