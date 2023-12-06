FROM python:3.11.5

COPY . /

WORKDIR /

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "src/main.py"]
