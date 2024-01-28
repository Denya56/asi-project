FROM python:3.10.7-bullseye 

COPY requirements.txt /build/
RUN pip install --no-cache-dir -r /build/requirements.txt

COPY package /app/package/
RUN pip install --no-cache-dir /app/package

COPY kedro /app/kedro/

WORKDIR /app/kedro/my-project/

ENTRYPOINT ["uvicorn"]
CMD ["--host", "0.0.0.0", "main:app"]