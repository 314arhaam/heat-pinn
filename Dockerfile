FROM python:3.12-slim

WORKDIR /opt/heat-pinn

COPY . .

RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python3", "heat-pinn-cli/heat.py"]