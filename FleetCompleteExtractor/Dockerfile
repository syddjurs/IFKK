# build from root intelligentfleetmanagement
FROM python:3.10.2-bullseye
WORKDIR /app
COPY FleetCompleteExtractor/requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt
# get necessary https driver for apt
# RUN apt-get update
# RUN apt-get install -y apt-transport-https curl

COPY src/fleetmanager/data_access/ FleetCompleteExtractor/data_access/
COPY src/fleetmanager/model/roundtripgenerator.py FleetCompleteExtractor/roundtripgenerator.py
COPY src/fleetmanager/model/__init__.py FleetCompleteExtractor/__init__.py
COPY FleetCompleteExtractor/main.py FleetCompleteExtractor/main.py

WORKDIR FleetCompleteExtractor
CMD ["python3", "-u", "main.py"]