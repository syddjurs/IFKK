# build from root context
FROM python:3.10.2-bullseye

WORKDIR /app

COPY SkyHostExtractor/requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

COPY src/fleetmanager/data_access/ SkyHostExtractor/data_access/
COPY src/fleetmanager/model/roundtripgenerator.py SkyHostExtractor/roundtripgenerator.py
COPY src/fleetmanager/model/__init__.py SkyHostExtractor/__init__.py
COPY SkyHostExtractor/xml_templates SkyHostExtractor/xml_templates
COPY SkyHostExtractor/soap_agent.py SkyHostExtractor/soap_agent.py
COPY SkyHostExtractor/parsers.py SkyHostExtractor/parsers.py
COPY SkyHostExtractor/updatedb.py SkyHostExtractor/updatedb.py

WORKDIR SkyHostExtractor
ENTRYPOINT ["/bin/sh", "-c", "./updatedb.py set-trackers && ./updatedb.py set-trips && ./updatedb.py set-roundtrips"]
