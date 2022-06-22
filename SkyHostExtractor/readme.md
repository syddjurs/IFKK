# FleetComplete - & SkyHost API Extractor
### Indledning
Formålet med de to moduler "FleetCompleteExtractor" og "SkyHostExtractor" er at automatisere træk af data direkte
fra flådesystemer. Systemerne gemmer logs for enkelte ture. Dvs. for hver gang bilen har været tændt og slukket,
vil der optræde et datapunkt - altså fra A til B. Applikationen bruger rundture, ture der går fra A til A. Årsagen er
bl.a. en bil kan ikke allokeres til en ny tur i delflåden, hvis ikke bilen er hjemme igen. Det giver bl.a. også den fordel,
at der for applikationen kun behøver at være adgang til `RoundTrips` og ikke `Trips`, som kan indeholde personfølsom data.
`RoundTrips` indeholder kun logninger, der for kunden er en kendt og gemt lokation, f.eks. en parkeringsplads. Disse to moduler foretager denne aggregation 
fra logninger til egentlige rundture.

Jobs kan sættes op til, at trække hver nat eller på ugebasis for at føde nyt data til applikationen.
I projektet har der været CronJobs sat op i K8s, som igangsættes hver nat. De sættes i gang om natten - både for at spare
resurser på applikationens cluster og for ikke at belaste flådestyringssystemerne i peak-time. <br><br>
Der bruges en recursive metode for at komme så tæt på de virkelige rundture, derfor kan det godt være en længerevarende
process at aggregerer - specielt ved første træk. <br><br>
**Kriterier for en rundtur**:
- En rundtur består af minimum 2 logninger
- En rundtur starter maximum 0,2 km. fra et tilladt startsted
- En rundtur skal minimum have en tilbagelagt distance på 0,5 km. 
- En rundtur skal stoppe maximum 0,2 km. fra det startsted den har udgangspunkt.
- En rundtur forsøges splittes op, hvis den ikke er afsluttet efter 16,9 timer. 

### SkyHost
SkyHosts API er bygget på SOAP og begrænser antal request via et point system. Derfor er der indbygget 
bookkeeping og sleep i API wrapperen. Første træk vil tage betydeligt længere tid, end de efterfølgende, som kører hver
nat. Det er på køretøjer (`updatedb.set_trackers`) kun muligt at trække et ID og/eller registreringsnummer, hvilket for applikationen ikke er
tilstrækkelig information til at køre en meningsfuld simulering. Derfor er det nødvendigt at berige det hentede data med
minimum følgende værdier defineret i `src.fleetmanager.data_access.dbschema.Cars`: **plate, make, model, type, fuel,
wltp_fossil/wltp_el, omkostning_aar, location**. <br><br>
Udover den begrænset data på køretøjer, der udstilles via API'et, er det ikke muligt at trække startsteder eller
hjemmelokationer for køretøjerne. Dette er betydningsfuld, da applikationen forventer, at kunne associere et køretøj til
en specifik delflåde. Derfor skal man manuelt tilføje disse, hvis man bruger SkyHost flådestyringssystem. Fordi der ikke
er link mellem `Cars` og `AllowedStarts`, skal dette også tilføjes manuelt **FØR** jobbet køres. <br><br>
I `updatedb` kan der loades et set "default" `AllowedStarts` og `Cars`. Der er fra kørebogen
kun tilføjet GPS koordinator på logninger fra slut februar 2022. <br><br>

Byg docker image fra rod `intelligentfleetmanagement`<br>

```
docker build -f SkyHostExtractor/Dockerfile . -t <tagname>
```

Kør med følgende environment variabler. `SOAP_KEY` leveres af SkyHost
```
docker run -e DB_NAME=<db_name> -e DB_USER=<db_user> -e DB_PASSWORD=<db_password> -e DB_URL=<db_url> -e DB_SERVER=mysql -e SOAP_KEY=<soap_key>
```

Eller defineres via `cron.yaml`
```
kubectl apply -f SkyHostExtractor/cron.yaml 
```

### FleetComplete
FleetComplete udstiller en almindelig REST api, som tilgås via en nøgle som udstedes af FleetComplete. Denne API kan
håndtere en forholdsvis stor mængde request. Fejler requesten, ventes der i 3 sekunder, hvorefter der forsøge igen indtil 
det maksimale antal request tilladt er nået. FleetComplete udstiller `Places`, som er ækvivalent til `AllowedStarts`.
Hver køretøj er associeret til en af disse. Metadata på køretøjer i FleetComplete inkluderer, id, nummerplade, make, model,
type, fuel, location. Såfremt det er vedligeholdt i FleetComplete er det kun nødvendigt at berige bildata med minimum følgende værdier defineret i 
`src.fleetmanager.data_access.dbschema.Cars`: **wltp_fossil/wltp_el, omkostning_aar**. <br><br>


Byg docker image fra rod `intelligentfleetmanagement`<br>

```
docker build -f FleetCompleteExtractor/Dockerfile . -t <tagname>
```

Kør med følgende environment variabler. `API_KEY` leveres af FleetComplete
```
docker run -e DB_NAME=<db_name> -e DB_USER=<db_user> -e DB_PASSWORD=<db_password> -e DB_URL=<db_url> -e DB_SERVER=mysql -e API_KEY=<soap_key>
```

Eller defineres via `cron.yaml`
```
kubectl apply -f SkyHostExtractor/cron.yaml 
```