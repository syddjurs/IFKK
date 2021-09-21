# Indledning
Dette repositorie indeholder simuleringstoolet som er udviklet i løbet af fase 1 af AI-signaturprojektet Intelligent flådestyring og klimasmarte kørselsmønstre (https://ifkk.syddjurs.dk/forside/). 
Dokumentationen er delt i to dele, en teknisk del rettet imod udviklere og leverandører og en anvendelsesorienteret rettet imod slutbrugere. Den tekniske del findes i det Github-repositorie som er tilknyttet projektet, https://github.com/syddjurs/IFKK/blob/master/dashboard/documentation/build/html/index.html. Den anvendelsesorienterede dokumentation findes i https://github.com/syddjurs/IFKK/blob/master/dashboard/documentation/simuleringstool_dokumentation_DA.docx.
Simuleringstoolet er udviklet i fase 1 er ikke en færdig løsning, men en funktionel prototype af den endelige løsning. Den funktionelle prototype vil kunne anvendes som udgangspunkt for den videre udvikling af værktøjet i fase 2.
# Formål
Simuleringstoolet er udviklet med formål om at kunne tage en proaktiv tilgang til indkøb af køretøjsflåden i en kommune.
I dag foretages indkøb af køretøjer med begrænset input fra data. Målet er at kunne anvende kommunens egne data i form af kørte ture og køretøjsflåde til at simulere scenarier med forskellig sammensætning af køretøjsflåden.
Simuleringstoolet vil kunne sammenligne den nuværende flåde med en fremtidig simuleret og give informationer om kapaciteten i flåden og om forskelle i økonomiske, transportmæssige og udledningsmæssige-konsekvenser imellem de to flåder. 
# Teknisk overblik
Simuleringstoolet er udviklet i python på backenden og brugergrænsefladen er ligeledes opsat med python-biblioteket Bokeh. Python og Bokeh er valgt da man hurtigt har skulle udvikle en funktionel prototype.
Dokumentationen på de enkelte dele af koden er at finde i det tilhørende Github-projekt, https://github.com/syddjurs/IFKK.
# Sådan kommer du i gang
## Forudsætninger
Applikationen er testet på Windows. Applikationen er afhængig af biblioteket ”xlwings” som ikke findes til Linux. Applikationen burde kunne eksekveres på OS X (Mac), men dette er ikke testet.
Applikationen er testet med Python 3.8.5. Den installerede version af Python kan checkes ved at køre følgende kommando i kommandolinjen i Windows:
```cmd
C:\Users\user>python -V
Python 3.8.5
```
De nødvendige biblioteker (bokeh, numpy, pandas, xdg og sqlalchemy) som anvendes er testet med følgende versioner:
```cmd
C:\Users\user> python -c "import bokeh; print(bokeh.__version__)"
2.2.3
C:\Users\user> python -c "import numpy; print(numpy.__version__)"
1.19.2
C:\Users\user> python -c "import pandas; print(pandas.__version__)"
1.1.3
C:\Users\user> python -c "import xdg; print(xdg.__version__)"
0.27
C:\Users\user> python -c "import sqlalchemy; print(sqlalchemy.__version__)"
1.3.20
```
Det er nødvendigt at have en fil med baggrundsdata i /sourcefiles/options.xlsx. Der er et eksempel vedlagt kildekoden men det skal tilpasses den enkelte kommune, se eventuelt afsnit om baggrundsdata.
## Opstart af applikation
Opstart af applikationen sker ved at køre nedenstående kommando i rodmappen (den mappe der indeholder mappen dashboard/). Applikationen giver et link til localhost hvor det grafiske interface vises, i nedenstående eksempel er dette http://localhost:5006/dashboard. Åbnes linket i browseren kan det grafiske interface tilgås.
```cmd
C:\Users\user\rodmappe>bokeh serve dashboard
2021-09-13 13:07:01,393 Starting Bokeh server version 2.2.3 (running on Tornado 6.0.4)
2021-09-13 13:07:01,396 User authentication hooks NOT provided (default user enabled)
2021-09-13 13:07:01,399 Bokeh app running at: http://localhost:5006/dashboard
2021-09-13 13:07:01,400 Starting Bokeh server with process id: 19768
```
