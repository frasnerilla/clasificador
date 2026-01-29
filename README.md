# Detector de SPAM - API REST (Flask)

Proyecto de clasificación (spam/no_spam) con Naive Bayes.
Se entrena un modelo y se despliega como API REST en Render.

## Archivos
- datos_spam.csv
- entrenar_modelo.py
- modelo_spam.pkl
- app.py
- requirements.txt

## Ejecutar en local
pip install -r requirements.txt
python entrenar_modelo.py
python app.py

## Endpoint
POST /predict
Body:
{"texto": "Gana dinero rápido haciendo clic aquí"}
