# Importamos el módulo os para trabajar con variables del sistema (puertos, rutas, etc.)
import os

# Importamos pickle para cargar el modelo entrenado desde un archivo .pkl
import pickle

# Importamos Flask y herramientas para recibir datos y devolver respuestas en JSON
from flask import Flask, request, jsonify


# Creamos la aplicación Flask
# __name__ indica que este archivo es el principal de la aplicación
app = Flask(__name__)

# -------------------------------
# CARGA DEL MODELO DE IA
# -------------------------------
# Abrimos el archivo modelo_spam.pkl en modo lectura binaria ("rb")
# Dentro del archivo están guardados:
# - el vectorizador (para convertir texto en números)
# - el modelo entrenado (Naive Bayes)
with open("modelo_spam.pkl", "rb") as f:
    vectorizador, modelo = pickle.load(f)

# -------------------------------
# ENDPOINT PRINCIPAL
# -------------------------------
# Este endpoint responde a peticiones GET en la ruta "/"
# Sirve para mostrar información básica sobre la API
@app.route("/", methods=["GET"])
def inicio():
    return jsonify({
        "mensaje": "API REST - Detector de SPAM",
        "endpoint_prediccion": "POST /predict",
        "ejemplo_json": {"texto": "Gana dinero rapido, entra aqui"}
    })

# -------------------------------
# ENDPOINT DE COMPROBACIÓN
# -------------------------------
# Este endpoint sirve para comprobar si la API está funcionando
# Devuelve un JSON sencillo con el estado del servicio
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# -------------------------------
# ENDPOINT DE PREDICCIÓN
# -------------------------------
# Este endpoint recibe una petición POST con un JSON
# y devuelve si el texto es spam o no_spam
@app.route("/predict", methods=["POST"])
def predict():

    # Leemos el contenido JSON enviado por el cliente
    datos = request.get_json()

    # Comprobamos que el JSON existe y contiene el campo "texto"
    if datos is None or "texto" not in datos:
        # Si no es correcto, devolvemos un error 400 (petición incorrecta)
        return jsonify({
            "error": "Debes enviar un JSON con el campo 'texto'."
        }), 400

    # Extraemos el texto del JSON y lo convertimos a string
    texto = str(datos["texto"])

    # Convertimos el texto a formato numérico usando el vectorizador entrenado
    # El modelo solo entiende números, no texto
    texto_vec = vectorizador.transform([texto])

    # Usamos el modelo para predecir la clase del texto
    # [0] porque devuelve una lista con una sola predicción
    pred = modelo.predict(texto_vec)[0]

    # Devolvemos la respuesta en formato JSON
    return jsonify({
        "texto": texto,
        "clase": pred,
        "es_spam": (pred == "spam")
    })

# -------------------------------
# EJECUCIÓN DE LA APLICACIÓN
# -------------------------------
# Este bloque solo se ejecuta si el archivo se lanza directamente
if __name__ == "__main__":

    # Render usa la variable de entorno PORT
    # Si no existe (ejecución en local), usamos el puerto 5000
    port = int(os.environ.get("PORT", 5000))

    # Iniciamos el servidor Flask
    # host="0.0.0.0" permite que la app sea accesible desde fuera
    app.run(host="0.0.0.0", port=port)
