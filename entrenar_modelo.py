# Importamos pandas para leer y trabajar con datos en forma de tabla (CSV)
import pandas as pd

# Importamos pickle para guardar el modelo entrenado en un archivo
import pickle

# Función para dividir los datos en conjunto de entrenamiento y de prueba
from sklearn.model_selection import train_test_split

# CountVectorizer convierte texto en números para que el modelo pueda trabajar con él
from sklearn.feature_extraction.text import CountVectorizer

# Modelo de clasificación Naive Bayes (muy usado para spam / no spam)
from sklearn.naive_bayes import MultinomialNB

# Métricas para evaluar modelos de clasificación
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1) Cargar datos
# Leemos el archivo CSV que contiene los correos y su etiqueta (spam o no_spam)
# Usamos sep=";" porque muchos Excel guardan los CSV con punto y coma
#df = pd.read_csv("datos_spam.csv", sep=";", encoding="latin-1")
#df = pd.read_csv("datos_spam.csv", sep=";", encoding="utf-8")
df = pd.read_csv("datos_spam.csv", sep=";")

# 2) Separar X e y
# X: datos de entrada → el texto del correo
X = df["texto"]

# y: datos de salida → la etiqueta asociada a cada texto
# Indica si el correo es spam o no_spam
y = df["etiqueta"]

# 3) Dividir train/test
# Separamos los datos en:
# - 75% para entrenar el modelo
# - 25% para probar si el modelo funciona bien
# random_state permite que la división sea siempre la misma
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4) Vectorización: convertir texto en números
# Los modelos de IA no entienden texto directamente,
# por eso usamos un vectorizador para convertir palabras en números
vectorizador = CountVectorizer()

# Ajustamos el vectorizador con los textos de entrenamiento
# y los transformamos a formato numérico
X_train_vec = vectorizador.fit_transform(X_train)

# Transformamos los textos de prueba usando el mismo vectorizador
X_test_vec = vectorizador.transform(X_test)

# 5) Entrenar modelo (Naive Bayes)
# Creamos el modelo de clasificación
modelo = MultinomialNB()

# Entrenamos el modelo con los textos ya convertidos a números
modelo.fit(X_train_vec, y_train)

# 6) Evaluación del modelo
# Usamos el modelo entrenado para predecir la clase de los textos de prueba
preds = modelo.predict(X_test_vec)

# Accuracy: porcentaje de aciertos del modelo
print("ACCURACY:", round(accuracy_score(y_test, preds), 2))

# Matriz de confusión: muestra aciertos y errores por clase
print("\nMATRIZ DE CONFUSION:")
print(confusion_matrix(y_test, preds, labels=["spam", "no_spam"]))

# Informe completo: precision, recall y f1-score
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, preds))

# 7) Guardar modelo + vectorizador
# Guardamos juntos el vectorizador y el modelo entrenado
# para poder reutilizarlos después en la API (app.py)
with open("modelo_spam.pkl", "wb") as f:
    pickle.dump((vectorizador, modelo), f)

print("\nModelo guardado como modelo_spam.pkl")
