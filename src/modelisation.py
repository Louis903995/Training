import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import json

print("Répertoire courant :", os.getcwd())
print("Fichiers dans ce dossier :", os.listdir())

# 1. Chargement des données CSV
data = pd.read_csv("data/tickets_categorie_final.csv", sep=';')

# 2. Pré-traitement (optionnel mais recommandé)
data['Produit'] = data['Produit'].str.lower()  # mettre en minuscule
data['Produit'] = data['Produit'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')  # enlever les accents si nécessaire

# 3. TF-Idata Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Produit'])  # transforme en vecteur TF-Idata
# Sauvegarder le vocabulaire et les valeurs IDF dans un fichier JSON
vocabulary = vectorizer.vocabulary_
idf = vectorizer.idf_
with open("modeles/tfidf_params.json", "w") as f:
    json.dump({"vocabulary": vocabulary, "idf": idf.tolist()}, f)


y = data['Categories_OFF']  # Labels (catégories)

# 4. Division en données d’entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Modèle 1 : Logistic Regression
print("=== Logistic Regression ===")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print(classification_report(y_test, y_pred_lr))

# Convertir et sauvegarder le modèle Logistic Regression en ONNX
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_lr = convert_sklearn(lr_model, initial_types=initial_type)
with open("modeles/lr_model.onnx", "wb") as f:
    f.write(onnx_lr.SerializeToString())

# 6. Modèle 2 : SVM
print("\n=== Support Vector Machine (SVM) ===")
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print(classification_report(y_test, y_pred_svm))

# Création du dossier 'modeles' s'il n'existe pas
#os.makedirs("modeles", exist_ok=True)
# Convertir et sauvegarder le modèle SVM en ONNX
onnx_svm = convert_sklearn(svm_model, initial_types=initial_type)
with open("modeles/svm_model.onnx", "wb") as f:
    f.write(onnx_svm.SerializeToString())
