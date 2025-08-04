import pandas as pd
import joblib
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer

# Charger modèle et vectorizer_joblib
model = joblib.load("modeles/logreg_model.joblib")
vectorizer_joblib = joblib.load("modeles/vectorizer.joblib")

# Charger les nouvelles données
df_nouveau = pd.read_csv("data/tickets_categorie_final.csv", sep=';')
joblib_idf= vectorizer_joblib.idf_
joblib_vocabulary = vectorizer_joblib.vocabulary_

# Prétraitement
df_nouveau['Produit'] = df_nouveau['Produit'].str.lower()
df_nouveau['Produit'] = df_nouveau['Produit'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

# Vectorisation
X_joblib = vectorizer_joblib.transform(df_nouveau['Produit'])


with open("modeles/tfidf_params.json", "r") as f:
    params = json.load(f)

onnx_vocabulary = params["vocab"]
onnx_idf = np.array(params["idf"])


print (f"len(joblib_idf)={len(joblib_idf)}, len(onnx_idf)={len(onnx_idf)}")
difference = np.abs(joblib_idf - onnx_idf)
if np.allclose(difference, np.zeros_like(difference)):
    print("Les valeurs IDF sont identiques.")
else:
    print("Les valeurs IDF sont différentes.")
    print("Différences :", difference)

if onnx_vocabulary == joblib_vocabulary:
    print("Les vocabulaires sont identiques.")
else:
    print("Les vocabulaires sont différents.")

    # Trouver les différences
    mots_onnx = set(onnx_vocabulary.keys())
    mots_joblib = set(joblib_vocabulary.keys())

    mots_uniquement_onnx = mots_onnx - mots_joblib
    mots_uniquement_joblib =  mots_joblib - mots_onnx 

    print("mots_uniquement_onnx:", mots_uniquement_onnx)
    print("mots_uniquement_joblib:", mots_uniquement_joblib)


vectorizer_onnx = TfidfVectorizer(vocabulary=onnx_vocabulary)

# Assigner les valeurs IDF
vectorizer_onnx.idf_ = onnx_idf


def compare_vectorizer_parameters(vectorizer1, vectorizer2):
    # Liste des paramètres à comparer
    parameters_to_compare = [
        'binary', 'decode_error', 'dtype', 'encoding',
        'input', 'lowercase', 'max_df', 'max_features',
        'min_df', 'ngram_range', 'norm', 'preprocessor',
        'smooth_idf', 'stop_words', 'strip_accents',
        'sublinear_tf', 'token_pattern', 'tokenizer',
        'use_idf', 'vocabulary'
    ]

    # Comparaison des paramètres
    for param in parameters_to_compare:
        value1 = getattr(vectorizer1, param)
        value2 = getattr(vectorizer2, param)
        if value1 != value2:
            print(f"Différence trouvée dans le paramètre '{param}': {value1} vs {value2}")

# Exemple d'utilisation
compare_vectorizer_parameters(vectorizer_onnx, vectorizer_onnx)



