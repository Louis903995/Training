import json
import numpy as np
import onnxruntime as rt
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Charger les modèles ONNX
sess_lr = rt.InferenceSession("modeles/lr_model.onnx")
# sess_svm = rt.InferenceSession("modeles/svm_model.onnx")

# Charger le vectorizer
with open("modeles/tfidf_params.json", "r") as f:
    params = json.load(f)
onnx_vocabulary = params["vocab"]
onnx_idf = np.array(params["idf"])
vectorizer_onnx = TfidfVectorizer(vocabulary=onnx_vocabulary)
vectorizer_onnx.idf_ = onnx_idf



# Charger les nouvelles données
df_nouveau = pd.read_csv("data/tickets_categorie_final.csv", sep=';')

# Prétraitement
df_nouveau['Produit'] = df_nouveau['Produit'].str.lower()
df_nouveau['Produit'] = df_nouveau['Produit'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

# créer 
X_nouveau = vectorizer_onnx.transform(df_nouveau['Produit']).astype(np.float32).toarray()


# Prédire avec le modèle Logistic Regression
input_name_lr = sess_lr.get_inputs()[0].name
label_name_lr = sess_lr.get_outputs()[0].name
predictions_lr = sess_lr.run([label_name_lr], {input_name_lr: X_nouveau})[0]

print("Prédictions avec Logistic Regression :", predictions_lr)

df_nouveau['Prediction_Categorie'] = predictions_lr

# Sauvegarder le DataFrame avec les prédictions
df_nouveau.to_csv("classification/nouveaux_tickets_predits_onnx.csv", sep=';', index=False)

# Afficher les premières lignes du DataFrame avec les colonnes souhaitées
print(df_nouveau[['Produit', 'Prediction_Categorie']].head())


# # Prédire avec le modèle SVM
# input_name_svm = sess_svm.get_inputs()[0].name
# label_name_svm = sess_svm.get_outputs()[0].name
# predictions_svm = sess_svm.run([label_name_svm], {input_name_svm: X_new})[0]
# print("Prédictions avec SVM :", predictions_svm)
