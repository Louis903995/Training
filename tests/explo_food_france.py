import pyarrow.dataset as ds
import pandas as pd
import numpy as np

pd.set_option("display.max_rows", 200)  # Mets une valeur plus élevée si besoin


# lecture de tous le fichier mais avec peu de colonnes
df = pd.read_parquet("data/food_france.parquet")


# inutile, c'est déjà nettoyé lors de la création de food_france.parquet
# # # # Diviser les chaînes de caractères en listes et nettoyer les espaces
# # # df["categories"] = (
# # #     df["categories"].str.split(",").apply(lambda x: [item.strip() for item in x])
# # # )
# # # # explode les catégories pourtransforme chaque catégorie en une ligne distincte

df_exploded = df.explode("categories")


# inutile, c'est déjà nettoyé lors de la création de food_france.parquet
# # # # Retirer le préfixe "fr:" et "en:"
# # # df_exploded["categories"] = df_exploded["categories"].str.replace(
# # #     r"^\s*(fr|en):\s*", "", regex=True
# # # )

categories_uniques = df_exploded["categories"].drop_duplicates().reset_index(drop=True)

# Créer un nouveau DataFrame avec les catégories uniques
categories = pd.DataFrame(categories_uniques, columns=["categories"])
categories.to_csv("data/categories.csv", index=False)
print(categories.head())
