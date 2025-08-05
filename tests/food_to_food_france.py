import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

pd.set_option("display.max_rows", 200)  # Mets une valeur plus élevée si besoin


# lecture uniquement de 100 lignes, toutes les conolles
# # parquet_file = pq.ParquetFile("data/food.parquet")
# # batch = next(parquet_file.iter_batches(batch_size=100))
# # df = batch.to_pandas()
# # print(df.columns)


# lecture de tous le fichier mais avec peu de colonnes
dataset = ds.dataset("data/food.parquet", format="parquet")
table = dataset.to_table(
    filter=ds.field("lang") == "fr", columns=["product_name", "lang", "categories"]
)
df = table.to_pandas()
df = df.rename(columns={"product_name": "product_name_original"})
print(df[df["lang"] != "fr"])


def extract_nom_fr(names):
    # Si c'est une liste de dicts ou un ndarray
    if isinstance(names, (list, np.ndarray)):
        items = list(names)
        # Cherche le texte en 'fr'
        for item in items:
            if isinstance(item, dict):
                if item.get("lang") == "fr":
                    return str(item.get("text"))
        # Sinon, cherche le texte en 'main'
        for item in items:
            if isinstance(item, dict):
                if item.get("lang") == "main":
                    return str(item.get("text"))
        # Si ce sont des strings, retourne le premier
        if items and isinstance(items[0], str):
            return items[0]
    # Si c'est déjà un str, retourne tel quel
    if isinstance(names, str):
        return names
    # Si c'est None ou autre, retourne None
    return None


# Fonction pour nettoyer les catégories
def clean_categories(categories):
    if categories is None:
        return None

    # Diviser les catégories et nettoyer chacune individuellement
    cleaned_categories = []
    for cat in categories.split(","):
        cat = cat.strip()
        if cat.startswith("fr:"):  # on nettoie fr:
            cleaned_categories.append(cat[3:])
        elif ":" not in cat and cat:  # on rejete en: ou de: ...
            cleaned_categories.append(cat)

    return ",".join(cleaned_categories)  # Joindre la liste cleaned_categories


df["product_name"] = df["product_name_original"].apply(extract_nom_fr)
df["categories"] = df["categories"].apply(clean_categories)

print(df.shape)
filtre = (
    df["product_name"].notnull()  # pas None ou NaN
    & df["categories"].notnull()  # pas None ou NaN
    & df["product_name"].apply(
        lambda x: isinstance(x, str) and x.strip() != ""  # string non vide
    )
)

print(df[filtre][["product_name", "categories"]].shape)
print(df[filtre][["product_name", "categories"]].head(100))
df[filtre][["product_name", "categories"]].to_parquet("data/food_france.parquet")
