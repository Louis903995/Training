import pandas as pd
import csv

# Liste des grandes catégories selon ton découpage
big_categories_keywords = {
    "Fruits": "Fruits & légumes",
    "Légumes": "Fruits & légumes",
    "Viandes": "Viandes & poissons",
    "Poissons": "Viandes & poissons",
    "Dinde": "Viandes & poissons",
    "Volailles": "Viandes & poissons",
    "Charcuteries": "Viandes & poissons",
    "Fromages": "Produits laitiers",
    "Yaourts": "Produits laitiers",
    "Lait": "Produits laitiers",
    "Produits laitiers": "Produits laitiers",
    "Épicerie salée": "Épicerie salée",
    "Snacks salés": "Épicerie salée",
    "Biscuits salés": "Épicerie salée",
    "Terrines": "Épicerie salée",
    "Épicerie sucrée": "Épicerie sucrée",
    "Biscuits sucrés": "Épicerie sucrée",
    "Gâteaux": "Épicerie sucrée",
    "Pâtisseries": "Épicerie sucrée",
    "Madeleines": "Épicerie sucrée",
    "Donuts": "Épicerie sucrée",
    "Viennoiseries": "Épicerie sucrée",
    "Pâtes à tartiner": "Épicerie sucrée",
    "Produits à tartiner sucrés": "Épicerie sucrée",
    "Surgelés": "Surgelés",
    "Frais": "Frais",
    "Eau": "Eaux",
    "Eaux": "Eaux",
    "Boissons alcoolisées": "Boissons alcoolisées",
    "Vin": "Boissons alcoolisées",
    "Bière": "Boissons alcoolisées",
    "Spiritueux": "Boissons alcoolisées",
    "Champagne": "Boissons alcoolisées",
    "Cidre": "Boissons alcoolisées",
    "Boissons": "Boissons non alcoolisées (hors eaux)",  # catch-all si pas eau ou alcool
    "Soda": "Boissons non alcoolisées (hors eaux)",
    "Jus": "Boissons non alcoolisées (hors eaux)",
    "Thé": "Boissons non alcoolisées (hors eaux)",
    "Café": "Boissons non alcoolisées (hors eaux)",
}

FALLBACK_CATEGORY = "Autres"


def assign_big_category(category_fine):
    # On teste chaque mot-clé pour chaque catégorie fine
    for keyword, big_cat in big_categories_keywords.items():
        if keyword.lower() in category_fine.lower():
            # Exception pour bien distinguer les boissons
            if big_cat == "Boissons non alcoolisées (hors eaux)":
                if "eau" in category_fine.lower():
                    return "Eaux"
                if (
                    "vin" in category_fine.lower()
                    or "bière" in category_fine.lower()
                    or "cidre" in category_fine.lower()
                    or "champagne" in category_fine.lower()
                    or "spiritueux" in category_fine.lower()
                    or "alcool" in category_fine.lower()
                ):
                    return "Boissons alcoolisées"
            return big_cat
    return FALLBACK_CATEGORY


# Lecture du CSV
with open("data/categories.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = [row[0] for row in reader]

mapping_list = []

for row in rows:
    fine_categories = [cat.strip() for cat in row.split(",")]
    for fine_cat in fine_categories:
        big_cat = assign_big_category(fine_cat)
        mapping_list.append({"category": fine_cat, "big_category": big_cat})

df_mapping = pd.DataFrame(mapping_list).drop_duplicates()
df_mapping.to_csv("data/mapping.csv", index=False)
