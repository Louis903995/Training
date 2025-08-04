```mermaid
graph LR
    A[data/mapping.json]:::dataset --> B[src/mapping.py]:::script
    C[data/tickets_nettoye.csv]:::dataset --> B
    B --> D[data/tickets_categorie_final.csv]:::dataset
    D --> E[src/modelisation.py]:::script
    E --> F[modeles/tfidf_params.json]:::dataset
    E --> G[modeles/lr_model.onnx]:::dataset

    classDef dataset fill:#f9f,stroke:#333,stroke-width:2px,shape=cylinder;
    classDef script fill:#bbf,stroke:#333,stroke-width:2px;
```
