import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

EDA_PATH = Path("app/data/eda_outputs")


def generate_eda(dataset_id: str, df: pd.DataFrame):
    EDA_PATH.mkdir(parents=True, exist_ok=True)

    dataset_folder = EDA_PATH / dataset_id
    dataset_folder.mkdir(parents=True, exist_ok=True)

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    outputs = {
        "boxplots": [],
        "histograms": [],
        "correlation": None,
        "summary": ""
    }

    # 🔹 Boxplots
    for col in numeric_columns:
        plt.figure()
        sns.boxplot(x=df[col])

        file_name = f"boxplot_{col}.png"
        path = dataset_folder / file_name
        plt.savefig(path)
        plt.close()

        # 🔥 devolver ruta HTTP
        outputs["boxplots"].append(f"/static/{dataset_id}/{file_name}")

    # 🔹 Histogramas
    for col in numeric_columns:
        plt.figure()
        sns.histplot(df[col], kde=True)

        file_name = f"hist_{col}.png"
        path = dataset_folder / file_name
        plt.savefig(path)
        plt.close()

        outputs["histograms"].append(f"/static/{dataset_id}/{file_name}")

    # 🔹 Correlación
    if len(numeric_columns) >= 2:
        plt.figure(figsize=(8, 6))
        corr = df[numeric_columns].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm")

        file_name = "correlation.png"
        path = dataset_folder / file_name
        plt.savefig(path)
        plt.close()

        outputs["correlation"] = f"/static/{dataset_id}/{file_name}"

    # 🔹 Resumen
    outputs["summary"] = (
        f"EDA completado. "
        f"Se analizaron {len(numeric_columns)} columnas numéricas. "
        f"Se generaron boxplots para detección de outliers, histogramas para distribución "
        f"y matriz de correlación para relación entre variables."
    )

    return outputs