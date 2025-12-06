from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

from produccion.config import RAW_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # Archivo RAW real que tienes en data/raw/
    input_path: Path = RAW_DATA_DIR / "sales_data_sample.csv",
    # Archivo que espera DVC como output del stage `preproc`
    output_path: Path = INTERIM_DATA_DIR / "feature_exploration_scaled.csv",
):
    logger.info(f"Cargando datos crudos desde: {input_path}")

    # 👇 IMPORTANTE: encoding para evitar el UnicodeDecodeError
    df = pd.read_csv(input_path, encoding="latin1")

    logger.info("Procesando dataset...")

    # Aquí pondrías tu lógica real de limpieza / escalado.
    # De momento, para que el pipeline funcione, simplemente copiamos el dataframe.
    df_final = df.copy()

    # Guardar el resultado en data/interim/feature_exploration_scaled.csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)

    logger.success(f"Dataset procesado y guardado en: {output_path}")


if __name__ == "__main__":
    app()