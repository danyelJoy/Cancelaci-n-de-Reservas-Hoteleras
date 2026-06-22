import pandas as pd
from pathlib import Path


RUTA_DATOS_INI = Path("datos/datos_ini/hotel_bookings.csv")
RUTA_DATOS_LIMP = Path("datos/datos_limp/hotel_bookings_clean.csv")


def cargar_datos(ruta=RUTA_DATOS_INI):
    return pd.read_csv(ruta)


def limpiar_datos(df):
    df = df.copy()

    # Eliminar duplicados
    df = df.drop_duplicates()

    # Manejo de valores nulos
    df["children"] = df["children"].fillna(0)
    df["country"] = df["country"].fillna("Unknown")
    df["agent"] = df["agent"].fillna(0)
    df["company"] = df["company"].fillna(0)

    # Conversión de tipos
    df["children"] = df["children"].astype(int)
    df["agent"] = df["agent"].astype(int)
    df["company"] = df["company"].astype(int)

    # Reglas de negocio: eliminar reservas sin huéspedes
    df = df[
        (df["adults"] + df["children"] + df["babies"]) > 0
    ]

    # Crear variable de estancia total
    df["total_nights"] = (
        df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    )

    # Eliminar registros con estancia negativa o inválida
    df = df[df["total_nights"] >= 0]

    return df


def guardar_datos(df, ruta=RUTA_DATOS_LIMP):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False)


def main():
    df = cargar_datos()
    print(f"Datos originales: {df.shape}")

    df_limpio = limpiar_datos(df)
    print(f"Datos limpios: {df_limpio.shape}")

    guardar_datos(df_limpio)
    print(f"Archivo guardado en: {RUTA_DATOS_LIMP}")


if __name__ == "__main__":
    main()