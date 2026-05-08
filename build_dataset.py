import pandas as pd
import os

INPUT_PATH = "/work/u10767335/data/Books_5core.jsonl"
OUTPUT_PATH = "/work/u10767335/data/amazon_model_dataset_full.csv"

chunksize = 50000

# cancella file se esiste
if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)

reader = pd.read_json(INPUT_PATH, lines=True, chunksize=chunksize)

first_chunk = True
global_item_id = 0

for chunk_id, df_chunk in enumerate(reader):

    print(f"Processing chunk {chunk_id}")

    # filtro testi validi
    df_chunk = df_chunk[df_chunk["text"].notna()]

    # selezione colonne
    df_model = df_chunk[["text", "parent_asin", "rating"]].copy()

    # rename
    df_model = df_model.rename(columns={
        "text": "sentence",
        "parent_asin": "participant",   # 🔥 tieni stringa!
        "rating": "target"
    })

    # item globale
    df_model["item"] = range(global_item_id, global_item_id + len(df_model))
    global_item_id += len(df_model)

    # 🔥 NON fare encoding qui!!

    # mapping target
    df_model["target"] = df_model["target"].map({
        1: 0,
        2: 0,
        3: 1,
        4: 2,
        5: 2
    })

    # salva
    df_model.to_csv(
        OUTPUT_PATH,
        mode="a",
        header=first_chunk,
        index=False
    )

    first_chunk = False

print("DONE dataset full")
