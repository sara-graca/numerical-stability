import pandas as pd

df = pd.read_parquet("data/features/word_embeddings.parquet")
emb_cols = [c for c in df.columns if c.startswith("dim_")]
print(df[emb_cols].dtypes.unique())  # probably float32