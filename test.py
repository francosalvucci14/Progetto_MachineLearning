import pandas as pd
import numpy as np

# --- Caricamento robusto ---
try:
    df = pd.read_csv("Dataset/web-page-phishing/dataset_phishing.csv")
except:
    df = pd.read_csv("Dataset/web-page-phishing/dataset_phishing.csv", sep=";")

# Manteniamo solo colonne numeriche
df_num = df.select_dtypes(include=[np.number])

# Se ultima colonna è target la rimuoviamo
features = df_num.columns[:-1]

results = []

for col in features:
    x = df_num[col]
    
    results.append({
        "feature": col,
        "min": x.min(),
        "max": x.max(),
        "mean": x.mean(),
        "std": x.std(),
        "median": x.median(),
        "99perc": x.quantile(0.99),
        "max/median": x.max() / (abs(x.median()) + 1e-9),
        "skewness": x.skew()
    })

stats_df = pd.DataFrame(results)

# Ordinamenti utili
by_max = stats_df.sort_values("max", ascending=False)
by_std = stats_df.sort_values("std", ascending=False)
by_skew = stats_df.sort_values("skewness", ascending=False)

print("=== Feature con valori massimi più elevati ===")
print(by_max)

print("\n=== Feature con deviazione standard più alta ===")
print(by_std)

print("\n=== Feature con maggiore asimmetria (skewness) ===")
print(by_skew)
