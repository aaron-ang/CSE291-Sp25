import re
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


# Parse drug and concentration from the column names
# Example: '_dyn_#AEE-788_inBT474 1000nM.Tech replicate 1 of 1'
def parse_drug_conc(col):
    m = re.match(r"_dyn_#([\w\-]+)(?:_in[\w\d]+)? ([\d]+nM|DMSO)", col)
    return (m.group(1), m.group(2)) if m else (None, None)


def to_nm(x: str):
    return int(x.rstrip("nM"))


# Function to check strict monotonicity
def is_strictly_monotonic(arr):
    return np.all(np.diff(arr) > 0) or np.all(np.diff(arr) < 0)


def same_sign_filter(intensities):
    intensities = np.array(intensities)
    return np.all(intensities >= 0) or np.all(intensities <= 0)


if __name__ == "__main__":
    # Read CSVs and filter to target proteins
    df = pd.read_csv("data/mq_variants_intensity_cleaned.csv")
    drug_stats = pd.read_csv("data/drug_distribution_stats.csv")
    target_proteins = drug_stats["Protein"].unique()
    df = df[df["Proteins"].isin(target_proteins)]

    # Group columns by drug
    intensity_cols = [col for col in df.columns if col.startswith("_dyn_#")]
    drug_cols = defaultdict(list)
    for col in intensity_cols:
        drug, conc = parse_drug_conc(col)
        if drug:
            drug_cols[drug].append((col, conc))

    # Sort drug columns by concentration
    keep_concs = ["3nM", "300nM", "3000nM", "30000nM"]
    drug_cols_sorted = {}
    for drug, cols in drug_cols.items():
        filtered = [(col, conc) for col, conc in cols if conc in keep_concs]
        drug_cols_sorted[drug] = sorted(filtered, key=lambda x: to_nm(x[1]))

    # Initialize global statistics
    stats = {
        "proteins_tested": 0,
        "drugs_tested": len(drug_cols_sorted),
        "variants_tested": 0,
        "strict_cases": 0,
    }

    # For each protein, check for strictly monotonic intensity changes across concentrations for each drug
    results = []
    for protein, grp in tqdm(df.groupby("Proteins")):
        stats["proteins_tested"] += 1
        variants = grp["Variant"]
        for drug, cols in drug_cols_sorted.items():
            cols, concs = zip(*cols)
            data = grp[list(cols)].to_numpy()
            stats["variants_tested"] += data.shape[0]

            mask = np.apply_along_axis(is_strictly_monotonic, 1, data)
            strict_n = mask.sum()
            stats["strict_cases"] += int(strict_n)

            for var, intens in zip(variants[mask], data[mask]):
                results.append(
                    {
                        "Protein": protein,
                        "Drug": drug,
                        "Variant": var,
                        "Concentrations": concs,
                        "Intensities": intens.tolist(),
                    }
                )

    # Print summary statistics
    print("=== Monotonic Summary ===")
    print(f"Proteins tested: {stats['proteins_tested']}")
    print(f"Drugs tested:    {stats['drugs_tested']}")
    print(f"Variants tested: {stats['variants_tested']}")
    print(f"Strictly monotonic cases: {stats['strict_cases']}")

    results_df = pd.DataFrame(results)

    # Filter out results where intensities change sign
    df_len = len(results_df)
    results_df_filtered = results_df[results_df["Intensities"].apply(same_sign_filter)]
    df_len_filtered = len(results_df_filtered)
    diff = df_len - df_len_filtered
    print(f"\nFiltered out {diff} ({(diff / df_len):.2%}) results that changed sign")

    results_df_filtered.to_csv("data/monotonic_variant_drug_combos.csv", index=False)
