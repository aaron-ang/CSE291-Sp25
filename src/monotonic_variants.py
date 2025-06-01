import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


def get_sorted_drug_cols(df: pd.DataFrame):
    # Group columns by drug
    intensity_cols = [col for col in df.columns if col.startswith("_dyn_#")]
    drug_cols = defaultdict(list)
    for col in intensity_cols:
        drug, conc = parse_drug_conc(col)
        if drug:
            drug_cols[drug].append((col, conc))

    # Sort drug columns by concentration
    sorted_drug_cols = {}
    for drug, cols in drug_cols.items():
        filtered = [(col, conc) for col, conc in cols if "DMSO" not in conc]
        sorted_drug_cols[drug] = sorted(filtered, key=lambda x: to_nm(x[1]))

    return sorted_drug_cols


def parse_drug_conc(col):
    """Parse drug and concentration from the column name.

    Example: '_dyn_#AEE-788_inBT474 1000nM.Tech replicate 1 of 1'
    """
    m = re.match(r"_dyn_#([\w\-]+)(?:_in[\w\d]+)? ([\d]+nM|DMSO)", col)
    return (m.group(1), m.group(2)) if m else (None, None)


def to_nm(x: str):
    return int(x.rstrip("nM"))


def is_approx_monotonic(arr, threshold=0.7):
    if len(arr) <= 1:
        return True
    indices = np.arange(len(arr))
    rho, _ = stats.spearmanr(indices, arr)
    return abs(rho) >= threshold


def same_sign_filter(intensities):
    intensities = np.array(intensities)
    return np.all(intensities >= 0) or np.all(intensities <= 0)


if __name__ == "__main__":
    # Read CSVs and filter to target proteins
    df = pd.read_csv("data/mq_variants_intensity_cleaned.csv")
    drug_stats = pd.read_csv("data/drug_distribution_stats.csv")

    target_proteins = drug_stats["Protein"].unique()
    df = df[df["Proteins"].isin(target_proteins)]
    sorted_drug_cols = get_sorted_drug_cols(df)

    # Initialize global statistics
    global_stats = {
        "proteins_tested": 0,
        "drugs_tested": len(sorted_drug_cols),
        "variants_tested": 0,
        "monotonic_cases": 0,
    }

    # For each protein, check for monotonic intensity changes across concentrations for each drug
    results = []
    for protein, grp in tqdm(df.groupby("Proteins")):
        global_stats["proteins_tested"] += 1
        variants = grp["Variant"]
        for drug, cols in sorted_drug_cols.items():
            cols, concs = zip(*cols)
            data = grp[list(cols)].to_numpy()  # shape (n_variants, n_concentrations)
            global_stats["variants_tested"] += data.shape[0]

            mask = np.apply_along_axis(is_approx_monotonic, 1, data)
            monotonic_n = mask.sum()
            global_stats["monotonic_cases"] += int(monotonic_n)

            for var, intens in zip(variants[mask], data[mask]):
                results.append(
                    {
                        "Protein": protein,
                        "Variant": var,
                        "Drug": drug,
                        "Concentrations": concs,
                        "Intensities": intens.tolist(),
                    }
                )

    # Print summary statistics
    print("=== Variant-Level Monotonic Summary ===")
    print(f"Proteins tested: {global_stats['proteins_tested']}")
    print(f"Drugs tested:    {global_stats['drugs_tested']}")
    print(f"Variants tested: {global_stats['variants_tested']}")
    print(f"Monotonic cases: {global_stats['monotonic_cases']}")

    results_df = pd.DataFrame(results)

    # Filter out results where intensities change sign
    df_len = len(results_df)
    results_df_filtered = results_df[results_df["Intensities"].apply(same_sign_filter)]
    df_len_filtered = len(results_df_filtered)
    diff = df_len - df_len_filtered
    print(f"\nFiltered out {diff} ({(diff / df_len):.2%}) results that changed sign")

    results_df_filtered.to_csv("data/monotonic_variant_drug_combos.csv", index=False)
