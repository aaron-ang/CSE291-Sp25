import numpy as np
import pandas as pd
from tqdm import tqdm

from monotonic_variants import (
    get_sorted_drug_cols,
    is_strictly_monotonic,
    same_sign_filter,
)


if __name__ == "__main__":
    # Read CSV and filter to target proteins
    df = pd.read_csv("data/mq_variants_intensity_cleaned.csv")
    drug_stats = pd.read_csv("data/drug_distribution_stats.csv")

    target_proteins = drug_stats["Protein"].unique()
    df = df[df["Proteins"].isin(target_proteins)]
    sorted_drug_cols = get_sorted_drug_cols(df)

    # Initialize statistics
    stats = {
        "proteins_tested": 0,
        "drugs_tested": len(sorted_drug_cols),
        "monotonic_protein_drug_pairs": 0,
    }

    # Calculate average intensities for each protein and drug combination
    results = []
    for protein, grp in tqdm(df.groupby("Proteins")):
        stats["proteins_tested"] += 1

        for drug, cols in sorted_drug_cols.items():
            cols, concs = zip(*cols)

            # Calculate average intensity for each concentration
            avg_intensities = []
            for col in cols:
                avg_intensity = float(grp[col].mean())
                avg_intensities.append(avg_intensity)

            if is_strictly_monotonic(np.array(avg_intensities)):
                stats["monotonic_protein_drug_pairs"] += 1
                results.append(
                    {
                        "Protein": protein,
                        "Drug": drug,
                        "Avg_Intensities": avg_intensities,
                        **{
                            f"Avg_Intensity_{conc}": intensity
                            for conc, intensity in zip(concs, avg_intensities)
                        },
                    }
                )

    # Print summary statistics
    print("=== Protein-Level Monotonic Summary ===")
    print(f"Proteins tested: {stats['proteins_tested']}")
    print(f"Drugs tested: {stats['drugs_tested']}")
    print(f"Strictly monotonic cases: {stats['monotonic_protein_drug_pairs']}")

    results_df = pd.DataFrame(results)

    # Filter out results where intensities change sign
    df_len = len(results_df)
    results_df_filtered = results_df[
        results_df["Avg_Intensities"].apply(same_sign_filter)
    ]
    df_len_filtered = len(results_df_filtered)
    diff = df_len - df_len_filtered
    print(f"\nFiltered out {diff} ({(diff / df_len):.2%}) results that changed sign")

    output_path = "data/monotonic_protein_averages.csv"
    results_df_filtered.to_csv(output_path, index=False)
