import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def get_drug_names(df: pd.DataFrame):
    """
    Extract sorted, unique drug names from intensity column headers.
    Expects columns like "_dyn_#DRUG 10nM.Tech replicate…".
    """
    pattern = r"_dyn_#(?P<drug>[^ ]+) \d+nM"
    drugs = df.columns.to_series().str.extract(pattern)["drug"].dropna().unique()
    return sorted(drugs)


def process_protein_data(df: pd.DataFrame):
    # Filter out rows where Proteins contain semicolons
    df_filtered = df[~df["Proteins"].str.contains(";", na=False)]

    # Count occurrences of each protein in descending order
    protein_counts = df_filtered["Proteins"].value_counts()

    # Filter proteins that appear in at least 10 rows
    sample_count = 20
    frequent_proteins = protein_counts[protein_counts >= sample_count]

    # Create a filtered dataset with only frequent proteins
    df_frequent = df_filtered[df_filtered["Proteins"].isin(frequent_proteins.index)]

    return df_frequent, frequent_proteins


def analyze_drug(drug, df, intensities, protein, output_dir, min_samples=30):
    """Analyze a single drug's distributions and create statistical results."""
    # Get columns for this drug
    cols = df.columns.to_series()
    # Still limit to 5 concentrations
    drug_cols = [col for col in cols if drug in col and "nM" in col][:5]

    if not drug_cols:
        return None

    # Create plot
    plt.figure(figsize=(15, 8))
    sns.kdeplot(
        data=intensities,
        label=f"Overall Distribution (n={len(intensities)})",
        linestyle="--",
        color="black",
    )

    stat_results = []

    for col in drug_cols:
        col_values = df[col].dropna().values
        if len(col_values) > 0:
            # Calculate effect size
            cohens_d = (np.mean(col_values) - np.mean(intensities)) / np.sqrt(
                (
                    (len(col_values) - 1) * np.var(col_values)
                    + (len(intensities) - 1) * np.var(intensities)
                )
                / (len(col_values) + len(intensities) - 2)
            )

            # Perform t-test
            t_stat, p_val = stats.ttest_ind(col_values, intensities, equal_var=False)

            # Interpret effect size
            if abs(cohens_d) < 0.2:
                effect = "negligible"
            elif abs(cohens_d) < 0.5:
                effect = "small"
            elif abs(cohens_d) < 0.8:
                effect = "medium"
            else:
                effect = "large"

            # Create label
            enough_samples = len(col_values) >= min_samples
            label = f"{col} (n={len(col_values)}, {effect} effect)"
            if not enough_samples:
                label += " [insufficient samples]"

            # Plot distribution
            sns.kdeplot(data=col_values, label=label)

            # Store results
            stat_results.append(
                {
                    "Drug": drug,
                    "Column": col,
                    "Sample_Size": len(col_values),
                    "Enough_Samples": enough_samples,
                    "p-value": p_val,
                    "Effect_Size": cohens_d,
                    "Effect_Interpretation": effect,
                    "Statistically_Significant": "Yes" if p_val < 0.05 else "No",
                    "Mean": np.mean(col_values),
                    "Std": np.std(col_values),
                }
            )

    if not stat_results:
        plt.close()
        return None

    plt.title(
        f"Distribution Comparison for {protein} - {drug}\n"
        f"Minimum sample size requirement: {min_samples}\n"
        "Effect size interpretation: |d| < 0.2 negligible, 0.2 ≤ |d| < 0.5 small, "
        "0.5 ≤ |d| < 0.8 medium, |d| ≥ 0.8 large"
    )
    plt.xlabel("Intensity")
    plt.ylabel("Density")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save plot
    plt.savefig(
        os.path.join(output_dir, f"distro_comparison_{drug}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return pd.DataFrame(stat_results)


if __name__ == "__main__":
    # Create output directory for plots
    output_dir = "data/drug_distributions"
    os.makedirs(output_dir, exist_ok=True)

    # Load and process data
    file_path = "data/mq_variants_intensity_cleaned.csv"
    df = pd.read_csv(file_path)
    drugs = get_drug_names(df)
    df_final, protein_counts = process_protein_data(df)

    # Get the most frequent protein
    most_frequent_protein = protein_counts.index[0]
    most_freq_protein_df = df_final[df_final["Proteins"] == most_frequent_protein]

    # Analyze each drug
    with open("data/all_intensities_distribution.txt") as f:
        intensities = [float(line.strip()) for line in f]

    all_results = []
    for drug in drugs:
        print(f"Analyzing drug {drug}...")
        results = analyze_drug(
            drug, most_freq_protein_df, intensities, most_frequent_protein, output_dir
        )
        if results is not None:
            all_results.append(results)

    # Combine and save all results
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        final_results.to_csv("data/all_drug_distribution_stats.csv", index=False)

        # Print summary statistics
        print("\nAnalysis Summary:")
        print(f"Total drugs analyzed: {len(all_results)}")
        print(
            f"Drugs with significant differences: {final_results['Statistically_Significant'].value_counts()['Yes']}"
        )

        # Effect size summary
        effect_summary = final_results["Effect_Interpretation"].value_counts()
        print("\nEffect Size Distribution:")
        print(effect_summary)

        # Sample size issues
        insufficient_samples = final_results[~final_results["Enough_Samples"]]
        if len(insufficient_samples) > 0:
            print(
                f"\nWARNING: {len(insufficient_samples)} conditions had insufficient samples."
            )
