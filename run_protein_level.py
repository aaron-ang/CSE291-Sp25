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


def compute_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen’s d for two samples."""
    na, nb = len(a), len(b)
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
    return (a.mean() - b.mean()) / np.sqrt(pooled) if pooled > 0 else 0.0


def interpret_effect(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def analyze_drug(
    drug: str,
    protein_df: pd.DataFrame,
    drug_intensities: pd.Series,
    protein: str,
    output_dir: str,
    min_samples: int = 30,
    alpha: float = 0.01,
):
    """
    Compare the distribution of a drug’s intensities against the overall background.
    """
    drug_cols = [c for c in protein_df.columns if f"{drug} " in c and "nM" in c]

    # Sort drug columns by concentration
    def concentration(col: str):
        concentration = col.split()[1].split("nM")[0]
        return int(concentration)

    drug_cols.sort(key=lambda x: concentration(x))

    # Plot background distribution
    fig, ax = plt.subplots(figsize=(15, 8))
    labels = []
    sns.kdeplot(
        drug_intensities,
        color="black",
        linestyle="--",
        ax=ax,
    )
    labels.append(f"Overall (n={len(drug_intensities)})")

    rows = []
    for col in drug_cols:
        vals = protein_df[col].dropna()
        n = len(vals)
        if n < 2:
            print(
                f"WARNING: {col} has only {n} samples. Skipping analysis for this drug."
            )
            continue

        d = compute_cohens_d(vals, drug_intensities)
        effect = interpret_effect(d)

        enough = n >= min_samples
        drug_concentration = col.split()[1].split(".")[0]
        lbl = f"{drug_concentration} (n={n}, {effect})"
        if not enough:
            lbl += " [low n]"

        sns.kdeplot(vals, ax=ax)
        labels.append(lbl)

        t_stat, p_val = stats.ttest_ind(vals, drug_intensities, equal_var=False)

        rows.append(
            {
                "Drug": drug,
                "Protein": protein,
                "Column": col,
                "Sample_Size": n,
                "Enough_Samples": enough,
                "Mean": vals.mean(),
                "Std": vals.std(ddof=1),
                "Cohens_d": d,
                "Effect": effect,
                "t_stat": t_stat,
                "p_value": p_val,
                "Significant": p_val < alpha,
            }
        )

    if not rows:
        plt.close(fig)
        return None

    ax.set_title(f"{protein} – {drug} (α={alpha})")
    ax.set_xlabel("Intensity")
    ax.legend(labels=labels, loc="best")
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{protein}_{drug}_distribution.png")
    fig.savefig(out_path)
    plt.close(fig)

    return pd.DataFrame(rows)


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
    print(
        f"Most frequent protein: {most_frequent_protein} ({protein_counts.iloc[0]} occurrences)"
    )
    most_freq_protein_df = df_final[df_final["Proteins"] == most_frequent_protein]

    # Analyze each drug
    intensities_df = pd.read_csv("data/variant_intensity_scores.csv")

    print(f"Overall intensities: {len(intensities_df)} samples")

    all_results = []
    for drug in drugs:
        print(f"Analyzing drug {drug}...")
        drug_intensities = intensities_df[intensities_df["drug"] == drug]["intensity"]
        results = analyze_drug(
            drug,
            most_freq_protein_df,
            drug_intensities,
            most_frequent_protein,
            output_dir,
        )
        if results is not None:
            all_results.append(results)

    # Combine and save all results
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        final_results.to_csv("data/all_drug_distribution_stats.csv", index=False)

        # Print summary statistics
        print("\nAnalysis Summary:")
        print(f"Total drugs analyzed: {len(drugs)}")
        signficant_count = final_results["Significant"].sum()
        print(
            f"{signficant_count} ({signficant_count / len(final_results) * 100:.2f}%) of drug dosages were significant."
        )

        # Effect size summary
        effect_summary = final_results[final_results["Significant"]][
            "Effect"
        ].value_counts()
        print("\nEffect Size Distribution (Significant Results):")
        print(effect_summary)

        # Sample size issues
        insufficient_samples = final_results[~final_results["Enough_Samples"]]
        if len(insufficient_samples) > 0:
            print(
                f"\nWARNING: {len(insufficient_samples)} ({len(insufficient_samples) / len(final_results) * 100:.2f}%) of analyzed dosages had insufficient samples."
            )
