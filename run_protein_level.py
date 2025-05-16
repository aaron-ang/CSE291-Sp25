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


def _compute_cohens_d(a: np.ndarray, b: np.ndarray, equal_var: bool = False) -> float:
    """Compute Cohen’s d for two samples."""
    na, nb = len(a), len(b)
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    if equal_var:
        pooled = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
        return (a.mean() - b.mean()) / np.sqrt(pooled) if pooled > 0 else 0.0
    # Welch’s d
    return (
        (a.mean() - b.mean()) / np.sqrt(var_a / na + var_b / nb)
        if (var_a > 0 and var_b > 0)
        else 0.0
    )


def _interpret_effect(d: float) -> str:
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
    df: pd.DataFrame,
    intensities: list[float],
    protein: str,
    output_dir: str,
    min_samples: int = 30,
    max_concentrations: int = 5,
    alpha: float = 0.05,
    equal_var: bool = False,
):
    """
    Compare the distribution of a drug’s intensities against the overall background.
    """
    # Validate inputs
    intensities = np.asarray(intensities, dtype=float)
    if intensities.size < min_samples:
        print(
            f"overall intensities ({intensities.size}) < min_samples ({min_samples}); skipping {drug}"
        )
        return None

    # Select up to max_concentrations
    drug_cols = [c for c in df.columns if drug in c and "nM" in c][:max_concentrations]
    if not drug_cols:
        return None

    # Create plot
    fig, ax = plt.subplots(figsize=(15, 8))
    labels = []
    sns.kdeplot(
        intensities,
        color="black",
        linestyle="--",
        ax=ax,
    )
    labels.append(f"Overall (n={len(intensities)})")

    rows = []
    for col in drug_cols:
        vals = df[col].dropna().astype(float).values
        n = len(vals)
        if n == 0:
            continue

        d = _compute_cohens_d(vals, intensities, equal_var=equal_var)
        t_stat, p_val = stats.ttest_ind(vals, intensities, equal_var=equal_var)
        effect = _interpret_effect(d)
        enough = n >= min_samples
        lbl = f"{col} (n={n}, {effect})" + ("" if enough else " [low n]")

        sns.kdeplot(vals, ax=ax)
        labels.append(lbl)

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

    ax.set_title(f"{protein} – {drug} (min n={min_samples}, α={alpha})")
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
            f"Drugs with significant differences: {final_results['Significant'].value_counts()[True]}"
        )

        # Effect size summary
        effect_summary = final_results["Effect"].value_counts()
        print("\nEffect Size Distribution:")
        print(effect_summary)

        # Sample size issues
        insufficient_samples = final_results[~final_results["Enough_Samples"]]
        if len(insufficient_samples) > 0:
            print(
                f"\nWARNING: {len(insufficient_samples)} conditions had insufficient samples."
            )
