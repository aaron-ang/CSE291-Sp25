import os

import pandas as pd
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from scipy import stats


def get_drug_names(df: pd.DataFrame):
    """
    Extract sorted, unique drug names from column headers.
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
    drug_fc: pd.Series,
    protein: str,
    min_samples: int = 20,
    alpha: float = 0.01,
):
    """
    Compare the distribution of a drug’s fold change values to the overall distribution.
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
    sns.kdeplot(drug_fc, color="black", linestyle="--", ax=ax)
    labels.append(f"Overall (n={len(drug_fc)})")

    rows = []
    for col in drug_cols:
        drug_concentration = col.split()[1].split(".")[0]
        vals = protein_df[col].dropna()
        n = len(vals)
        if n < min_samples:
            continue

        d = compute_cohens_d(vals, drug_fc)
        effect = interpret_effect(d)
        if effect == "negligible" or effect == "small":
            continue
        t_stat, p_val = stats.ttest_ind(vals, drug_fc, equal_var=False)
        if np.isnan(p_val):
            print(f"WARNING: p-value is NaN for {drug} at {drug_concentration} nM.")
            continue
        if np.isnan(t_stat):
            print(f"WARNING: t-statistic is NaN for {drug} at {drug_concentration} nM.")
            continue
        if p_val > alpha:
            print(
                f"Skipping {drug} at {drug_concentration} nM: p-value {p_val:.4f} > alpha {alpha}"
            )
            continue
        rows.append(
            {
                "Drug": drug,
                "Protein": protein,
                "Concentration": drug_concentration,
                "Sample_Size": n,
                "Mean": vals.mean(),
                "Std": vals.std(ddof=1),
                "Cohens_d": d,
                "Effect": effect,
                "t_stat": t_stat,
                "p_value": p_val
            }
        )
        
        lbl = f"{drug_concentration} (n={n}, {effect})"
        sns.kdeplot(vals, ax=ax)
        labels.append(lbl)
    
    if len(labels) < 2:
        print(f"No significant results for {drug} on {protein}.")
        plt.close(fig)
        return pd.DataFrame({
            "Drug": [],
            "Protein": [],
            "Concentration": [],
            "Sample_Size": [],
            "Mean": [],
            "Std": [],
            "Cohens_d": [],
            "Effect": [],
            "t_stat": [],
            "p_value": []
        })
    
    ax.set_title(f"Distribution of {drug} fold changes for {protein}")
    ax.set_xlabel("Fold Change")
    ax.set_ylabel("Density")
    ax.legend(labels, loc="upper right")
    plt.tight_layout()
    
    output_dir = f"data/drug_distributions"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(
        output_dir, f"{protein}"
    )
    
    if sys.platform == "win32":
        output_path = output_path.replace("|", "-")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    output_path = os.path.join(
        output_path, f"{protein}_{drug}.png"
    )
    if sys.platform == "win32":
        output_path = output_path.replace("|", "-")
    fig.savefig(
        output_path
    )
    plt.clf()
    plt.close(fig)
    
    df = pd.DataFrame(rows)
    return df


def print_summary_statistics(results: pd.DataFrame, drugs):
    print("\nAnalysis Summary:")
    print(f"Total drugs analyzed: {len(drugs)}")

    # Effect size summary
    effect_summary = results["Effect"].value_counts()
    print("\nEffect Size Distribution (Significant Results):")
    print(effect_summary)


def main():
    output_dir = "data/drug_distributions"
    input_file_path = "data/mq_variants_intensity_cleaned.csv"
    peptide_scores_path = "data/variant_scores.csv"
    results_output_path = "data/drug_distribution_stats.csv"

    # Create output directory for plots
    os.makedirs(output_dir, exist_ok=True)

    # Load and process data
    df = pd.read_csv(input_file_path)
    drugs = get_drug_names(df)
    df_final, protein_counts = process_protein_data(df)

    # Get the most frequent protein
    results = []
    most_frequent_proteins = protein_counts.index[:3]
    for most_frequent_protein in most_frequent_proteins:
        print(
            f"Analyzing protein: {most_frequent_protein} ({protein_counts.iloc[0]} occurrences)"
        )
        most_freq_protein_df = df_final[df_final["Proteins"] == most_frequent_protein]

        # Analyze each drug
        peptide_scores = pd.read_csv(peptide_scores_path)
        for drug in drugs:
            print(f"Analyzing drug {drug}...")
            drug_fc = peptide_scores[peptide_scores["drug"] == drug]["log_fold_change"]
            result_df = analyze_drug(
                drug,
                most_freq_protein_df,
                drug_fc,
                most_frequent_protein
            )
            if result_df is not None:
                results.append(result_df)

    # Combine and save all results
    if not results:
        print("No results to save.")
        return

    combined_results = pd.concat(results, ignore_index=True)

    print_summary_statistics(combined_results, drugs)

    combined_results.to_csv(results_output_path, index=False)


if __name__ == "__main__":
    main()
