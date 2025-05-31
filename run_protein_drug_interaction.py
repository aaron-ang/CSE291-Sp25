import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm


def get_drug_names(df: pd.DataFrame):
    """
    Extract sorted, unique drug names from column headers.
    Expects columns like "_dyn_#DRUG 10nM.Tech replicate…".
    """
    pattern = r"_dyn_#(?P<drug>[^ ]+) \d+nM"
    drugs = df.columns.to_series().str.extract(pattern)["drug"].dropna().unique()
    return sorted(drugs)


def process_protein_data(df: pd.DataFrame, min_occurrences=20):
    # Filter peptides variants that are mapped to multiple proteins
    initial_rows = len(df)
    df_filtered = df[~df["Proteins"].str.contains(";", na=False)]
    rows_after_semicolon = len(df_filtered)

    filtered_rows = initial_rows - rows_after_semicolon
    print(
        f"Filtered out {filtered_rows} ({(filtered_rows / initial_rows):.2%}) "
        f"peptide variants mapped to multiple proteins; {rows_after_semicolon} remain."
    )

    # Filter proteins with fewer than min_occurrences
    protein_counts = df_filtered["Proteins"].value_counts()
    df_filtered = df_filtered[
        df_filtered["Proteins"].isin(
            protein_counts[protein_counts >= min_occurrences].index
        )
    ]
    filtered_rows = rows_after_semicolon - len(df_filtered)
    print(
        f"Filtered out {filtered_rows} ({(filtered_rows / rows_after_semicolon):.2%}) "
        f"peptide variants mapped to proteins with fewer than {min_occurrences} occurrences; "
        f"{len(df_filtered)} remain."
    )

    return df_filtered


def compute_cohens_d(a: np.ndarray, b: np.ndarray):
    """Compute Cohen's d for two samples."""
    na, nb = len(a), len(b)
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
    return (a.mean() - b.mean()) / np.sqrt(pooled) if pooled > 0 else 0.0


def interpret_effect(d: float):
    """Interpret Cohen's d effect size."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def get_drug_columns(protein_df: pd.DataFrame, drug: str):
    """Get and sort drug columns by concentration."""
    drug_cols = [c for c in protein_df.columns if f"{drug} " in c and "nM" in c]
    return sorted(drug_cols, key=extract_concentration)


def extract_concentration(col: str):
    """Extract numeric concentration from column name."""
    return int(col.split()[1].split("nM")[0])


def create_safe_path(base_path: str, filename: str):
    """Create safe file path, handling Windows character restrictions."""
    path = Path(base_path) / filename
    if os.name == "nt":  # Windows
        path = Path(str(path).replace("|", "-"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def analyze_drug(
    drug: str,
    protein_df: pd.DataFrame,
    drug_fc: pd.Series,
    min_samples: int = 20,
    alpha: float = 0.01,
):
    """Analyze drug effects on a specific protein."""
    protein = protein_df["Proteins"].iloc[0]
    drug_cols = get_drug_columns(protein_df, drug)

    # Track filtering statistics
    filter_counts = defaultdict(int)
    results = []

    # Analyze each concentration
    for conc in drug_cols:
        result, filter_reason = analyze_single_concentration(
            conc,
            protein_df,
            drug_fc,
            drug,
            min_samples,
            alpha,
        )
        filter_counts[filter_reason] += 1
        if result:
            results.append(result)

    # Compile statistics
    stats_dict = {
        "tested": len(drug_cols),
        "kept": filter_counts["kept"],
        "filtered_low_n": filter_counts["low_n"],
        "filtered_small_effect": filter_counts["small_effect"],
        "filtered_pval": filter_counts["pval"],
    }

    if not results:
        return pd.DataFrame(), stats_dict

    # Create and save plot
    fig = create_dist_plot(drug_fc, results, drug, protein)
    output_path = create_safe_path(
        "data/drug_distributions",
        f"{protein}/{protein}_{drug}.png",
    )
    fig.savefig(output_path)
    plt.close(fig)

    return pd.DataFrame(results), stats_dict


def analyze_single_concentration(
    col: str,
    protein_df: pd.DataFrame,
    drug_fc: pd.Series,
    drug: str,
    min_samples: int,
    alpha: float,
):
    """Analyze a single drug concentration."""
    drug_conc = col.split()[1].split(".")[0]
    vals = protein_df[col].dropna()
    n = len(vals)

    # Check sample size
    if n < min_samples:
        return None, "low_n"

    # Check effect size
    d = compute_cohens_d(vals, drug_fc)
    effect = interpret_effect(d)
    if effect in ("negligible", "small"):
        return None, "small_effect"

    # Statistical test
    t_stat, p_val = stats.ttest_ind(vals, drug_fc, equal_var=False)
    if np.isnan(p_val) or np.isnan(t_stat):
        print(f"WARNING: NaN stats for {drug}@{drug_conc}nM.")
        return None, "nan_stats"

    if p_val > alpha:
        return None, "pval"

    # Return successful result
    result = {
        "Protein": protein_df["Proteins"].iloc[0],
        "Drug": drug,
        "Concentration": drug_conc,
        "Sample_Size": n,
        "Mean": vals.mean(),
        "Std": vals.std(ddof=1),
        "Cohens_d": d,
        "Effect": effect,
        "t_stat": t_stat,
        "p_value": p_val,
        "values": vals,  # For plotting
    }
    return result, "kept"


def create_dist_plot(drug_fc: pd.Series, results: list[dict], drug: str, protein: str):
    """Create and return distribution plot."""
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot overall distribution
    sns.kdeplot(
        drug_fc,
        color="black",
        linestyle="--",
        ax=ax,
        label=f"Overall (n={len(drug_fc)})",
    )

    # Plot each concentration
    for result in results:
        vals = result.pop("values")
        label = (
            f"{result['Concentration']} (n={result['Sample_Size']}, {result['Effect']})"
        )
        sns.kdeplot(vals, ax=ax, label=label)

    ax.set_title(f"Distribution of {drug} fold changes for {protein}")
    ax.set_xlabel("Fold Change")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    plt.tight_layout()

    return fig


def print_summary_statistics(results: pd.DataFrame):
    """Print analysis summary statistics."""
    if not results.empty:
        print("\nEffect Size Distribution (Significant Results):")
        print(results["Effect"].value_counts())


def print_global_filter_summary(global_stats: dict):
    """Print global filtering statistics."""
    print("\nGlobal Filter Summary:")
    print(f"  Total channels tested:      {global_stats['tested']}")
    print(f"  Total channels kept:        {global_stats['kept']}")
    print(f"  Total filtered (low-n):     {global_stats['filtered_low_n']}")
    print(f"  Total filtered (small eff): {global_stats['filtered_small_effect']}")
    print(f"  Total filtered (p > α):     {global_stats['filtered_pval']}")


def main():
    """Main analysis pipeline."""
    # File paths
    input_file_path = "data/mq_variants_intensity_cleaned.csv"
    peptide_scores_path = "data/variant_scores.csv"
    results_output_path = "data/drug_distribution_stats.csv"

    # Load and process data
    df = pd.read_csv(input_file_path)
    drugs = get_drug_names(df)
    df_final = process_protein_data(df)
    peptide_scores = pd.read_csv(peptide_scores_path)

    # Analyze top proteins
    results = []
    global_stats = defaultdict(int)
    unique_proteins = df_final["Proteins"].value_counts().index.tolist()

    print(
        f"Analyzing {len(unique_proteins)} unique proteins across {len(drugs)} drugs..."
    )

    for protein in tqdm(unique_proteins):
        protein_df = df_final[df_final["Proteins"] == protein]
        tqdm.write(f"Analyzing protein: {protein} ({len(protein_df)} occurrences)")
        for drug in drugs:
            drug_fc = peptide_scores[peptide_scores["drug"] == drug]["log_fold_change"]
            result_df, stats = analyze_drug(drug, protein_df, drug_fc)
            results.append(result_df)
            # Accumulate statistics
            for key, value in stats.items():
                global_stats[key] += value

    # Compile and save results
    combined_results = pd.concat(results, ignore_index=True)
    combined_results.to_csv(results_output_path, index=False)

    # Print summaries
    print_global_filter_summary(global_stats)
    print_summary_statistics(combined_results)


if __name__ == "__main__":
    main()
