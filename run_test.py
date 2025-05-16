import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm


def get_drug_names(df: pd.DataFrame):
    """
    Extract sorted, unique drug names from intensity column headers.
    Expects columns like "_dyn_#DRUG 10nM.Tech replicate…".
    """
    pattern = r"_dyn_#(?P<drug>[^ ]+) \d+nM"
    drugs = df.columns.to_series().str.extract(pattern)["drug"].dropna().unique()
    return sorted(drugs)


def filter_protein_df(df: pd.DataFrame, protein: str):
    """
    Return rows of the DataFrame where the "Proteins" column contains the given protein.
    """
    # TODO: check if we need this
    # mask_single = df["Proteins"].str.count(";") == 0
    mask_protein = df["Proteins"].str.contains(protein)
    return df[mask_protein]


def collect_log_intensities(df: pd.DataFrame, cols: pd.Index) -> np.ndarray:
    """
    Take the given columns, stack non‐NA values, and return log2‐transformed intensities.
    """
    return (
        df[cols]
        .stack()  # drop NaN values
        .pipe(np.log2)  # log transform
        .values  # convert to numpy array
    )


def build_intensities(df: pd.DataFrame, drug: str):
    """
    For a single peptide-DF and drug:
      - control_log: log‐intensities for DMSO.Tech replicate columns
      - dose_logs: dict mapping each dose label → log‐intensities
    """
    cols = df.columns.to_series()
    base = cols.str.contains("_dyn_") & cols.str.contains(drug)

    ctrl_cols = cols[base & cols.str.contains("DMSO.Tech replicate")].index
    treat_cols = cols[base & cols.str.contains("nM.Tech replicate")].index

    dose_pattern = re.compile(rf"{drug} (\d+)nM")
    doses = sorted(
        {m.group(1) for col in treat_cols if (m := dose_pattern.search(col))}
    )

    control_log = collect_log_intensities(df, ctrl_cols)
    dose_logs = {
        dose: collect_log_intensities(
            df, treat_cols[treat_cols.str.contains(f"{drug} {dose}nM")]
        )
        for dose in doses
    }
    return control_log, dose_logs


def run_t_test(
    control: np.ndarray,
    dose_logs: dict[str, np.ndarray],
    alpha: float = 0.05,
):
    """
    Perform one‐sided Welch's t-test (alternative='less') of each dose vs control.
    Returns DataFrame with columns ['dose','t_statistic','p_value','significant'].
    """
    records = []
    for dose, vals in dose_logs.items():
        result = stats.ttest_ind(
            vals,
            control,
            equal_var=False,
            nan_policy="raise",
            alternative="less",
        )
        t_stat, p_val = result.statistic, result.pvalue
        if p_val != 0:
            records.append(
                {
                    "dose": dose,
                    "t_statistic": t_stat,
                    "p_value": p_val,
                    "significant": p_val < alpha,
                }
            )

    if not records:
        return None

    return (
        pd.DataFrame.from_records(records).sort_values("p_value").reset_index(drop=True)
    )


def process_protein(protein: str, df: pd.DataFrame, drugs: list[str]):
    """Run all drugs for one peptide, return DataFrame of t‐test results."""
    subdf = filter_protein_df(df, protein)
    if subdf.empty:
        return None

    rows = []
    for drug in drugs:
        ctrl, doses = build_intensities(subdf, drug)
        if ctrl.size == 0 or all(v.size == 0 for v in doses.values()):
            continue
        tdf = run_t_test(ctrl, doses)
        if tdf is not None:
            tdf.insert(0, "protein", protein)
            tdf.insert(1, "drug", drug)
            rows.append(tdf)

    if not rows:
        return None

    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    data_path_csv = "data/mq_variants_intensity_cleaned.csv"
    df = pd.read_csv(data_path_csv)

    treatment_cols = [col for col in df.columns if col.startswith("_dyn_")]
    treatment_series = df.set_index("Variant")[treatment_cols].stack()
    treatment_df = treatment_series.reset_index(level=1, name="intensity").rename(
        columns={"level_1": "condition"}
    )
    
    # Filter out zero values
    treatment_df_filtered = treatment_df[treatment_df['intensity'] != 0.0].copy()
    
    # Save just the intensity values to a simple CSV
    treatment_df_filtered['intensity'].to_csv('data/all_intensities_distribution.csv', index=False, header=False)
    
    # Print information about filtered values
    total_points = len(treatment_df)
    filtered_points = len(treatment_df_filtered)
    print(f"\nFiltered {total_points - filtered_points} zero values out of {total_points} total points")
    
    # Calculate and print statistics on filtered data
    stats_dict = {
        'Mean': np.mean(treatment_df_filtered['intensity']),
        'Standard Deviation': np.std(treatment_df_filtered['intensity']),
        'Median': np.median(treatment_df_filtered['intensity']),
        'Q1 (25th percentile)': np.percentile(treatment_df_filtered['intensity'], 25),
        'Q3 (75th percentile)': np.percentile(treatment_df_filtered['intensity'], 75),
        'Min': np.min(treatment_df_filtered['intensity']),
        'Max': np.max(treatment_df_filtered['intensity']),
        'Skewness': stats.skew(treatment_df_filtered['intensity']),
        'Kurtosis': stats.kurtosis(treatment_df_filtered['intensity'])
    }
    
    print("\nDistribution Statistics (excluding zero values):")
    for stat_name, value in stats_dict.items():
        print(f"{stat_name}: {value:.2f}")
    
    # Create distribution plot with filtered data
    plt.figure(figsize=(10, 6))
    sns.histplot(data=treatment_df_filtered, x='intensity', kde=True)
    plt.title('Distribution of Intensity Values (excluding zeros)')
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    
    # Add vertical lines for mean and median
    plt.axvline(stats_dict['Mean'], color='red', linestyle='--', label='Mean')
    plt.axvline(stats_dict['Median'], color='green', linestyle='--', label='Median')
    plt.legend()
    
    # Save the plot instead of showing it
    plt.savefig('data/intensity_distribution_no_zeros.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    # Calculate z-scores for all points
    z_scores = stats.zscore(treatment_df_filtered['intensity'])
    
    # Calculate two-sided p-values for all points
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    
    # Count significant points (p < 0.05)
    significant_points = np.sum(p_values < 0.05)
    
    print("\nZ-score Analysis:")
    print(f"Total non-zero points analyzed: {len(treatment_df_filtered)}")
    print(f"Points with p < 0.05: {significant_points}")
    print(f"Percentage of significant points: {(significant_points/len(treatment_df_filtered))*100:.2f}%")
    
    # Add z-scores and p-values to the DataFrame for reference
    treatment_df_filtered['z_score'] = z_scores
    treatment_df_filtered['p_value'] = p_values
    
    # Save the DataFrame to CSV
    treatment_df_filtered.to_csv('data/pval_intensities.csv', index=True)


    