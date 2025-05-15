import re

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
    return drugs


def build_column_map(df: pd.DataFrame, drugs: np.ndarray):
    """
    Build a map from (drug, dose_label) or (drug, 'control') to lists of column names.
    """
    column_map = {}
    cols = df.columns.to_series()
    dose_pattern = re.compile(r" (\d+)nM")

    for drug in drugs:
        drug_cols = cols[cols.str.contains(f"_dyn_#{drug}")]
        if drug_cols.empty:
            continue

        column_map[drug] = {}

        # Control columns
        ctrl_cols = drug_cols[drug_cols.str.contains("DMSO.Tech replicate")].index
        if not ctrl_cols.empty:
            column_map[drug]["control"] = ctrl_cols

        # Treatment columns
        treat_cols_series = drug_cols[drug_cols.str.contains("nM.Tech replicate")]
        doses = sorted(
            {
                m.group(1)
                for col in treat_cols_series.index
                if (m := dose_pattern.search(col))
            }
        )
        for dose in doses:
            dose_cols = treat_cols_series[
                treat_cols_series.str.contains(f" {dose}nM")
            ].index
            if not dose_cols.empty:
                column_map[drug][dose] = dose_cols

    return column_map


def filter_protein_df(df: pd.DataFrame, protein: str):
    """
    Return rows of the DataFrame where the "Proteins" column contains the given protein.
    """
    # TODO: Check if this is needed
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


def build_intensities(df: pd.DataFrame, drug: str, column_map: dict):
    """
    For a single peptide-DF and drug:
      - control_log: log‐intensities for DMSO.Tech replicate columns
      - dose_logs: dict mapping each dose label → log‐intensities
    Uses the pre-calculated column_map to get column names.
    """
    drug_map = column_map[drug]
    control_cols = drug_map["control"]
    control_log = collect_log_intensities(df, control_cols)
    dose_logs = {}
    doses = sorted(d for d in drug_map if d != "control")
    dose_logs = {dose: collect_log_intensities(df, drug_map[dose]) for dose in doses}
    return control_log, dose_logs


def run_t_test(
    control: np.ndarray,
    dose_logs: dict[str, np.ndarray],
    alpha: float = 0.05,
):
    """
    Perform one‐sided Welch’s t-test (alternative='less') of each dose vs control.
    Returns DataFrame with columns ['dose','t_statistic','p_value','significant'].
    """
    records = []
    for dose, vals in dose_logs.items():
        t_stat, p_val = stats.ttest_ind(
            vals,
            control,
            equal_var=False,
            nan_policy="raise",
            alternative="less",
        )
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


def process_protein(
    protein: str,
    df: pd.DataFrame,
    drugs: np.ndarray,
    column_map: dict,
):
    """Run all drugs for one peptide, return DataFrame of t‐test results."""
    subdf = filter_protein_df(df, protein)
    if subdf.empty:
        return None

    rows = []
    for drug in drugs:
        ctrl, doses = build_intensities(subdf, drug, column_map)
        tdf = run_t_test(ctrl, doses)
        if tdf:
            tdf.insert(0, "protein", protein)
            tdf.insert(1, "drug", drug)
            rows.append(tdf)

    if not rows:
        return None

    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    data_path_csv = "data/mq_variants_intensity_cleaned.csv"
    print(f"Reading data from {data_path_csv}...")
    df = pd.read_csv(data_path_csv)

    drugs = get_drug_names(df)
    proteins = df["Proteins"].str.split(";").explode().dropna().unique()
    print(f"Found {len(drugs)} drugs and {len(proteins)} proteins")

    print("Building column map...")
    column_map = build_column_map(df, drugs)

    all_results = []
    for protein in tqdm(proteins, desc="Proteins"):
        res = process_protein(protein, df, drugs, column_map)
        if res and not res.empty:
            all_results.append(res)

    print("Concatenating results...")
    if all_results:
        results = pd.concat(all_results, ignore_index=True)
        sig = results[results["significant"]]
        print(f"Collected {len(results)} tests; {len(sig)} significant")
        sig.to_csv("data/significant_t_tests.csv", index=False)
    else:
        print("No significant results found or no valid data processed.")
