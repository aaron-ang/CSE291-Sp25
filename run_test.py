import re
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    mask_single = df["Proteins"].str.count(";") == 0  # TODO: check if we need this
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
        records.append(
            {
                "dose": dose,
                "t_statistic": t_stat,
                "p_value": p_val,
                "significant": p_val < alpha,
            }
        )

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
        tdf["Peptide"] = protein
        tdf["Drug"] = drug
        rows.append(tdf)

    return pd.concat(rows, ignore_index=True) if rows else None


if __name__ == "__main__":
    data_path_csv = "data/mq_variants_intensity_cleaned.csv"
    df = pd.read_csv(data_path_csv)
    drugs = get_drug_names(df)
    proteins = df["Proteins"].str.split(";").explode().unique()
    print(f"Found {len(drugs)} drugs and {len(proteins)} proteins")

    all_results = []

    # # Single core processing
    # for protein in tqdm(proteins, desc="Proteins"):
    #     res = process_protein(protein, df, drugs)
    #     if res is not None:
    #         all_results.append(res)

    # Multi‐core processing
    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(process_protein, p, df, drugs) for p in proteins]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Peptides"):
            res = fut.result()
            if res is not None:
                all_results.append(res)

    results = pd.concat(all_results, ignore_index=True)
    sig = results[results["significant"]]
    print(f"Performed {len(results)} tests; {len(sig)} significant")
    print(sig.to_string(index=False))
    sig.to_csv("data/significant_t_tests.csv", index=False)
