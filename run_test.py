import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    treatment_cols = [col for col in df.columns if col.startswith("_dyn_")]

    # Group by Unmod variant and drug
    treatment_series = df.set_index("Unmod variant")[treatment_cols].stack()
    treatment_df = treatment_series.reset_index(level=1, name="log_fold_change").rename(
        columns={"level_1": "drug"}
    )

    # Remove concentration from drug names
    treatment_df["drug"] = treatment_df["drug"].str.split().str[0].str.split("#").str[1]

    return treatment_df


def filter_and_calculate_stats(df):
    # Filter out zero values
    total_points = len(df)
    filtered_df = df[df["log_fold_change"] != 0.0].copy()
    filtered_points = len(filtered_df)

    print(
        f"\nFiltered {total_points - filtered_points} zero values out of {total_points} total points"
    )

    # Calculate statistics
    stats_dict = {
        "Mean": np.mean(filtered_df["log_fold_change"]),
        "Standard Deviation": np.std(filtered_df["log_fold_change"]),
        "Median": np.median(filtered_df["log_fold_change"]),
        "Q1 (25th percentile)": np.percentile(filtered_df["log_fold_change"], 25),
        "Q3 (75th percentile)": np.percentile(filtered_df["log_fold_change"], 75),
        "Min": np.min(filtered_df["log_fold_change"]),
        "Max": np.max(filtered_df["log_fold_change"]),
        "Skewness": stats.skew(filtered_df["log_fold_change"]),
        "Kurtosis": stats.kurtosis(filtered_df["log_fold_change"]),
    }

    print("\nDistribution Statistics (excluding zero values):")
    for stat_name, value in stats_dict.items():
        print(f"{stat_name}: {value:.2f}")

    return filtered_df, stats_dict


def calculate_significance(df, alpha=0.01):
    z_scores = stats.zscore(df["log_fold_change"])

    # Calculate two-sided p-values
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

    # Get significant fold changes
    significant_fc = df["log_fold_change"][p_values < alpha]

    # Count significant points
    significant_points = np.sum(p_values < alpha)

    print("\nZ-score Analysis:")
    print(f"Total non-zero points analyzed: {len(df)}")
    print(f"Points with p < {alpha}: {significant_points}")
    print(
        f"Percentage of significant points: {(significant_points / len(df)) * 100:.2f}%"
    )

    # Add z-scores and p-values to the DataFrame
    df["z_score"] = z_scores
    df["p_value"] = p_values

    return df, significant_fc


def create_distribution_plot(df, stats_dict, significant_fc, alpha, output_path: str):
    # Create a histogram of log fold changes
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="log_fold_change", kde=True)
    # Add rug plot for significant points
    sns.rugplot(x=significant_fc, color="purple", label=f"p < {alpha}")
    plt.xlabel("Log Fold Change")
    plt.ylabel("Count")
    plt.title("Distribution of Log Fold Change")

    # Add mean and median lines
    plt.axvline(stats_dict["Mean"], color="red", linestyle="--", label="Mean")
    plt.axvline(stats_dict["Median"], color="green", linestyle="--", label="Median")
    plt.legend()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Create a volcano plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df["log_fold_change"], -np.log10(df["p_value"]), marker=".", alpha=0.3)
    # Add threshold lines
    plt.axhline(-np.log10(alpha), color="red", linestyle="--", label=f"p = {alpha}")
    plt.axvline(x=-1, color="g", linestyle="--", label="log2FC = -1 and 1")
    plt.axvline(x=1, color="g", linestyle="--")

    plt.xlabel("log2(Fold Change)")
    plt.ylabel("-log10(p-value)")
    plt.legend()

    plt.savefig(
        output_path.replace("_distribution.png", "_volcano.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_results(df: pd.DataFrame, output_path):
    df.to_csv(output_path)
    print(f"\nFiltered data saved to {output_path}")


def main():
    data_path_csv = "data/mq_variants_intensity_cleaned.csv"
    plot_output_path = "data/log_fc_distribution.png"
    results_output_path = "data/variant_scores.csv"
    alpha = 0.01

    treatment_df = load_and_preprocess_data(data_path_csv)

    filtered_df, stats_dict = filter_and_calculate_stats(treatment_df)

    result_df, significant_fc = calculate_significance(filtered_df, alpha)

    create_distribution_plot(
        result_df,
        stats_dict,
        significant_fc,
        alpha,
        plot_output_path,
    )

    save_results(result_df, results_output_path)


if __name__ == "__main__":
    main()
