import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm


if __name__ == "__main__":
    data_path_csv = "data/mq_variants_intensity_cleaned.csv"
    df = pd.read_csv(data_path_csv)

    treatment_cols = [col for col in df.columns if col.startswith("_dyn_")]
    treatment_series = df.set_index("Variant")[treatment_cols].stack()
    treatment_df = treatment_series.reset_index(level=1, name="intensity").rename(
        columns={"level_1": "condition"}
    )

    # Filter out zero values
    treatment_df_filtered = treatment_df[treatment_df["intensity"] != 0.0].copy()

    # Save just the intensity values to a simple CSV
    treatment_df_filtered["intensity"].to_csv(
        "data/all_intensities_distribution.txt", index=False, header=False
    )

    # Print information about filtered values
    total_points = len(treatment_df)
    filtered_points = len(treatment_df_filtered)
    print(
        f"\nFiltered {total_points - filtered_points} zero values out of {total_points} total points"
    )

    # Calculate and print statistics on filtered data
    stats_dict = {
        "Mean": np.mean(treatment_df_filtered["intensity"]),
        "Standard Deviation": np.std(treatment_df_filtered["intensity"]),
        "Median": np.median(treatment_df_filtered["intensity"]),
        "Q1 (25th percentile)": np.percentile(treatment_df_filtered["intensity"], 25),
        "Q3 (75th percentile)": np.percentile(treatment_df_filtered["intensity"], 75),
        "Min": np.min(treatment_df_filtered["intensity"]),
        "Max": np.max(treatment_df_filtered["intensity"]),
        "Skewness": stats.skew(treatment_df_filtered["intensity"]),
        "Kurtosis": stats.kurtosis(treatment_df_filtered["intensity"]),
    }

    print("\nDistribution Statistics (excluding zero values):")
    for stat_name, value in stats_dict.items():
        print(f"{stat_name}: {value:.2f}")

    # Create distribution plot with filtered data
    plt.figure(figsize=(10, 6))
    sns.histplot(data=treatment_df_filtered, x="intensity", kde=True)
    plt.title("Distribution of Intensity Values (excluding zeros)")
    plt.xlabel("Intensity")
    plt.ylabel("Count")

    # Add vertical lines for mean and median
    plt.axvline(stats_dict["Mean"], color="red", linestyle="--", label="Mean")
    plt.axvline(stats_dict["Median"], color="green", linestyle="--", label="Median")
    plt.legend()

    # Save the plot instead of showing it
    plt.savefig(
        "data/intensity_distribution_no_zeros.png", dpi=300, bbox_inches="tight"
    )
    plt.close()  # Close the figure to free memory

    # Calculate z-scores for all points
    z_scores = stats.zscore(treatment_df_filtered["intensity"])

    # Calculate two-sided p-values for all points
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

    # Count significant points (p < 0.05)
    significant_points = np.sum(p_values < 0.05)

    print("\nZ-score Analysis:")
    print(f"Total non-zero points analyzed: {len(treatment_df_filtered)}")
    print(f"Points with p < 0.05: {significant_points}")
    print(
        f"Percentage of significant points: {(significant_points / len(treatment_df_filtered)) * 100:.2f}%"
    )

    # Add z-scores and p-values to the DataFrame for reference
    treatment_df_filtered["z_score"] = z_scores
    treatment_df_filtered["p_value"] = p_values

    # Save the DataFrame to CSV
    treatment_df_filtered.to_csv("data/pval_intensities.csv", index=True)
