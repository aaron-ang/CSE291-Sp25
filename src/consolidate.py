import pandas as pd
import numpy as np
import scipy.stats as stats

df = pd.read_csv("data/drug_distribution_stats.csv")

# Convert "30nM", "1000nM", etc. into a numeric column (float, in nM)
df["Concentration_nM"] = df["Concentration"].str.replace("nM", "").astype(int)

# Iterate per (Protein, Drug), check sign‐consistency and compute Spearman ρ
results = []
for (protein, drug), group in df.groupby(["Protein", "Drug"]):
    means = group["Mean"].values
    concs = group["Concentration_nM"].values

    # Compute Spearman correlation (ρ) between concentration and mean
    if len(means) > 1:
        rho, _ = stats.spearmanr(concs, means)
    else:
        # Only one concentration ⇒ ρ = NaN
        rho = np.nan

    # Determine sign consistency
    if (means >= 0).all():
        sign_label = "non-negative"
    elif (means <= 0).all():
        sign_label = "non-positive"
    else:
        sign_label = "mixed"

    results.append(
        {
            "Protein": protein,
            "Drug": drug,
            "NumConcentrations": len(means),
            "SpearmanRho": rho,
            "Sign": sign_label,
        }
    )

res_df = pd.DataFrame(results)
total_rows = len(res_df)
print(f"Total protein-drug pairs: {total_rows}")

# Filter down to those with a strong monotonic trend: |ρ| ≥ 0.7
monotonic_df = res_df[res_df["SpearmanRho"].abs() >= 0.7]
non_monotonic_df = res_df[res_df["SpearmanRho"].abs() < 0.7]
print(
    f"Pairs filtered due to insufficient monotonic trend (|ρ| < 0.7): {len(non_monotonic_df)}"
)
print(
    f"Remaining pairs after monotonic filter: {len(monotonic_df)} ({len(monotonic_df) / total_rows:.2%})"
)

# Filter to keep only protein-drug pairs with consistent sign
final_df = monotonic_df[monotonic_df["Sign"] != "mixed"]
mixed_sign_df = monotonic_df[monotonic_df["Sign"] == "mixed"]
print(f"Pairs filtered due to inconsistent sign: {len(mixed_sign_df)}")
print(
    f"Remaining pairs after all filters: {len(final_df)} ({len(final_df) / total_rows:.2%})"
)

final_df.to_csv("data/monotonic_protein_drug_mapping.csv", index=False)

# Cross-reference filtered pairs with variant-level data
filtered_df = pd.concat([non_monotonic_df, mixed_sign_df])
variant_drug_df = pd.read_csv("data/monotonic_variant_drug_combos.csv")

print("\n=== Peptide-Level Analysis for Filtered Protein-Drug Pairs ===")
print(f"Total filtered protein-drug pairs: {len(filtered_df)}")
print(f"Total peptide-drug combinations: {len(variant_drug_df)}")

protein_drug_to_peptides = {}

# For each filtered protein-drug pair, find corresponding peptides
for _, row in filtered_df.iterrows():
    protein = row["Protein"]
    drug = row["Drug"]
    key = (protein, drug)

    # Find all peptides for this protein-drug combination
    matching_peptides = variant_drug_df[
        (variant_drug_df["Protein"] == protein) & (variant_drug_df["Drug"] == drug)
    ]

    count = len(matching_peptides)
    if count > 0:
        protein_drug_to_peptides[key] = matching_peptides["Variant"].tolist()

# Show summary
pairs_with_peptides = sum(
    1 for peptides in protein_drug_to_peptides.values() if peptides
)
print(
    f"Filtered pairs with identified peptides: {pairs_with_peptides} ({pairs_with_peptides / len(filtered_df):.2%})"
)

total_peptides = sum(len(peptides) for peptides in protein_drug_to_peptides.values())
print(f"Total peptides identified across filtered protein-drug pairs: {total_peptides}")

# Create a detailed mapping file that shows peptides for filtered protein-drug pairs
peptide_drug_mapping = []
for key, peptides in protein_drug_to_peptides.items():
    protein, drug = key
    for peptide in peptides:
        peptide_data = (
            variant_drug_df[
                (variant_drug_df["Protein"] == protein)
                & (variant_drug_df["Drug"] == drug)
                & (variant_drug_df["Variant"] == peptide)
            ]
            .iloc[0]
            .to_dict()
        )
        # Combine protein-drug data with peptide data
        mapping_row = {
            "Protein": protein,
            "Variant": peptide,
            "Drug": drug,
            "Intensities": peptide_data["Intensities"],
        }
        peptide_drug_mapping.append(mapping_row)

# Save detailed peptide mapping for filtered pairs
peptide_mapping_df = pd.DataFrame(peptide_drug_mapping)
peptide_mapping_df.to_csv("data/monotonic_peptide_drug_mapping.csv", index=False)
