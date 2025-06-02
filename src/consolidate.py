import pandas as pd
import scipy.stats as stats

# Load and preprocess data
df = pd.read_csv("data/drug_distribution_stats.csv")
df["Concentration_nM"] = df["Concentration"].str.replace("nM", "").astype(int)

# Analyze each protein-drug pair for monotonic trends and sign consistency
results = []
for (protein, drug), group in df.groupby(["Protein", "Drug"]):
    means = group["Mean"].values
    concs = group["Concentration_nM"].values

    # Calculate Spearman correlation
    rho = stats.spearmanr(concs, means)[0] if len(means) > 1 else 0

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

# Filter for strong monotonic trend (|ρ| ≥ 0.7)
monotonic_df = res_df[res_df["SpearmanRho"].abs() >= 0.7]
non_monotonic_df = res_df[res_df["SpearmanRho"].abs() < 0.7]

# Filter for consistent sign (non-mixed)
final_df = monotonic_df[monotonic_df["Sign"] != "mixed"]
mixed_sign_df = monotonic_df[monotonic_df["Sign"] == "mixed"]

# Print filtering results
print(
    f"Pairs filtered due to insufficient monotonic trend (|ρ| < 0.7): {len(non_monotonic_df)}"
)
print(f"Pairs filtered due to inconsistent sign: {len(mixed_sign_df)}")
print(
    f"Remaining pairs after all filters: {len(final_df)} ({len(final_df) / total_rows:.2%})"
)

# Save filtered results
final_df.to_csv("data/monotonic_protein_drug_mapping.csv", index=False)

# Prepare filtered data for peptide analysis
filtered_df = pd.concat([non_monotonic_df, mixed_sign_df]).sort_values(
    ["Protein", "Drug"]
)
variant_drug_df = pd.read_csv("data/monotonic_variant_drug_combos.csv")

print("\n=== Peptide-Level Analysis for Filtered Protein-Drug Pairs ===")
print(f"Total filtered protein-drug pairs: {len(filtered_df)}")
print(f"Total peptide-drug combinations: {len(variant_drug_df)}")

# Find peptides for each filtered protein-drug pair
protein_drug_to_peptides = {}
for _, row in filtered_df.iterrows():
    protein, drug = row["Protein"], row["Drug"]

    matching_peptides = variant_drug_df[
        (variant_drug_df["Protein"] == protein) & (variant_drug_df["Drug"] == drug)
    ]

    if len(matching_peptides) > 0:
        protein_drug_to_peptides[(protein, drug)] = matching_peptides[
            "Variant"
        ].tolist()

# Print summary statistics
pairs_with_peptides = len(protein_drug_to_peptides)
total_peptides = sum(len(peptides) for peptides in protein_drug_to_peptides.values())

print(
    f"Filtered pairs with identified peptides: {pairs_with_peptides} ({pairs_with_peptides / len(filtered_df):.2%})"
)
print(f"Total peptides identified: {total_peptides}")

# Create a detailed mapping file that shows peptides for filtered protein-drug pairs
peptide_drug_mapping = []
for (protein, drug), peptides in protein_drug_to_peptides.items():
    # Get all matching records for this protein-drug combination
    matching_records = variant_drug_df[
        (variant_drug_df["Protein"] == protein) & (variant_drug_df["Drug"] == drug)
    ]
    # Create mapping entries for each peptide
    for _, record in matching_records.iterrows():
        mapping_row = {
            "Protein": record["Protein"],
            "Variant": record["Variant"],
            "Drug": record["Drug"],
            "Intensities": record["Intensities"],
        }
        peptide_drug_mapping.append(mapping_row)

# Save detailed peptide mapping for filtered pairs
peptide_mapping_df = pd.DataFrame(peptide_drug_mapping)
peptide_mapping_df.to_csv("data/monotonic_peptide_drug_mapping.csv", index=False)
