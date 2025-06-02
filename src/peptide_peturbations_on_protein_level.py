import pandas as pd

# Part 1: Process monotonic protein drug mapping data first (we need this list of proteins)
print("=== Processing Monotonic Protein Drug Mapping Data ===")
# Read the monotonic mapping CSV file
monotonic_df = pd.read_csv('data/monotonic_protein_drug_mapping.csv')

# Filter rows where NumConcentrations is 3 or greater
filtered_monotonic_df = monotonic_df[monotonic_df['NumConcentrations'] >= 3]

# Get unique drugs and proteins from monotonic data
drugs_of_interest = filtered_monotonic_df['Drug'].unique()
proteins_of_interest = filtered_monotonic_df['Protein'].unique()

# Print protein-drug combinations from monotonic data
print("\nProtein-Drug combinations with monotonic response:")
for _, row in filtered_monotonic_df.iterrows():
    print(f"- {row['Protein']} with {row['Drug']} ({row['Sign']} response)")

# Part 2: Process variants data to create mapping
variants_df = pd.read_csv('data/mq_variants_intensity_cleaned.csv')
single_protein_variants_df = variants_df[~variants_df['Proteins'].str.contains(';', na=False)]
final_filtered_df = single_protein_variants_df[
    single_protein_variants_df['Proteins'].isin(proteins_of_interest)
]
variant_to_protein = dict(zip(final_filtered_df['Variant'], final_filtered_df['Proteins']))

# Part 3: Process p-value intensities data
pval_df = pd.read_csv('data/pval_intensities.csv')
filtered_pval_df = pval_df[pval_df['Variant'].isin(variant_to_protein.keys())]
drug_filter = filtered_pval_df['condition'].str.contains('|'.join(drugs_of_interest), case=False, na=False)
final_pval_df = filtered_pval_df[drug_filter]

def analyze_concentration(protein, drug, protein_variants, final_pval_df, concentration):
    # Filter p-value data for this drug and these variants at the specified concentration
    drug_data = final_pval_df[
        (final_pval_df['Variant'].isin(protein_variants)) & 
        (final_pval_df['condition'].str.contains(drug, case=False, na=False)) &
        (final_pval_df['condition'].str.contains(f'{concentration}nM', case=False, na=False))
    ]
    
    total_variants = len(drug_data)
    significant_variants = len(drug_data[drug_data['p_value'] < 0.05])
    percentage = (significant_variants / total_variants * 100) if total_variants > 0 else 0
    
    print(f"\n{concentration}nM concentration:")
    print(f"Total variants measured: {total_variants}")
    print(f"Significantly perturbed variants (p < 0.05): {significant_variants}")
    print(f"Percentage significantly perturbed: {percentage:.2f}%")
    
    if significant_variants > 0:
        print("\nSignificantly perturbed variants:")
        significant_data = drug_data[drug_data['p_value'] < 0.05].sort_values('p_value')
        for _, row in significant_data.iterrows():
            print(f"- {row['Variant']}: p-value = {row['p_value']:.6f}, intensity = {row['intensity']:.6f}")
    
    return total_variants, significant_variants, percentage

print("\n=== Analysis of Significant Perturbations at High Concentrations (p < 0.05) ===")
print("For each protein-drug combination, showing perturbations at 3000nM and 30000nM")
print("")

# Store results for summary
results = []

# Analyze each protein-drug combination
for _, mono_row in filtered_monotonic_df.iterrows():
    protein = mono_row['Protein']
    drug = mono_row['Drug']
    
    # Get variants for this protein
    protein_variants = [var for var, prot in variant_to_protein.items() if prot == protein]
    
    print(f"\nProtein: {protein}")
    print(f"Drug: {drug}")
    print(f"Response type: {mono_row['Sign']}")
    
    # Analyze both concentrations
    total_3000, sig_3000, perc_3000 = analyze_concentration(protein, drug, protein_variants, final_pval_df, 3000)
    total_30000, sig_30000, perc_30000 = analyze_concentration(protein, drug, protein_variants, final_pval_df, 30000)
    
    # Store results for summary
    results.append({
        'protein': protein.split('|')[1],  # Extract protein name without sp| prefix
        'drug': drug,
        'response': mono_row['Sign'],
        'total_3000': total_3000,
        'sig_3000': sig_3000,
        'perc_3000': perc_3000,
        'total_30000': total_30000,
        'sig_30000': sig_30000,
        'perc_30000': perc_30000,
        'perc_change': perc_30000 - perc_3000
    })

# Print summary table
print("\n=== Summary of Concentration Effects ===")
print("\nProtein-Drug Combinations (sorted by absolute change in perturbation percentage):")
print(f"{'Protein':<10} {'Drug':<20} {'Response':<12} {'3000nM %':>8} {'30000nM %':>9} {'Change':>8}")
print("-" * 70)

# Sort results by absolute percentage change
results.sort(key=lambda x: abs(x['perc_change']), reverse=True)

for result in results:
    print(f"{result['protein']:<10} {result['drug']:<20} {result['response']:<12} "
          f"{result['perc_3000']:>7.1f}% {result['perc_30000']:>8.1f}% {result['perc_change']:>7.1f}%")

# Print interesting findings
print("\nKey Findings:")
print("1. Largest increases in perturbation from 3000nM to 30000nM:")
increases = [r for r in results if r['perc_change'] > 0]
for r in sorted(increases, key=lambda x: x['perc_change'], reverse=True)[:3]:
    print(f"   - {r['protein']} with {r['drug']}: {r['perc_change']:.1f}% increase")

print("\n2. Largest decreases in perturbation from 3000nM to 30000nM:")
decreases = [r for r in results if r['perc_change'] < 0]
for r in sorted(decreases, key=lambda x: x['perc_change'])[:3]:
    print(f"   - {r['protein']} with {r['drug']}: {abs(r['perc_change']):.1f}% decrease")

print("\n3. Most stable responses (smallest absolute change):")
stable = sorted(results, key=lambda x: abs(x['perc_change']))[:3]
for r in stable:
    print(f"   - {r['protein']} with {r['drug']}: {abs(r['perc_change']):.1f}% change")

# Save the final filtered dataset
final_pval_df.to_csv('data/pval_intensities_filtered_by_monotonic_drugs.csv', index=False)
print("\nFinal filtered p-value intensities dataset saved to 'data/pval_intensities_filtered_by_monotonic_drugs.csv'")
