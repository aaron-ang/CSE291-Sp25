import pandas as pd
import re

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
pval_df = pd.read_csv('data/variant_scores.csv')
filtered_pval_df = pval_df[pval_df['Variant'].isin(variant_to_protein.keys())]
drug_filter = filtered_pval_df['condition'].str.contains('|'.join(drugs_of_interest), case=False, na=False)
final_pval_df = filtered_pval_df[drug_filter]

def extract_concentration(condition):
    """Extract concentration in nM from condition string."""
    match = re.search(r'(\d+)nM', condition)
    return int(match.group(1)) if match else None

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
            print(f"- {row['Variant']}: p-value = {row['p_value']:.6f}, intensity = {row['log_fold_change']:.6f}")
    
    return total_variants, significant_variants, percentage

print("\n=== Analysis of Significant Perturbations at All Concentrations (p < 0.05) ===")

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
    
    # Get all available concentrations for this drug
    drug_data = final_pval_df[
        (final_pval_df['Variant'].isin(protein_variants)) & 
        (final_pval_df['condition'].str.contains(drug, case=False, na=False))
    ]
    concentrations = sorted(set(drug_data['condition'].apply(extract_concentration).dropna()))
    
    print(f"Available concentrations: {concentrations}")
    
    # Analyze all concentrations
    for concentration in concentrations:
        total, sig, perc = analyze_concentration(protein, drug, protein_variants, final_pval_df, concentration)
        
        # Store results for summary
        results.append({
            'protein': protein.split('|')[1],  # Extract protein name without sp| prefix
            'drug': drug,
            'response': mono_row['Sign'],
            'concentration': concentration,
            'total_variants': total,
            'sig_variants': sig,
            'percentage': perc
        })

# Print summary table
print("\n=== Summary of Concentration Effects ===")
print("\nProtein-Drug Combinations (sorted by protein and drug):")
print(f"{'Protein':<10} {'Drug':<20} {'Response':<12} {'Conc.(nM)':<10} {'% Perturbed':<12}")
print("-" * 70)

# Sort results by protein, drug, and concentration
results.sort(key=lambda x: (x['protein'], x['drug'], x['concentration']))

# Group results by protein and drug
current_protein = None
current_drug = None

for result in results:
    # Print headers when protein or drug changes
    if result['protein'] != current_protein or result['drug'] != current_drug:
        print("-" * 70)
        current_protein = result['protein']
        current_drug = result['drug']
    
    print(f"{result['protein']:<10} {result['drug']:<20} {result['response']:<12} "
          f"{result['concentration']:<10} {result['percentage']:>8.1f}%")

# Print interesting findings
print("\nKey Findings:")

# 1. Find maximum perturbation for each protein-drug combo
print("\n1. Maximum perturbation for each protein-drug combination:")
for protein in set(r['protein'] for r in results):
    protein_results = [r for r in results if r['protein'] == protein]
    for drug in set(r['drug'] for r in protein_results):
        drug_results = [r for r in protein_results if r['drug'] == drug]
        max_result = max(drug_results, key=lambda x: x['percentage'])
        if max_result['percentage'] > 0:
            print(f"   - {protein} with {drug}: {max_result['percentage']:.1f}% at {max_result['concentration']}nM")

# 2. Find strongest concentration dependence
print("\n2. Strongest concentration dependence (largest % change between min and max):")
for protein in set(r['protein'] for r in results):
    protein_results = [r for r in results if r['protein'] == protein]
    for drug in set(r['drug'] for r in protein_results):
        drug_results = [r for r in protein_results if r['drug'] == drug]
        if len(drug_results) > 1:
            min_perc = min(r['percentage'] for r in drug_results)
            max_perc = max(r['percentage'] for r in drug_results)
            change = max_perc - min_perc
            if change > 0:
                print(f"   - {protein} with {drug}: {change:.1f}% change "
                      f"(from {min_perc:.1f}% to {max_perc:.1f}%)")

# 3. Most stable responses across concentrations
print("\n3. Most stable responses (smallest variation across concentrations):")
stable_responses = []
for protein in set(r['protein'] for r in results):
    protein_results = [r for r in results if r['protein'] == protein]
    for drug in set(r['drug'] for r in protein_results):
        drug_results = [r for r in protein_results if r['drug'] == drug]
        if len(drug_results) > 1:
            min_perc = min(r['percentage'] for r in drug_results)
            max_perc = max(r['percentage'] for r in drug_results)
            change = max_perc - min_perc
            stable_responses.append((protein, drug, change))

for protein, drug, change in sorted(stable_responses, key=lambda x: x[2])[:5]:
    print(f"   - {protein} with {drug}: {change:.1f}% variation")

# Save the final filtered dataset
final_pval_df.to_csv('data/pval_intensities_filtered_by_monotonic_drugs.csv', index=False)
print("\nFinal filtered p-value intensities dataset saved to 'data/pval_intensities_filtered_by_monotonic_drugs.csv'")
