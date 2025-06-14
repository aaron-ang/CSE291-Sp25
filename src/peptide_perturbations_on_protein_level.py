import os

import pandas as pd
import re
import matplotlib.pyplot as plt

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
    significant_variants = len(drug_data[drug_data['p_value'] < 0.01])
    percentage = (significant_variants / total_variants * 100) if total_variants > 0 else 0
    
    print(f"\n{concentration}nM concentration:")
    print(f"Total variants measured: {total_variants}")
    print(f"Significantly perturbed variants (p < 0.01): {significant_variants}")
    print(f"Percentage significantly perturbed: {percentage:.2f}%")
    
    if significant_variants > 0:
        print("\nSignificantly perturbed variants:")
        significant_data = drug_data[drug_data['p_value'] < 0.01].sort_values('p_value')
        for _, row in significant_data.iterrows():
            print(f"- {row['Variant']}: p-value = {row['p_value']:.6f}, intensity = {row['log_fold_change']:.6f}")
    
    return total_variants, significant_variants, percentage

print("\n=== Analysis of Significant Perturbations at All Concentrations (p < 0.01) ===")

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

# Create plots for each protein-drug combination
os.makedirs('plots', exist_ok=True)
unique_proteins = set(r['protein'] for r in results)
unique_drugs = set(r['drug'] for r in results)

for protein in unique_proteins:
    protein_results = [r for r in results if r['protein'] == protein]
    for drug in unique_drugs:
        drug_results = [r for r in protein_results if r['drug'] == drug]
        if drug_results:  # Only create plot if we have data for this combination
            # Sort by concentration
            drug_results.sort(key=lambda x: x['concentration'])
            
            # Create the plot
            plt.figure(figsize=(12, 8))  # Made figure larger to accommodate labels
            
            # Filter concentrations with at least 10 peptides
            valid_concentrations = []
            valid_percentages = []
            valid_variants = []
            
            for concentration in concentrations:
                # Get all peptides measured at this concentration
                drug_data = final_pval_df[
                    (final_pval_df['Variant'].isin(protein_variants)) & 
                    (final_pval_df['condition'].str.contains(drug, case=False, na=False)) &
                    (final_pval_df['condition'].str.contains(f'{concentration}nM', case=False, na=False))
                ]
                
                # Only include if we have at least 10 peptides
                if len(drug_data) >= 10:
                    # Calculate percentage directly from the data
                    sig_drug_data = drug_data[drug_data['p_value'] < 0.01]
                    percentage = (len(sig_drug_data) / len(drug_data)) * 100
                    
                    valid_concentrations.append(concentration)
                    valid_percentages.append(percentage)
                    valid_variants.append(sig_drug_data)  # Store only significant variants
            
            # Only plot if we have valid concentrations
            if valid_concentrations:
                plt.plot(
                    valid_concentrations,
                    valid_percentages,
                    'o-',  # Line with dots
                    linewidth=2,
                    markersize=8
                )
                
                # Add labels and title
                plt.xlabel('Concentration (nM)')
                plt.ylabel('Perturbed Peptides (%)')
                plt.title(f'{protein} response to {drug}')
                
                # Add grid
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Use log scale for x-axis since concentrations vary by orders of magnitude
                plt.xscale('log')
                
                # Set y-axis limits from 0 to max percentage + some padding
                plt.ylim(0, max(valid_percentages) * 1.1)
                
                # Add peptide labels for each concentration
                for i, (concentration, y_val, sig_drug_data) in enumerate(zip(valid_concentrations, valid_percentages, valid_variants)):
                    # Only add labels if there are perturbations (y_val > 0)
                    if y_val > 0 and not sig_drug_data.empty:
                        # Create label text with peptide names
                        peptide_labels = [f"{row['Variant']} (p={row['p_value']:.2e})" 
                                        for _, row in sig_drug_data.iterrows()]
                        
                        # Calculate vertical spacing for labels
                        vertical_spacing = 10.0  # Increased spacing between labels
                        total_height = len(peptide_labels) * vertical_spacing
                        
                        # Start position for the first label
                        start_y = y_val + (total_height / 2)
                        
                        for j, label in enumerate(peptide_labels):
                            # Calculate vertical position for this label
                            label_y = start_y - (j * vertical_spacing)
                            
                            # Add text annotation
                            plt.annotate(label,
                                       xy=(concentration, y_val),
                                       xytext=(10, label_y - y_val),  # Offset from data point
                                       textcoords='offset points',
                                       ha='left',
                                       va='center',
                                       fontsize=8,
                                       bbox=dict(boxstyle='round,pad=0.5',
                                               fc='yellow',
                                               alpha=0.3))
                
                # Adjust layout to prevent label cutoff
                plt.tight_layout()
                
                # Save the plot
                plt.savefig(f'plots/{protein}_{drug}_perturbation.png', bbox_inches='tight', dpi=300)
            plt.close()

print("\nPlots have been saved in the 'plots' directory.")

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
