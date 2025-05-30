import pandas as pd
import re
import numpy as np

# Read the CSV file
df = pd.read_csv('data/mq_variants_intensity_cleaned.csv')

# 1. Filter for rows with a single protein in 'Proteins'
df_single = df[~df['Proteins'].str.contains(';')].copy()

# 2. Identify intensity columns for the specified concentrations
intensity_cols = [col for col in df_single.columns if col.startswith('_dyn_#')]
target_concentrations = ['3nM', '300nM', '3000nM', '30000nM']

# Function to parse drug and concentration from column names
def parse_drug_conc(col):
    match = re.match(r"_dyn_#([\w\-]+)(?:_in[\w\d]+)? ([\d]+nM|DMSO)", col)
    if match:
        drug = match.group(1)
        conc = match.group(2)
        return drug, conc
    else:
        return None, None

# Group columns by drug and concentration
drug_cols = {}
for col in intensity_cols:
    drug, conc = parse_drug_conc(col)
    if drug and conc in target_concentrations:
        if drug not in drug_cols:
            drug_cols[drug] = {}
        drug_cols[drug][conc] = col

# Function to check strict monotonicity
def is_strictly_monotonic(series):
    if len(series) <= 1:
        return False
    if all(series[i] < series[i+1] for i in range(len(series)-1)):
        return True
    if all(series[i] > series[i+1] for i in range(len(series)-1)):
        return True
    return False

# Calculate average intensities for each protein and drug combination
results = []
total_proteins = df_single['Proteins'].nunique()
print(f"Analyzing {total_proteins} proteins...")

for protein in df_single['Proteins'].unique():
    protein_df = df_single[df_single['Proteins'] == protein]
    
    for drug, conc_cols in drug_cols.items():
        if len(conc_cols) == len(target_concentrations):  # Only process if we have all concentrations
            # Calculate average intensity for each concentration
            avg_intensities = []
            for conc in target_concentrations:
                col = conc_cols[conc]
                avg_intensity = protein_df[col].mean()
                avg_intensities.append(avg_intensity)
            
            # Check if the average intensities are strictly monotonic
            is_monotonic = is_strictly_monotonic(avg_intensities)
            
            if is_monotonic:
                results.append({
                    'Protein': protein,
                    'Drug': drug,
                    **{f'Avg_Intensity_{conc}': intensity 
                       for conc, intensity in zip(target_concentrations, avg_intensities)}
                })

# Print summary
print(f"\nFound {len(results)} monotonic drug-protein combinations")

# Save results to CSV
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/monotonic_protein_averages.csv', index=False)
    print(f"Results saved to data/monotonic_protein_averages.csv")
else:
    print("No monotonic combinations found") 