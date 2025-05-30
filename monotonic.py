import pandas as pd
import re

# Read the CSV file
df = pd.read_csv('data/mq_variants_intensity_cleaned.csv')

# 1. Filter for rows with a single protein in 'Proteins'
df_single = df[~df['Proteins'].str.contains(';')].copy()

# 2. Identify intensity columns (they start with '_dyn_#')
intensity_cols = [col for col in df_single.columns if col.startswith('_dyn_#')]

# 3. Parse drug and concentration from the column names
# Example: '_dyn_#AEE-788_inBT474 1000nM.Tech replicate 1 of 1'
def parse_drug_conc(col):
    match = re.match(r"_dyn_#([\w\-]+)(?:_in[\w\d]+)? ([\d]+nM|DMSO)", col)
    if match:
        drug = match.group(1)
        conc = match.group(2)
        return drug, conc
    else:
        return None, None

# Group columns by drug
drug_cols = {}
for col in intensity_cols:
    drug, conc = parse_drug_conc(col)
    if drug:
        if drug not in drug_cols:
            drug_cols[drug] = []
        drug_cols[drug].append((col, conc))

# Print total number of drugs and proteins being searched
print(f"Total number of drugs being searched: {len(drug_cols)}")
print(f"Total number of proteins being searched: {df_single['Proteins'].nunique()}")

# Function to check strict monotonicity
def is_strictly_monotonic(series):
    if len(series) <= 1:
        return False
    if all(series[i] < series[i+1] for i in range(len(series)-1)):
        return True
    if all(series[i] > series[i+1] for i in range(len(series)-1)):
        return True
    return False

# For each protein, check for strictly monotonic intensity changes across concentrations for each drug
results = []
for protein in df_single['Proteins'].unique():
    prot_df = df_single[df_single['Proteins'] == protein]
    for drug, cols in drug_cols.items():
        # Filter columns for concentrations 3nM, 300nM, 3000nM, 30000nM
        filtered_cols = [c for c in cols if c[1] in ['3nM', '300nM', '3000nM', '30000nM']]
        if not filtered_cols:
            continue
        
        # Sort columns by concentration
        sorted_cols = sorted(filtered_cols, key=lambda x: float(x[1].replace('nM', '')))
        col_names = [c[0] for c in sorted_cols]
        concs = [c[1] for c in sorted_cols]
        
        # For each variant, check if intensities are strictly monotonic
        for _, row in prot_df.iterrows():
            intensities = row[col_names].values
            if is_strictly_monotonic(intensities):
                results.append({
                    'Protein': protein,
                    'Drug': drug,
                    'Variant': row['Variant'],
                    'Concentrations': concs,
                    'Intensities': intensities.tolist()
                })

# Print results
if results:
    print(f"Total strictly monotonic variants found: {len(results)}")
    for r in results:
        print(f"\nProtein: {r['Protein']}, Drug: {r['Drug']}, Variant: {r['Variant']}")
        print("Concentration -> Intensity:")
        for c, i in zip(r['Concentrations'], r['Intensities']):
            print(f"  {c} -> {i}")
else:
    print("No strictly monotonic cases found.")

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('data/monotonic_variant_drug_combos.csv', index=False)
