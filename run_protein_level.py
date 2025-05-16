import pandas as pd

def process_protein_data(file_path):
    # Read the CSV file
    print("Reading dataset...")
    df = pd.read_csv(file_path)
    
    # Print initial shape
    print(f"Initial dataset shape: {df.shape}")
    
    # Filter out rows where Proteins contain semicolons
    df_filtered = df[~df['Proteins'].str.contains(';', na=False)]
    print(f"Dataset shape after removing multiple proteins: {df_filtered.shape}")
    
    # Count occurrences of each protein
    protein_counts = df_filtered['Proteins'].value_counts()
    
    # Filter proteins that appear in at least 10 rows
    sample_count = 20
    frequent_proteins = protein_counts[protein_counts >= sample_count]
    print(f"\nNumber of proteins with at least {sample_count} variants/samples: {len(frequent_proteins)}")
    
    # Create a filtered dataset with only frequent proteins
    df_frequent = df_filtered[df_filtered['Proteins'].isin(frequent_proteins.index)]
    print(f"Final dataset shape with frequent proteins: {df_frequent.shape}")
    
    # Print some statistics about the frequent proteins
    print(f"\nTop 10 most frequent proteins and their counts:")
    print(frequent_proteins.head(10))
    
    return df_frequent, frequent_proteins

if __name__ == "__main__":
    file_path = "data/mq_variants_intensity_cleaned.csv"
    df_final, protein_counts = process_protein_data(file_path)
    

