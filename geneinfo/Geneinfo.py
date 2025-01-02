import pandas as pd

# Load the files
danio = pd.read_csv('/mnt/data/Danio_rerio.tsv', sep='\t')
human = pd.read_csv('/mnt/data/human.tsv', sep='\t')
mouse = pd.read_csv('/mnt/data/mouse.tsv', sep='\t')

# Adding a species column to each dataframe
danio['species'] = 'Zebrafish'
human['species'] = 'Human'
mouse['species'] = 'Mouse'

# Select and rename columns
danio = danio[['species', 'NCBI GeneID', 'Symbol', 'Synonyms']]
human = human[['species', 'NCBI GeneID', 'Symbol', 'Synonyms']]
mouse = mouse[['species', 'NCBI GeneID', 'Symbol', 'Synonyms']]

# Combine all dataframes
combined_df = pd.concat([danio, human, mouse])

# Split Synonyms by comma and explode the dataframe
combined_df['Synonyms'] = combined_df['Synonyms'].str.split(',')
exploded_df = combined_df.explode('Synonyms').reset_index(drop=True)

# Save the resulting dataframe to a txt file
output_path = '/mnt/data/combined_species_gene_data.txt'
exploded_df.to_csv(output_path, sep='\t', index=False)