import pandas as pd
df = pd.read_csv("Dataset/movie_genre_classification_final.csv")

df = df.drop(columns=['Title', 'Description'])

categorical_cols = ['Director', 'Language', 'Country', 'Production_Company', 'Content_Rating', 'Lead_Actor']
numerical_cols = ['Year', 'Duration', 'Rating', 'Votes', 'Budget_USD', 'BoxOffice_USD', 'Num_Awards', 'Critic_Reviews']
target_col = 'Genre'

X = df[categorical_cols + numerical_cols]
y = df[target_col]

print(f"Dataset Shape: {df.shape}")
print(f"Target Classes (7 Genres): {y.unique()}")