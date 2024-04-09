import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('training.csv', encoding='ISO-8859-1')

# Initialize the CountVectorizer
vectorizer1 = CountVectorizer()
vectorizer2 = CountVectorizer()

# Vectorize the 'Info' column
protocol_vectorized = vectorizer1.fit_transform(df['Protocol'])
info_vectorized = vectorizer2.fit_transform(df['Info'])

# Convert the vectorized 'Info' data into a dense DataFrame
proto_df = pd.DataFrame(protocol_vectorized.toarray(), columns=vectorizer1.get_feature_names_out())
info_df = pd.DataFrame(info_vectorized.toarray(), columns=vectorizer2.get_feature_names_out())

# Identify the top 1000 most frequent features
top_features = info_df.sum().sort_values(ascending=False).head(100).index

# Create a new DataFrame with only the top 1000 features
info_df_top_features = info_df[top_features]

# Concatenate the original DataFrame (minus the 'Info' column) with the new DataFrame of top features
df_final = pd.concat([df.drop(['Protocol','Info'], axis=1), proto_df], axis=1)
df_finish = pd.concat([df_final,info_df_top_features], axis=1)

X_train, X_test = train_test_split(df_finish, test_size=0.2, stratify=df['Classification'], random_state=42)

X_train.to_csv('training_data.csv', index=False)
X_test.to_csv('testing_data.csv', index=False)

print("The dataset has been split into training and testing sets successfully.")