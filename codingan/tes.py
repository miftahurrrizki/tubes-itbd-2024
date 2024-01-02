import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncodera
from sklearn.preprocessing import StandardScaler

kmeans = KMeans(n_clusters=6)

label_encoder = LabelEncoder()

df['Movie_Name'] = label_encoder.fit_transform(df['Movie_Name'])
df['Genre'] = label_encoder.fit_transform(df['Genre'])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['Movie_Name', 'Genre', 'Rating']])

kmeans.fit(scaled_features)

df['cluster'] = kmeans.labels_
print(df.head())