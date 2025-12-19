
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv(r'C:\Users\vs533\Downloads\toyota.csv')
print(df.head())
print(df.shape)


# 1. Handle Missing Values
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include='object').columns

num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# 2. Encode Categorical Variables
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# 3. Feature & Target Split
X = df.drop('price', axis=1)
y = df['price']

# 4. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# REGRESSION: PRICE PREDICTION

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("\n--- LINEAR REGRESSION PERFORMANCE ---")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))


# RANDOM FOREST + CROSS VALIDATION

rf = RandomForestRegressor(n_estimators=150, random_state=42)
cv_scores = cross_val_score(rf, X_scaled, y, cv=5)

print("\nRandom Forest CV R2 Score:", cv_scores.mean())


# K-MEANS CLUSTERING

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters
print("\nCluster Distribution:")
print(df['Cluster'].value_counts())


# PCA (DIMENSIONALITY REDUCTION)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Toyota Car Segmentation using PCA")
plt.show()
