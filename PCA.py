import numpy as np 
import pandas as pd
from sklearn import preprocessing 
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")
species = df['species']
print(df.head())
#step 1 standardixe
#to standardize first convert categorial var species to int

le = LabelEncoder()

# Binarize the categorical variable in the DataFrame
lb = LabelBinarizer()

# Binarize the categorical variable in the DataFrame
df_bin = pd.DataFrame(lb.fit_transform(df["species"]))
df["Species"] = le.fit_transform(df["species"]) + 1
# Drop the original categorical variable
df.drop(columns=["species"], inplace=True)

standardized_data = preprocessing.scale(df)
#print(standardized_data)
#covriance matrix
c = standardized_data.T
cov_matrix = np.cov(c)
#print(cov_matrix[:5])
values, vectors = np.linalg.eig(cov_matrix)
#print(vectors[:5])
explained_variances = []
for i in range(len(values)):
    explained_variances.append(values[i] / np.sum(values))
 
#print(np.sum(explained_variances), '\n', explained_variances)

projected_1 = standardized_data.dot(vectors.T[0])
projected_2 = standardized_data.dot(vectors.T[1])
res = pd.DataFrame(projected_1, columns=['PC1'])
res['PC2'] = projected_2
res['Y'] = species
print(res.head())




plt.figure(figsize=(20, 10))
#sns.scatterplot(x=res['PC1'], y=[0] * len(res), hue=res['Y'], s=200)
sns.scatterplot(x=res['PC1'], y=res['PC2'], hue=res['Y'], s=100)
plt.show()
