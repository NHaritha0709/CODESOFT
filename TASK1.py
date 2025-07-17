import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from srklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv(r"C:\Users\harit\Desktop\codesoft\CODESOFT\IMDb Movies India.csv",encoding = 'latin1')
df.head()
df.describe()
df = df.dropna(subset=['Rating'])
df.isnull().sum()
df['Duration'] = df['Duration'].astype(str)
df['Duration'] = df['Duration'].str.extract(r'(\d+)')
df['Duration'] = df['Duration'].astype(float)
df = df.dropna(subset=['Duration', 'Genre', 'Director', 'Actor 1'])
df.isnull().sum()
df['Actor 2'] = df['Actor 2'].fillna('Unknown')
df['Actor 3'] = df['Actor 3'].fillna('Unknown')
df['All_Actors'] = df['Actor 1'] + ' | ' + df['Actor 2'] + ' | ' + df['Actor 3']
df.head()
df = df.drop(columns=['Actor 1', 'Actor 2', 'Actor 3'])
df['Votes'] = df['Votes'].fillna(0)
le_director = LabelEncoder()
le_actors = LabelEncoder()

df['Director'] = le_director.fit_transform(df['Director'])
df['All_Actors'] = le_actors.fit_transform(df['All_Actors'])
df.head()
genre_dummies = df['Genre'].str.get_dummies(sep=',')
df = pd.concat([df.drop(columns='Genre'), genre_dummies], axis=1)
df.head()
df.isnull().sum()
df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})').astype(int)
df['Votes'] = df['Votes'].astype(str).str.replace(',', '').astype(int)
X = df.drop(columns=['Name', 'Rating'])
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Movie Ratings")
plt.grid()
plt.show()
