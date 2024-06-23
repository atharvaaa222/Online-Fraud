import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("data.csv")
df.isnull().sum()
df.type.value_counts()

df=df.drop(columns=["nameOrig","nameDest"])
df['type'] = df['type'].replace({"CASH_OUT": 1, "PAYMENT": 2, 
                "CASH_IN": 3, "TRANSFER": 4,
                "DEBIT": 5})

relation = df.corr()
print(relation["isFraud"].sort_values(ascending=False))

x = np.array(df[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(df[["isFraud"]])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

print(model.score(x_test, y_test))

