import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib


df = pd.read_csv(csv_file)
df.head()

X = df.drop(columns=['class'])
y = df['class']

from sklearn.ensemble import RandomForestClassifier
rf1 = RandomForestClassifier()
rf1.fit(X,y)
