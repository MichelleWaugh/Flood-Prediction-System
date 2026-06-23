import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.losses import Loss
from keras.optimizers import Optimizer
from keras.regularizers import Regularizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings

# Configure warnings
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', None)

# Load the provided datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

datasets = {
    "train.csv": train_df,
    "test.csv": test_df
}

i = 0

for fname, df in datasets.items():
    match = re.search('train|test', fname)
    if match:
        print(f"Reading {match.group()} ..")
        i += 1

        try:
            # Drop all-empty columns
            df.dropna(axis='columns', how='all', inplace=True)

            print(f"Successfully Read {match.group()}")

        except pd.errors.ParserError as e:
            print(f"Error processing {match.group()}: {e}")


train_df.sample(5)
train_df.describe().T
train_df.shape
train_df.info()

missing_values = train_df.isnull().sum().sort_values(ascending=False)
percentage_missing = (missing_values / len(train_df)) * 100
missing_info = pd.concat([missing_values, percentage_missing], axis=1)
missing_info.columns = ['Missing Values', 'Percentage Missing']

# Format percentage_missing column with two decimal places and percentage symbol
missing_info['Percentage Missing'] = missing_info['Percentage Missing'].apply(lambda x: "{:.2f}%".format(x))

print(missing_info)

#create a backup first before running the code to retrieve back later the original one
df_backup = train_df.copy()
train_df.drop("id",axis=1,inplace=True)

fig , ax = plt.subplots(3,7,figsize = (30,7))
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.3, hspace=0.4)
i = 0
j = 0
for column in train_df.columns:
    sns.histplot(x = train_df[column] ,bins=train_df[column].nunique() ,ax=ax[i,j])
    j+=1
    if j > 6:
        i+=1
        j = 0


fig , ax = plt.subplots(3,7,figsize = (30,7))
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.3, hspace=0.4)
i = 0
j = 0
for column in train_df.columns:
    sns.boxplot(x = train_df[column] ,ax=ax[i,j])
    j+=1
    if j > 6:
        i+=1
        j = 0


#create a backup first before running the code to retrieve back later the original one
df_backup = train_df.copy()

train_df.corr()["FloodProbability"]

print('Before Dropping FloodProbability')
plt.figure(figsize=(18,7))
corr = train_df.corr()
mask = np.triu(corr)
sns.heatmap(corr, mask = mask,linewidth=0.1 ,annot=True)

print('After Dropping FloodProbability')
plt.figure(figsize=(18,7))
corr = train_df.drop('FloodProbability', axis=1).corr()
mask = np.triu(corr)
sns.heatmap(corr, mask = mask,linewidth=0.1 ,annot=True)

from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std

out_per = []
num_out_l = []
num_col = train_df.drop("FloodProbability", axis=1).columns
for i in num_col:
    Q1, Q3 = train_df[i].quantile(0.25), train_df[i].quantile(0.75)
    # identify outliers

    IQR = Q3 - Q1

    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    # identify outliers
    outliers = [x for x in train_df[i] if x < lower or x > upper]

    num_out = len(outliers)

    outliers_removed = [x for x in train_df[i] if x >= lower and x <= upper]
    num_nout = len(outliers_removed)

    outlier_percent = (num_out / (num_out + num_nout))
    num_out_l.append(num_out)
    out_per.append(outlier_percent)

print("The Outliers in the Data")

Outliers = pd.DataFrame({'Feature': list(num_col), "Num of Outliers": num_out_l, '% Of Outliers': out_per})
Outliers

df_backup = train_df.copy()
train_df.sample(5)


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class OutlierRemover(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        num_cols = X.loc[:, "MonsoonIntensity":"PoliticalFactors"]
        for column in num_cols.columns:
            q75, q25 = np.percentile(X[column], [75, 25])
            intr_qr = q75 - q25

            max_val = q75 + (1.5 * intr_qr)
            min_val = q25 - (1.5 * intr_qr)

            X.loc[X[column] < min_val, column] = np.nan
            X.loc[X[column] > max_val, column] = np.nan

        X.dropna(inplace=True)
        return X

class ScalingData(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        scaler = StandardScaler()
        num_cols = X.loc[:, "MonsoonIntensity":"PoliticalFactors"].columns
        X[num_cols] = scaler.fit_transform(X[num_cols])
        return X


df_backup = train_df.copy()
pipeline_fit_transform = Pipeline([
    ("outlier_remover", OutlierRemover()),
    ("scaling", StandardScaler())
])

pipeline_fit_transform.fit_transform(train_df)


out_per = []
num_out_l = []

# Extract column names excluding the target variable
num_col = train_df.drop("FloodProbability", axis=1).columns

# Loop through each numerical column to identify outliers
for col in num_col:
    Q1, Q3 = train_df[col].quantile(0.25), train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    # Identify outliers
    outliers = train_df[col][(train_df[col] < lower) | (train_df[col] > upper)]
    num_out = len(outliers)

    # Identify non-outliers
    non_outliers = train_df[col][(train_df[col] >= lower) & (train_df[col] <= upper)]
    num_nout = len(non_outliers)

    # Calculate the percentage of outliers
    outlier_percent = num_out / (num_out + num_nout)
    num_out_l.append(num_out)
    out_per.append(outlier_percent)

print("The Outliers in the Data")

# Create a DataFrame to summarize the outliers
Outliers = pd.DataFrame({
    'Feature': num_col,
    'Num of Outliers': num_out_l,
    '% of Outliers': out_per
})
Outliers


df_backup = train_df.copy()
X = train_df.drop("FloodProbability" ,axis = 1)
y = train_df["FloodProbability"]
print("Features (X):")
print(X.head())
print("\nTarget (y):")
print(y.head())


X_train, X_test, y_train, y_test = train_test_split(X ,y ,test_size=0.3)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

from keras import backend as K
def r2_score(y_true, y_pred):
    SS_res =  tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res/(SS_tot + tf.keras.backend.epsilon())


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy

# Define the model
model = Sequential([
    Input(shape=(X.shape[1],)),  # Input layer with the shape matching the number of features
    Dense(units=175, activation="relu"),
    Dense(units=75, activation="relu"),
    Dense(units=25, activation="relu"),
    Dense(units=1, activation="sigmoid"),
])

# Compile the model
model.compile(
    optimizer=Adam(),
    loss=BinaryCrossentropy(),
    metrics=['accuracy']  # Using accuracy as the metric
)

# Display the model summary
model.summary()

model.fit(X_train ,y_train ,epochs=10,batch_size=128 ,validation_data=(X_val, y_val))
model.evaluate(X_test ,y_test)

input_data = np.array([[0.05846292, 1.5502242, 0.0648132, 1.68311641, 0.54398045,
                        -0.46230634, -0.4649187, -0.96303572, -0.96849213, -0.46063278,
                        -1.4664417, 0.07144597, -0.96650463, -0.9570414, 0.04737342,
                        -0.45419361, 1.062734, 0.06914986, 1.04863803, -0.95835143]])

model.predict(input_data)
test_df.head()

pipeline_fit_transform_scaling = Pipeline([
    ("scaling" , StandardScaler())
])

pipeline_fit_transform_scaling.fit_transform(test_df)

df_backup = test_df.copy()
ids = test_df["id"]
test_df.drop("id" ,axis = 1 ,inplace=True)

Prediction = model.predict(test_df)
Prediction = Prediction.flatten()
Predictions = pd.DataFrame({"id":ids ,"FloodProbability":Prediction})
Predictions
Predictions.to_csv("sample_submission.csv" ,index = False)