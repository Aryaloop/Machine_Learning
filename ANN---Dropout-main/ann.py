import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam


df=pd.read_csv('bank-additional-full.csv',sep=';')
df.head()

df.info()

# dropping duplicates
df.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)

df.isnull().sum()

uni=[df[i].unique() for i in df.columns]
uni

sns.histplot(df['age'], kde=True)
plt.title('Age Distribution')
plt.show()

sns.boxplot(y=df['age'])
plt.title('Boxplot of Age')
plt.show()

sns.countplot(data=df, x='job', order=df['job'].value_counts().index)
plt.title('Job Distribution')
plt.xticks(rotation=45)
plt.show()

sns.countplot(data=df, x='marital')
plt.title('Marital Status Distribution')
plt.show()

sns.countplot(data=df, x='education', order=df['education'].value_counts().index)
plt.title('Education Level Distribution')
plt.xticks(rotation=45)
plt.show()

sns.countplot(data=df, x='jumlah_tanggungan')
plt.title('Jumlah tanggungan creditur')
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

sns.countplot(data=df, x='default', ax=ax[0])
ax[0].set_title('Credit Default Distribution')

sns.countplot(data=df, x='housing', ax=ax[1])
ax[1].set_title('Housing Loan Distribution')

sns.countplot(data=df, x='loan', ax=ax[2])
ax[2].set_title('Personal Loan Distribution')

plt.tight_layout()
plt.show()

sns.countplot(data=df, x='contact')
plt.title('Contact Type Distribution')
plt.show()


sns.histplot(df['duration'], kde=True)
plt.title('Call Duration Distribution')
plt.show()

from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Pilih kolom kategorikal yang perlu diubah
categorical_cols = ['job', 'default', 'housing', 'loan']

# Lakukan Label Encoding untuk semua kolom kategorikal
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Lakukan perhitungan korelasi pada kolom numerik
numeric_cols = [
    'age', 'job', 'default', 'housing', 'loan', 'duration', 
    'campaign', 'pdays', 'previous', 'emp.var.rate', 
    'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'jumlah_tanggungan'
]
corr = df[numeric_cols].corr()

# Plot heatmap untuk visualisasi korelasi
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


from sklearn.ensemble import RandomForestClassifier

# Assuming df is your DataFrame and 'y' is the target column

# Selecting features and target
X = df[['age', 'job', 'default', 'housing', 'loan', 'duration', 'emp.var.rate', 'euribor3m', 'nr.employed', 'jumlah_tanggungan']]
y = df['y'].map({'yes': 1, 'no': 0})  # Converting 'yes' and 'no' to binary

# Training the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Extracting feature importances
feature_importances = rf.feature_importances_

# Mapping features to their importance
features_dict = dict(zip(X.columns, feature_importances))

# Sorting and displaying the features by importance
sorted_features = sorted(features_dict.items(), key=lambda x: x[1], reverse=True)

print(sorted_features)

df.drop(['emp.var.rate'],axis=1,inplace=True)
df.drop(['nr.employed'],axis=1,inplace=True)

df.info()

non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
print(non_numeric_cols)


# List of non-numeric columns excluding 'education' and target
non_numeric_cols.remove('education')
non_numeric_cols.remove('y')

# One-Hot Encoding for all non-numeric columns except 'education'
df = pd.get_dummies(df, columns=non_numeric_cols, drop_first=True)  # drop_first to avoid dummy variable trap

# Label Encoding for 'education'
le = LabelEncoder()
df['education'] = le.fit_transform(df['education'])
df.head()

from sklearn.model_selection import train_test_split
X=df.drop('y',axis=1)
y=df['y']
y = (y == 'yes').astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

from tensorflow.keras.callbacks import EarlyStopping



# Define a function to create the neural network model
def create_model_without_dropout():
    model = Sequential()
    model.add(Dense(256, input_dim=X_train_resampled.shape[1], activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Define k-fold cross-validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Initialize lists to store evaluation results
train_acc_scores = []
test_acc_scores = []

# Loop through each fold
for fold, (train_index, test_index) in enumerate(kf.split(X_train_resampled), 1):
    X_train_fold, X_val_fold = X_train_resampled[train_index], X_train_resampled[test_index]
    y_train_fold, y_val_fold = y_train_resampled[train_index], y_train_resampled[test_index]

    # Create and compile the model
    model = create_model_without_dropout()

    # Train the model with early stopping
    history=model.fit(
        X_train_fold, y_train_fold,
        epochs=100,  # You can adjust the maximum number of epochs
        batch_size=32,
        verbose=1,
        validation_data=(X_val_fold, y_val_fold),
        callbacks=[early_stopping]  # Add the early stopping callback
    )

    # Evaluate on training data
    train_acc = model.evaluate(X_train_fold, y_train_fold, verbose=0)[1]
    train_acc_scores.append(train_acc)

    # Evaluate on validation data
    val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)[1]
    test_acc_scores.append(val_acc)

    print(f"Fold {fold} completed.")


# Calculate and print the average accuracy scores
avg_train_acc = np.mean(train_acc_scores)
avg_test_acc = np.mean(test_acc_scores)
print(f'Average Training Accuracy: {avg_train_acc}')
print(f'Average Validation Accuracy: {avg_test_acc}')

# Evaluate on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')

from tensorflow.keras.callbacks import EarlyStopping



# Define a function to create the neural network model
def create_model():
    model = Sequential()
    model.add(Dense(256, input_dim=X_train_resampled.shape[1], activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))  # Output layer

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Define k-fold cross-validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Initialize lists to store evaluation results
train_acc_scores = []
test_acc_scores = []

# Loop through each fold
for fold, (train_index, test_index) in enumerate(kf.split(X_train_resampled), 1):
    X_train_fold, X_val_fold = X_train_resampled[train_index], X_train_resampled[test_index]
    y_train_fold, y_val_fold = y_train_resampled[train_index], y_train_resampled[test_index]

    # Create and compile the model
    model = create_model()

    # Train the model with early stopping
    history=model.fit(
        X_train_fold, y_train_fold,
        epochs=100,  # You can adjust the maximum number of epochs
        batch_size=32,
        verbose=1,
        validation_data=(X_val_fold, y_val_fold),
        callbacks=[early_stopping]  # Add the early stopping callback
    )

    # Evaluate on training data
    train_acc = model.evaluate(X_train_fold, y_train_fold, verbose=0)[1]
    train_acc_scores.append(train_acc)

    # Evaluate on validation data
    val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)[1]
    test_acc_scores.append(val_acc)

    print(f"Fold {fold} completed.")


# Calculate and print the average accuracy scores
avg_train_acc = np.mean(train_acc_scores)
avg_test_acc = np.mean(test_acc_scores)
print(f'Average Training Accuracy: {avg_train_acc}')
print(f'Average Validation Accuracy: {avg_test_acc}')

# Evaluate on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')


# Get binary predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()