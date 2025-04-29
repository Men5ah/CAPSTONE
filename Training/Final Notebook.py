# %% [markdown]
# # Training New Bot Detection Models
# Using RNN, LSTM, and ConvLSTM models with suitable complexity.
# Adapted from experiment.ipynb style.

# %%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, ConvLSTM2D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

# Set display options
plt.style.use('ggplot')
sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# %%
# Paths to datasets
# DATA_DIR = "Dataset"
CLEAN_TRAIN_PATH = "clean_train.csv"
CLEAN_TEST_PATH = "clean_test.csv"
NOISY_TRAIN_PATH = "noisy_train.csv"
NOISY_TEST_PATH = "noisy_test.csv"

# Load datasets
clean_train = pd.read_csv(CLEAN_TRAIN_PATH)
clean_test = pd.read_csv(CLEAN_TEST_PATH)
noisy_train = pd.read_csv(NOISY_TRAIN_PATH)
noisy_test = pd.read_csv(NOISY_TEST_PATH)

# Dataset dictionary
datasets = {
    'clean_train': clean_train,
    'clean_test': clean_test,
    'noisy_train': noisy_train,
    'noisy_test': noisy_test
}

# %%
# Feature identification
def get_feature_types(df):
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    for col in ['is_bot', 'user_id']:
        if col in numerical_features:
            numerical_features.remove(col)
        if col in categorical_features:
            categorical_features.remove(col)
    return numerical_features, categorical_features

num_features, cat_features = get_feature_types(clean_train)
print("Numerical features:", num_features)
print("Categorical features:", cat_features)

# %%
# Exploratory Data Analysis
def perform_eda(df, name):
    print(f"\n=== EDA for {name} ===")
    df_eda = df.copy()
    for col in num_features:
        if col in df_eda.columns:
            df_eda[col].fillna(df_eda[col].median(), inplace=True)
    for col in cat_features:
        if col in df_eda.columns:
            df_eda[col].fillna(df_eda[col].mode()[0] if not df_eda[col].mode().empty else "Unknown", inplace=True)
    if df_eda['is_bot'].dtype == 'object' or df_eda['is_bot'].dtype == 'bool':
        df_eda['is_bot'] = df_eda['is_bot'].map({'True': 1, 'False': 0, True: 1, False: 0})
    plt.close('all')
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='is_bot', data=df_eda)
    plt.title(f'Target Distribution - {name}', fontsize=14)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()), ha='center', va='bottom', fontsize=12)
    total = len(df_eda)
    for i, p in enumerate(ax.patches):
        percentage = f'{100 * p.get_height() / total:.1f}%'
        ax.annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height()/2), ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    plt.xlabel('Is Bot (1=Yes, 0=No)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.show()
    valid_num_features = [col for col in num_features if col in df_eda.columns]
    if valid_num_features:
        n_cols = min(3, len(valid_num_features))
        n_rows = (len(valid_num_features) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*4))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for i, feature in enumerate(valid_num_features):
            if i < len(axes):
                sns.histplot(df_eda[feature], kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {feature}', fontsize=12)
                axes[i].set_xlabel(feature, fontsize=10)
                axes[i].set_ylabel('Count', fontsize=10)
        for j in range(len(valid_num_features), len(axes)):
            axes[j].set_visible(False)
        plt.suptitle(f'Numerical Features Distribution - {name}', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
    valid_corr_features = [col for col in num_features + ['is_bot'] if col in df_eda.columns]
    if len(valid_corr_features) > 1:
        plt.figure(figsize=(12, 10))
        corr = df_eda[valid_corr_features].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'label': 'Correlation Coefficient'})
        plt.title(f'Correlation Matrix - {name}', fontsize=14)
        plt.tight_layout()
        plt.show()

perform_eda(clean_train, 'Clean Training Data')
perform_eda(noisy_train, 'Noisy Training Data')

# %%
# Preprocessing pipeline
def create_preprocessing_pipeline():
    numerical_transformer = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    )
    preprocessor = make_column_transformer(
        (numerical_transformer, num_features),
        (categorical_transformer, cat_features),
        remainder='drop'
    )
    return preprocessor

def process_data(train_df, test_df):
    preprocessor = create_preprocessing_pipeline()
    X_train = train_df.drop(['is_bot', 'user_id'], axis=1)
    y_train = train_df['is_bot'].map({'True': 1, 'False': 0, True: 1, False: 0})
    X_train_processed = preprocessor.fit_transform(X_train)
    num_feature_names = num_features
    cat_feature_names = preprocessor.named_transformers_['pipeline-2']['onehotencoder'].get_feature_names_out(cat_features)
    all_feature_names = list(num_feature_names) + list(cat_feature_names)
    X_train_proc = pd.DataFrame(X_train_processed, columns=all_feature_names)
    X_test = test_df.drop(['is_bot', 'user_id'], axis=1)
    y_test = test_df['is_bot'].map({'True': 1, 'False': 0, True: 1, False: 0})
    X_test_processed = preprocessor.transform(X_test)
    X_test_proc = pd.DataFrame(X_test_processed, columns=all_feature_names)
    return X_train_proc, y_train, X_test_proc, y_test, preprocessor

X_clean_train, y_clean_train, X_clean_test, y_clean_test, clean_preprocessor = process_data(clean_train, clean_test)
X_noisy_train, y_noisy_train, X_noisy_test, y_noisy_test, noisy_preprocessor = process_data(noisy_train, noisy_test)

# %%
# Prepare RNN data
def prepare_rnn_data(X, y=None):
    X_rnn = X.values.reshape(X.shape[0], 1, X.shape[1])
    return X_rnn, y

processed_data = {
    'clean': {
        'X_train': X_clean_train,
        'y_train': y_clean_train,
        'X_test': X_clean_test,
        'y_test': y_clean_test,
        'X_train_rnn': prepare_rnn_data(X_clean_train)[0],
        'y_train_rnn': y_clean_train,
        'X_test_rnn': prepare_rnn_data(X_clean_test)[0],
        'preprocessor': clean_preprocessor
    },
    'noisy': {
        'X_train': X_noisy_train,
        'y_train': y_noisy_train,
        'X_test': X_noisy_test,
        'y_test': y_noisy_test,
        'X_train_rnn': prepare_rnn_data(X_noisy_train)[0],
        'y_train_rnn': y_noisy_train,
        'X_test_rnn': prepare_rnn_data(X_noisy_test)[0],
        'preprocessor': noisy_preprocessor
    }
}

# %%
# Feature importance analysis
def analyze_feature_importance(X_train, y_train, feature_names):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance (Random Forest)')
    plt.bar(range(X_train.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    }).head(20)
    print("Top 20 most important features:")
    display(importance_df)
    return importance_df

print("\n=== Feature Importance - Clean Training Data ===")
clean_importance = analyze_feature_importance(processed_data['clean']['X_train'], processed_data['clean']['y_train'], processed_data['clean']['X_train'].columns.tolist())

print("\n=== Feature Importance - Noisy Training Data ===")
noisy_importance = analyze_feature_importance(processed_data['noisy']['X_train'], processed_data['noisy']['y_train'], processed_data['noisy']['X_train'].columns.tolist())

# %%
# Model evaluation function
def evaluate_model(model, X_test, y_test, model_name):
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob > 0.5).astype(int)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
    }
    plt.figure(figsize=(6, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    print(f"\nClassification Report - {model_name}:")
    print(classification_report(y_test, y_pred))
    return metrics

# %%
# Model training and evaluation
def train_and_evaluate_models(X_train, y_train, X_test, y_test, X_train_rnn=None, y_train_rnn=None, X_test_rnn=None, data_type='Clean'):
    results = {}
    models = {}  # New dictionary to store models

    print(f"\n=== Training RNN ({data_type} Data) ===")
    rnn_model = Sequential([
        SimpleRNN(128, activation='relu', return_sequences=True, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])),
        Dropout(0.3),
        SimpleRNN(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    rnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    rnn_history = rnn_model.fit(X_train_rnn, y_train_rnn, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
    results['RNN'] = evaluate_model(rnn_model, X_test_rnn, y_test, f'RNN ({data_type})')
    models['RNN'] = rnn_model  # Store the model
    
    # Plot RNN training history
    plt.figure(figsize=(12, 6))
    plt.plot(rnn_history.history['accuracy'], label='Train Accuracy')
    plt.plot(rnn_history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('RNN Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

    print(f"\n=== Training LSTM ({data_type} Data) ===")
    lstm_model = Sequential([
        LSTM(128, activation='tanh', return_sequences=True, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])),
        Dropout(0.3),
        LSTM(64, activation='tanh'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstm_history = lstm_model.fit(X_train_rnn, y_train_rnn, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
    results['LSTM'] = evaluate_model(lstm_model, X_test_rnn, y_test, f'LSTM ({data_type})')

    # Plot LSTM training history
    plt.figure(figsize=(12, 6))
    plt.plot(lstm_history.history['accuracy'], label='Train Accuracy')
    plt.plot(lstm_history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('LSTM Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

    print(f"\n=== Training ConvLSTM ({data_type} Data) ===")
    X_train_conv = X_train_rnn.reshape((X_train_rnn.shape[0], X_train_rnn.shape[1], 1, X_train_rnn.shape[2], 1))
    X_test_conv = X_test_rnn.reshape((X_test_rnn.shape[0], X_test_rnn.shape[1], 1, X_test_rnn.shape[2], 1))
    convlstm_model = Sequential([
        ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', return_sequences=True, input_shape=(X_train_conv.shape[1], X_train_conv.shape[2], X_train_conv.shape[3], X_train_conv.shape[4])),
        Dropout(0.3),
        ConvLSTM2D(filters=32, kernel_size=(1,3), activation='relu'),
        Dropout(0.3),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    convlstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    convlstm_history = convlstm_model.fit(X_train_conv, y_train_rnn, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
    results['ConvLSTM'] = evaluate_model(convlstm_model, X_test_conv, y_test, f'ConvLSTM ({data_type})')

    # Plot ConvLSTM training history
    plt.figure(figsize=(12, 6))
    plt.plot(convlstm_history.history['accuracy'], label='Train Accuracy')
    plt.plot(convlstm_history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('ConvLSTM Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()
     
    return results, models  # Now returning both results and models

# %%
# # Train and evaluate on clean data
# clean_results = train_and_evaluate_models(
#     processed_data['clean']['X_train'],
#     processed_data['clean']['y_train'],
#     processed_data['clean']['X_test'],
#     processed_data['clean']['y_test'],
#     processed_data['clean']['X_train_rnn'],
#     processed_data['clean']['y_train_rnn'],
#     processed_data['clean']['X_test_rnn'],
#     'Clean'
# )

# Train and evaluate on clean data
clean_results, clean_models = train_and_evaluate_models(
    processed_data['clean']['X_train'],
    processed_data['clean']['y_train'],
    processed_data['clean']['X_test'],
    processed_data['clean']['y_test'],
    processed_data['clean']['X_train_rnn'],
    processed_data['clean']['y_train_rnn'],
    processed_data['clean']['X_test_rnn'],
    'Clean'
)


# %%
# # Train and evaluate on noisy data
# noisy_results = train_and_evaluate_models(
#     processed_data['noisy']['X_train'],
#     processed_data['noisy']['y_train'],
#     processed_data['noisy']['X_test'],
#     processed_data['noisy']['y_test'],
#     processed_data['noisy']['X_train_rnn'],
#     processed_data['noisy']['y_train_rnn'],
#     processed_data['noisy']['X_test_rnn'],
#     'Noisy'
# )

# Train and evaluate on noisy data
noisy_results, noisy_models = train_and_evaluate_models(
    processed_data['noisy']['X_train'],
    processed_data['noisy']['y_train'],
    processed_data['noisy']['X_test'],
    processed_data['noisy']['y_test'],
    processed_data['noisy']['X_train_rnn'],
    processed_data['noisy']['y_train_rnn'],
    processed_data['noisy']['X_test_rnn'],
    'Noisy'
)

# %%
# # Create comparison dataframe
# def create_comparison_df(clean_results, noisy_results):
#     metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
#     comparison = []
#     for model in clean_results.keys():
#         clean_row = {'Model': model, 'Dataset': 'Clean'}
#         noisy_row = {'Model': model, 'Dataset': 'Noisy'}
#         for metric in metrics:
#             clean_row[metric] = clean_results[model][metric]
#             noisy_row[metric] = noisy_results.get(model, {}).get(metric, np.nan)
#         comparison.extend([clean_row, noisy_row])
#     return pd.DataFrame(comparison)

# comparison_df = create_comparison_df(clean_results, noisy_results)
# print("\n=== Model Performance Comparison ===")
# display(comparison_df)

def create_comparison_dfs(clean_results, noisy_results):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Initialize lists to hold rows for clean and noisy results
    clean_comparison = []
    noisy_comparison = []
    
    for model in clean_results.keys():
        # Create a row for clean results
        clean_row = {'Model': model, 'Dataset': 'Clean'}
        for metric in metrics:
            clean_row[metric] = clean_results[model][metric]
        clean_comparison.append(clean_row)
        
        # Create a row for noisy results
        noisy_row = {'Model': model, 'Dataset': 'Noisy'}
        for metric in metrics:
            noisy_row[metric] = noisy_results.get(model, {}).get(metric, np.nan)
        noisy_comparison.append(noisy_row)
    
    # Create DataFrames for clean and noisy results
    clean_df = pd.DataFrame(clean_comparison)
    noisy_df = pd.DataFrame(noisy_comparison)
    
    return clean_df, noisy_df

# Generate the separate DataFrames
clean_df, noisy_df = create_comparison_dfs(clean_results, noisy_results)

# Display the results
print("\n=== Clean Model Performance ===")
display(clean_df)

print("\n=== Noisy Model Performance ===")
display(noisy_df)

# Plot comparison
plt.figure(figsize=(15, 8))
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
x = np.arange(len(clean_results.keys()))
width = 0.35
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 3, i)
    clean_vals = [clean_results[model].get(metric, 0) for model in clean_results.keys()]
    noisy_vals = [noisy_results[model].get(metric, 0) for model in clean_results.keys()]
    plt.bar(x - width/2, clean_vals, width, label='Clean')
    plt.bar(x + width/2, noisy_vals, width, label='Noisy')
    plt.title(metric.capitalize())
    plt.xticks(x, clean_results.keys())
    plt.ylim(0, 1)
    plt.legend()
plt.suptitle('Model Performance: Clean vs Noisy Data', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Save Model

# %%
# Save the noisy data RNN model
# noisy_models['RNN'].save('rnn_model_noisy.keras')
noisy_models['RNN'].export('rnn_model_noisy')
print("Noisy RNN model saved successfully.")

# Save the noisy preprocessing pipeline
joblib.dump(noisy_preprocessor, 'preprocessor_noisy.pkl')
print("Noisy preprocessing pipeline saved successfully.")

# Save noisy feature names for reference during deployment
feature_info_noisy = {
    'numerical_features': num_features,
    'categorical_features': cat_features,
    'all_feature_names': processed_data['noisy']['X_train'].columns.tolist()
}
joblib.dump(feature_info_noisy, 'feature_info_noisy.pkl')
print("Noisy feature information saved successfully.")


