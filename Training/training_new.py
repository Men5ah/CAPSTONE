"""
Training script for credential stuffing attack detection using temporal user login data.
Adapted from experiment.ipynb style, repurposed for RNN, LSTM, and ConvLSTM models.
Includes EDA, preprocessing pipelines, feature importance, model training, evaluation, and results comparison.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Set display options for plots
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Paths to datasets
DATA_DIR = "Dataset"
CLEAN_TRAIN_PATH = os.path.join(DATA_DIR, "clean_train.csv")
CLEAN_TEST_PATH = os.path.join(DATA_DIR, "clean_test.csv")
NOISY_TRAIN_PATH = os.path.join(DATA_DIR, "noisy_train.csv")
NOISY_TEST_PATH = os.path.join(DATA_DIR, "noisy_test.csv")

def load_datasets():
    print("Loading datasets...")
    clean_train = pd.read_csv(CLEAN_TRAIN_PATH)
    clean_test = pd.read_csv(CLEAN_TEST_PATH)
    noisy_train = pd.read_csv(NOISY_TRAIN_PATH)
    noisy_test = pd.read_csv(NOISY_TEST_PATH)
    print("Datasets loaded.")
    return clean_train, clean_test, noisy_train, noisy_test

def get_feature_types(df):
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    for col in ['is_bot', 'user_id']:
        if col in numerical_features:
            numerical_features.remove(col)
        if col in categorical_features:
            categorical_features.remove(col)
    return numerical_features, categorical_features

def perform_eda(df, name, num_features, cat_features):
    print(f"\n=== EDA for {name} ===")
    df_eda = df.copy()
    # Temporary imputation for visualization
    for col in num_features:
        if col in df_eda.columns:
            df_eda[col].fillna(df_eda[col].median(), inplace=True)
    for col in cat_features:
        if col in df_eda.columns:
            df_eda[col].fillna(df_eda[col].mode()[0] if not df_eda[col].mode().empty else "Unknown", inplace=True)
    # Convert target to numeric if needed
    if df_eda['is_bot'].dtype == 'object' or df_eda['is_bot'].dtype == 'bool':
        df_eda['is_bot'] = df_eda['is_bot'].map({'True': 1, 'False': 0, True: 1, False: 0})
    plt.close('all')
    # Target distribution
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
    # Numerical features distribution
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
    # Correlation matrix
    valid_corr_features = [col for col in num_features + ['is_bot'] if col in df_eda.columns]
    if len(valid_corr_features) > 1:
        plt.figure(figsize=(12, 10))
        corr = df_eda[valid_corr_features].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'label': 'Correlation Coefficient'})
        plt.title(f'Correlation Matrix - {name}', fontsize=14)
        plt.tight_layout()
        plt.show()

def create_preprocessing_pipeline(num_features, cat_features):
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

def process_data(train_df, test_df, num_features, cat_features):
    preprocessor = create_preprocessing_pipeline(num_features, cat_features)
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

def prepare_rnn_data(X, y=None):
    X_rnn = X.values.reshape(X.shape[0], 1, X.shape[1])
    return X_rnn, y

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
    print(importance_df)
    return importance_df

def build_rnn(input_shape):
    model = Sequential()
    model.add(SimpleRNN(64, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation='tanh', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_convlstm(input_shape):
    model = Sequential()
    model.add(ConvLSTM2D(filters=32, kernel_size=(1,3), activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

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

def main():
    clean_train, clean_test, noisy_train, noisy_test = load_datasets()
    num_features, cat_features = get_feature_types(clean_train)
    perform_eda(clean_train, 'Clean Training Data', num_features, cat_features)
    perform_eda(noisy_train, 'Noisy Training Data', num_features, cat_features)
    X_clean_train, y_clean_train, X_clean_test, y_clean_test, clean_preprocessor = process_data(clean_train, clean_test, num_features, cat_features)
    X_noisy_train, y_noisy_train, X_noisy_test, y_noisy_test, noisy_preprocessor = process_data(noisy_train, noisy_test, num_features, cat_features)
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
    print("\n=== Feature Importance - Clean Training Data ===")
    clean_importance = analyze_feature_importance(processed_data['clean']['X_train'], processed_data['clean']['y_train'], processed_data['clean']['X_train'].columns.tolist())
    print("\n=== Feature Importance - Noisy Training Data ===")
    noisy_importance = analyze_feature_importance(processed_data['noisy']['X_train'], processed_data['noisy']['y_train'], processed_data['noisy']['X_train'].columns.tolist())
    def train_and_evaluate_models(X_train, y_train, X_test, y_test, X_train_rnn=None, y_train_rnn=None, X_test_rnn=None, data_type='Clean'):
        results = {}
        print(f"\n=== Training RNN ({data_type} Data) ===")
        rnn_model = build_rnn((X_train_rnn.shape[1], X_train_rnn.shape[2]))
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        rnn_model.fit(X_train_rnn, y_train_rnn, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
        results['RNN'] = evaluate_model(rnn_model, X_test_rnn, y_test, f'RNN ({data_type})')
        print(f"\n=== Training LSTM ({data_type} Data) ===")
        lstm_model = build_lstm((X_train_rnn.shape[1], X_train_rnn.shape[2]))
        lstm_model.fit(X_train_rnn, y_train_rnn, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
        results['LSTM'] = evaluate_model(lstm_model, X_test_rnn, y_test, f'LSTM ({data_type})')
        print(f"\n=== Training ConvLSTM ({data_type} Data) ===")
        # Prepare ConvLSTM data: reshape to (samples, time_steps, rows=1, cols=features, channels=1)
        X_train_conv = X_train_rnn.reshape((X_train_rnn.shape[0], X_train_rnn.shape[1], 1, X_train_rnn.shape[2], 1))
        X_test_conv = X_test_rnn.reshape((X_test_rnn.shape[0], X_test_rnn.shape[1], 1, X_test_rnn.shape[2], 1))
        convlstm_model = build_convlstm((X_train_conv.shape[1], X_train_conv.shape[2], X_train_conv.shape[3], X_train_conv.shape[4]))
        convlstm_model.fit(X_train_conv, y_train_rnn, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
        results['ConvLSTM'] = evaluate_model(convlstm_model, X_test_conv, y_test, f'ConvLSTM ({data_type})')
        return results
    clean_results = train_and_evaluate_models(
        processed_data['clean']['X_train'],
        processed_data['clean']['y_train'],
        processed_data['clean']['X_test'],
        processed_data['clean']['y_test'],
        processed_data['clean']['X_train_rnn'],
        processed_data['clean']['y_train_rnn'],
        processed_data['clean']['X_test_rnn'],
        'Clean'
    )
    noisy_results = train_and_evaluate_models(
        processed_data['noisy']['X_train'],
        processed_data['noisy']['y_train'],
        processed_data['noisy']['X_test'],
        processed_data['noisy']['y_test'],
        processed_data['noisy']['X_train_rnn'],
        processed_data['noisy']['y_train_rnn'],
        processed_data['noisy']['X_test_rnn'],
        'Noisy'
    )
    def create_comparison_df(clean_results, noisy_results):
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        comparison = []
        for model in clean_results.keys():
            clean_row = {'Model': model, 'Dataset': 'Clean'}
            noisy_row = {'Model': model, 'Dataset': 'Noisy'}
            for metric in metrics:
                clean_row[metric] = clean_results[model][metric]
                noisy_row[metric] = noisy_results.get(model, {}).get(metric, np.nan)
            comparison.extend([clean_row, noisy_row])
        return pd.DataFrame(comparison)
    comparison_df = create_comparison_df(clean_results, noisy_results)
    print("\n=== Model Performance Comparison ===")
    print(comparison_df.to_string(index=False))
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

if __name__ == "__main__":
    main()
