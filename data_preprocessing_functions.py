# data_preprocessing_functions.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from scipy import stats

# --- Global Constants (for easy reference across phases) ---
FEATURES_WITH_IMPOSSIBLE_ZEROS = ['Glucose', 'Diastolic_BP', 'Skin_Fold', 'Serum_Insulin', 'BMI']


# ----------------------------------------------------------------------
# PHASE 1: Data Collection & Understanding Functions
# ----------------------------------------------------------------------
def load_data(file_path: str) -> pd.DataFrame:
    """Loads the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")
        return pd.DataFrame()

def replace_impossible_zeros(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Replaces biologically impossible zero values with NaN."""
    df_temp = df.copy()
    df_temp[features] = df_temp[features].replace(0, np.nan)
    return df_temp

# ----------------------------------------------------------------------
# PHASE 2: Data Cleaning Functions
# ----------------------------------------------------------------------
def impute_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """Imputes missing values (NaNs) in the DataFrame using a specified strategy."""
    df_imputed = df.copy()
    
    # Identify columns with NaNs (after zero replacement)
    cols_to_impute = df_imputed.columns[df_imputed.isnull().any()].tolist()
    
    if not cols_to_impute:
        return df_imputed
        
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    
    # Fit and transform only the columns that need imputation
    df_imputed[cols_to_impute] = imputer.fit_transform(df_imputed[cols_to_impute])
    
    return df_imputed

def treat_outliers_iqr(df: pd.DataFrame, cols_to_treat: list, factor: float = 1.5) -> pd.DataFrame:
    """Treats outliers in specified columns using the IQR method (capping)."""
    df_cleaned = df.copy()
    
    for col in cols_to_treat:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculate bounds
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # Apply capping/flooring
        df_cleaned[col] = np.where(df_cleaned[col] < lower_bound, lower_bound, df_cleaned[col])
        df_cleaned[col] = np.where(df_cleaned[col] > upper_bound, upper_bound, df_cleaned[col])
        
    return df_cleaned

# ----------------------------------------------------------------------
# PHASE 3: Data Transformation Functions
# ----------------------------------------------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new features (Age_Group, BMI_Category) from existing columns."""
    df_transformed = df.copy()
    
    # 1. Age Group
    age_bins = [0, 25, 40, 60, df_transformed['Age'].max() + 1]
    age_labels = ['Young', 'Middle-Aged', 'Senior', 'Elderly']
    df_transformed['Age_Group'] = pd.cut(df_transformed['Age'], bins=age_bins, labels=age_labels, right=False)
    
    # 2. BMI Category (Optional for this context, but good practice)
    # bmi_bins = [0, 18.5, 25, 30, df_transformed['BMI'].max() + 1]
    # bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
    # df_transformed['BMI_Category'] = pd.cut(df_transformed['BMI'], bins=bmi_bins, labels=bmi_labels, right=False)
    
    return df_transformed

def encode_features(df: pd.DataFrame, ordinal_col: str, labels: list) -> pd.DataFrame:
    """Applies Label Encoding to an ordinal feature."""
    df_encoded = df.copy()
    
    # Define the custom mapping for ordinal encoding
    mapping = {label: i for i, label in enumerate(labels)}
    df_encoded[ordinal_col + '_Encoded'] = df_encoded[ordinal_col].map(mapping)
    
    # Drop the original categorical column
    df_encoded = df_encoded.drop(columns=[ordinal_col])
    
    return df_encoded

def scale_data(df: pd.DataFrame, scaler_type: str = 'StandardScaler') -> (pd.DataFrame, object):
    """Applies a specified scaling method to all features (excluding the target 'Class')."""
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaler_type. Choose 'StandardScaler' or 'MinMaxScaler'.")
        
    X_scaled = scaler.fit_transform(X)
    
    # Convert back to DataFrame
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    df_scaled['Class'] = y.values
    
    return df_scaled, scaler

# ----------------------------------------------------------------------
# PHASE 4: Data Reduction Functions
# ----------------------------------------------------------------------
def feature_selection_kbest(X: pd.DataFrame, y: pd.Series, k: int = 5) -> pd.DataFrame:
    """Performs feature selection using SelectKBest with mutual information."""
    
    # Initialize SelectKBest
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()]
    
    # Return DataFrame with only selected features
    X_fs = X[selected_features]
    
    return X_fs

def perform_pca(X: pd.DataFrame, variance_threshold: float = 0.95) -> (pd.DataFrame, PCA):
    """Performs PCA and selects components to meet a minimum explained variance."""
    
    # Initialize PCA
    pca = PCA()
    pca.fit(X)
    
    # Determine the optimal number of components
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_optimal = np.where(cumulative_variance >= variance_threshold)[0][0] + 1
    
    # Re-run PCA with the optimal number of components
    pca_optimal = PCA(n_components=n_components_optimal)
    X_pca = pca_optimal.fit_transform(X)
    
    # Create DataFrame for PCA results
    pca_cols = [f'PC{i+1}' for i in range(n_components_optimal)]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)
    
    return df_pca, pca_optimal

# ----------------------------------------------------------------------
# PHASE 5: Data Imbalance Handling Functions
# ----------------------------------------------------------------------
def handle_imbalance_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> (pd.DataFrame, pd.Series):
    """Applies SMOTE to balance the target variable."""
    
    # Initialize SMOTE
    smote = SMOTE(random_state=random_state)
    
    # Apply SMOTE
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Convert X_resampled back to DataFrame with original column names
    X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    
    return X_resampled_df, y_resampled