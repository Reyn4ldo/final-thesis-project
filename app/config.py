"""
Configuration file for the AMR Pattern Recognition Streamlit app.

Contains paths, antibiotic lists, species names, and app settings.
"""

from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "reports" / "results"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

# Model files
MAR_MODEL_PATH = MODEL_DIR / "best_model_mar.pkl"
SPECIES_MODEL_PATH = MODEL_DIR / "best_model_species.pkl"
FEATURE_NAMES_PATH = DATA_DIR / "feature_names.json"
ENCODING_MAPPINGS_PATH = DATA_DIR / "encoding_mappings.json"

# Data files
CLEANED_DATA_PATH = DATA_DIR / "cleaned_data.csv"
X_TRAIN_PATH = DATA_DIR / "X_train.csv"
X_TEST_PATH = DATA_DIR / "X_test.csv"

# ============================================================================
# Antibiotic List (from feature_names.json)
# ============================================================================
ANTIBIOTICS = [
    'ampicillin',
    'amoxicillin/clavulanic_acid',
    'ceftaroline',
    'cefalexin',
    'cefalotin',
    'cefpodoxime',
    'cefotaxime',
    'cefovecin',
    'ceftiofur',
    'ceftazidime/avibactam',
    'imepenem',
    'amikacin',
    'gentamicin',
    'neomycin',
    'nalidixic_acid',
    'enrofloxacin',
    'marbofloxacin',
    'pradofloxacin',
    'doxycycline',
    'tetracycline',
    'nitrofurantoin',
    'chloramphenicol',
    'trimethoprim/sulfamethazole'
]

# Antibiotic categories for organization
ANTIBIOTIC_CATEGORIES = {
    'Beta-lactams': [
        'ampicillin',
        'amoxicillin/clavulanic_acid',
        'ceftaroline',
        'cefalexin',
        'cefalotin',
        'cefpodoxime',
        'cefotaxime',
        'cefovecin',
        'ceftiofur',
        'ceftazidime/avibactam',
        'imepenem'
    ],
    'Aminoglycosides': [
        'amikacin',
        'gentamicin',
        'neomycin'
    ],
    'Fluoroquinolones': [
        'nalidixic_acid',
        'enrofloxacin',
        'marbofloxacin',
        'pradofloxacin'
    ],
    'Tetracyclines': [
        'doxycycline',
        'tetracycline'
    ],
    'Other': [
        'nitrofurantoin',
        'chloramphenicol',
        'trimethoprim/sulfamethazole'
    ]
}

# ============================================================================
# Species List
# ============================================================================
SPECIES_NAMES = [
    'enterobacter_cloacae_complex',
    'klebsiella_pneumoniae_ssp_pneumoniae',
    'enterobacter_aerogenes',
    'escherichia_coli',
    'salmonella_group',
    'pseudomonas_aeruginosa',
    'vibrio_cholerae',
    'Other'
]

# Human-readable species names
SPECIES_DISPLAY_NAMES = {
    'enterobacter_cloacae_complex': 'Enterobacter cloacae complex',
    'klebsiella_pneumoniae_ssp_pneumoniae': 'Klebsiella pneumoniae ssp. pneumoniae',
    'enterobacter_aerogenes': 'Enterobacter aerogenes',
    'escherichia_coli': 'Escherichia coli',
    'salmonella_group': 'Salmonella group',
    'pseudomonas_aeruginosa': 'Pseudomonas aeruginosa',
    'vibrio_cholerae': 'Vibrio cholerae',
    'Other': 'Other species'
}

# ============================================================================
# Resistance Encoding
# ============================================================================
RESISTANCE_ENCODING = {
    'S': 0,  # Susceptible
    'I': 1,  # Intermediate
    'R': 2   # Resistant
}

RESISTANCE_LABELS = {
    0: 'Susceptible (S)',
    1: 'Intermediate (I)',
    2: 'Resistant (R)'
}

# ============================================================================
# MAR Thresholds
# ============================================================================
MAR_THRESHOLD = 0.2  # Threshold for classifying as high MAR

# ============================================================================
# App Settings
# ============================================================================
PAGE_TITLE = "AMR Pattern Recognition"
PAGE_ICON = "🦠"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# ============================================================================
# UI Settings
# ============================================================================
# Color scheme
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'warning': '#ff9800',
    'danger': '#d62728',
    'info': '#17a2b8'
}

# Chart settings
CHART_HEIGHT = 500
CHART_WIDTH = 700

# ============================================================================
# Model Settings
# ============================================================================
MODEL_TYPES = [
    'Random Forest',
    'XGBoost',
    'Logistic Regression',
    'SVM',
    'KNN',
    'Naive Bayes'
]

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    'high': 0.8,
    'medium': 0.6,
    'low': 0.0
}

# ============================================================================
# Example Data
# ============================================================================
EXAMPLE_PROFILES = {
    'Low Resistance': {
        'description': 'Isolate with minimal antibiotic resistance',
        'profile': {k: 0 for k in ANTIBIOTICS}
    },
    'High Resistance': {
        'description': 'Multi-drug resistant isolate',
        'profile': {k: 2 for k in ANTIBIOTICS}
    },
    'Mixed Profile': {
        'description': 'Isolate with varied resistance pattern',
        'profile': {
            'ampicillin': 2,
            'amoxicillin/clavulanic_acid': 2,
            'ceftaroline': 0,
            'cefalexin': 2,
            'cefalotin': 2,
            'cefpodoxime': 0,
            'cefotaxime': 0,
            'cefovecin': 0,
            'ceftiofur': 0,
            'ceftazidime/avibactam': 0,
            'imepenem': 0,
            'amikacin': 0,
            'gentamicin': 0,
            'neomycin': 0,
            'nalidixic_acid': 0,
            'enrofloxacin': 0,
            'marbofloxacin': 0,
            'pradofloxacin': 0,
            'doxycycline': 0,
            'tetracycline': 1,
            'nitrofurantoin': 0,
            'chloramphenicol': 0,
            'trimethoprim/sulfamethazole': 0
        }
    }
}

# ============================================================================
# Help Text
# ============================================================================
HELP_TEXT = {
    'resistance_input': """
    **Resistance Interpretation:**
    - **S (Susceptible)**: The antibiotic is effective against the bacterial isolate
    - **I (Intermediate)**: The isolate shows intermediate resistance; clinical efficacy is uncertain
    - **R (Resistant)**: The antibiotic is not effective against the bacterial isolate
    """,
    'mar_index': """
    **MAR (Multiple Antibiotic Resistance) Index:**
    - Calculated as: (Number of resistant antibiotics) / (Total antibiotics tested)
    - Range: 0.0 (no resistance) to 1.0 (resistant to all)
    - Threshold: {threshold} (isolates above this are classified as High MAR)
    """.format(threshold=MAR_THRESHOLD),
    'confidence': """
    **Prediction Confidence:**
    - **High (>80%)**: Model is very confident in the prediction
    - **Medium (60-80%)**: Model has moderate confidence
    - **Low (<60%)**: Model has low confidence; results should be interpreted cautiously
    """,
    'batch_upload': """
    **CSV Format Requirements:**
    - Each row represents one bacterial isolate
    - Columns should be named after antibiotics (see list below)
    - Values should be: 0 (S), 1 (I), or 2 (R)
    - Missing values will be handled automatically
    """
}
