# AMR Pattern Recognition using Machine Learning

A comprehensive machine learning project for identifying and predicting antimicrobial resistance (AMR) patterns in bacterial isolates.

## 📋 Project Overview

This thesis project develops and compares multiple machine learning approaches to:
- Identify patterns in antimicrobial resistance data
- Predict resistance profiles for bacterial isolates
- Discover associations between different antibiotic resistances
- Provide an interactive tool for AMR prediction

## 🎯 Objectives

1. Perform exploratory data analysis on AMR datasets
2. Apply unsupervised learning techniques (clustering, dimensionality reduction)
3. Develop supervised learning models for resistance prediction
4. Compare model performance and select the best approach
5. Deploy an interactive web application for predictions

## 📁 Project Structure

```
├── data/
│   ├── raw/                    # Original unprocessed data
│   │   └── rawdata.csv
│   ├── processed/              # Cleaned and encoded datasets
│   └── README.md               # Data dictionary and descriptions
├── notebooks/
│   ├── 00_data_exploration.ipynb
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_unsupervised_learning.ipynb
│   ├── 03_supervised_learning.ipynb
│   └── 04_model_comparison.ipynb
├── src/
│   ├── data/                   # Data processing modules
│   │   ├── preprocessing.py
│   │   └── splitting.py
│   ├── features/               # Feature engineering
│   │   └── build_features.py
│   ├── models/                 # ML models
│   │   ├── unsupervised.py
│   │   ├── supervised.py
│   │   └── evaluation.py
│   └── visualization/          # Plotting functions
│       └── plots.py
├── models/                     # Saved trained models
├── reports/
│   ├── figures/                # Generated plots
│   └── results/                # Model comparison tables
├── app/
│   ├── streamlit_app.py        # Web application
│   ├── utils.py                # App helper functions
│   └── README.md               # Deployment instructions
├── tests/                      # Unit tests
│   └── test_preprocessing.py
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
├── .gitignore
├── README.md
└── LICENSE
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Reyn4ldo/final-thesis-project.git
cd final-thesis-project
```

2. Create and activate a virtual environment:
```bash
# On macOS/Linux
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## 📊 Usage

### Phase 1: Data Exploration

Explore the AMR dataset to understand its structure and characteristics:

```bash
jupyter notebook notebooks/00_data_exploration.ipynb
```

### Phase 2: Data Preprocessing

Clean and prepare the data for modeling:

```bash
jupyter notebook notebooks/01_data_preprocessing.ipynb
```

### Phase 3: Unsupervised Learning

Apply clustering and dimensionality reduction:

```bash
jupyter notebook notebooks/02_unsupervised_learning.ipynb
```

### Phase 4: Supervised Learning

Train classification models:

```bash
jupyter notebook notebooks/03_supervised_learning.ipynb
```

### Phase 5: Model Comparison

Compare model performance and select the best:

```bash
jupyter notebook notebooks/04_model_comparison.ipynb
```

### Phase 6: Web Application

Run the Streamlit application:

```bash
cd app
streamlit run streamlit_app.py
```

## 🔬 Methodology

### Data Processing
- Data cleaning and missing value handling
- Categorical encoding (R/S/I → numerical)
- Feature normalization and scaling
- MAR (Multiple Antibiotic Resistance) index calculation

### Unsupervised Learning
- **Clustering**: K-Means, Hierarchical, DBSCAN
- **Dimensionality Reduction**: PCA, t-SNE, UMAP
- **Association Rules**: Apriori algorithm for pattern discovery

### Supervised Learning
- **Models**: Random Forest, XGBoost, Logistic Regression, SVM, KNN, Naive Bayes
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Validation**: Stratified K-fold cross-validation

## 📈 Results

Results will be documented in the `reports/` directory after running the notebooks:
- Model comparison tables
- Performance visualizations
- Feature importance analysis
- Cluster analysis results

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## 📦 Dependencies

Key libraries used:
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost
- **Visualization**: matplotlib, seaborn, plotly
- **Dimensionality Reduction**: umap-learn
- **Association Rules**: mlxtend
- **Web App**: streamlit
- **Testing**: pytest

See `requirements.txt` for complete list.

## 🔗 Related Issues

This project is organized into phases. See the following issues for detailed tasks:

1. [Phase 1: Data Exploration](#) - Initial EDA and data understanding
2. [Phase 2: Data Preprocessing](#) - Data cleaning and preparation
3. [Phase 3: Unsupervised Learning](#) - Clustering and pattern discovery
4. [Phase 4: Supervised Learning](#) - Classification model development
5. [Phase 5: Model Comparison](#) - Performance evaluation and selection
6. [Phase 6: Deployment](#) - Web application development

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**[Your Name]**
- Thesis Project
- [University Name]
- [Year]

## 🙏 Acknowledgments

- Dataset source: [To be specified]
- Advisors and supervisors
- [Any other acknowledgments]

## 📧 Contact

For questions or feedback, please contact: [your.email@example.com]

## 🔄 Project Status

This project is currently under development as part of a thesis research project.

---

**Note**: Replace placeholders (author name, university, contact info, etc.) with actual information before finalizing.
