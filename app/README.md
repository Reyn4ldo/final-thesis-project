# AMR Pattern Recognition - Streamlit App

## Overview

This directory contains the complete Streamlit web application for the AMR (Antimicrobial Resistance) Pattern Recognition system. The app provides an interactive interface for predicting multi-drug resistance, identifying bacterial species, and exploring resistance patterns.

## Features

### 🏠 Home Page
- Project overview and introduction
- Quick start guide
- System statistics
- Model status indicators

### 🦠 MDR Prediction
- Predict Multi-Drug Resistance (MDR) status
- Manual input or CSV upload
- Confidence scores and MAR index
- Clinical interpretation guidance

### 🔬 Species Prediction
- Identify bacterial species from resistance patterns
- Top 3 predictions with probabilities
- Species-specific information
- Resistance pattern analysis

### 📊 Batch Prediction
- Process multiple isolates from CSV
- Both MDR and species predictions
- Download results
- Summary visualizations

### 📈 Model Insights
- Feature importance analysis
- Model performance comparison
- Confusion matrices
- Which antibiotics matter most

### 🗺️ Data Explorer
- UMAP/t-SNE/PCA visualizations
- Cluster analysis
- Species distribution
- Interactive data filtering

## File Structure

```
app/
├── streamlit_app.py          # Main entry point
├── config.py                 # Configuration and settings
├── utils.py                  # Helper functions
├── components.py             # Reusable UI components
├── pages/
│   ├── 1_🏠_Home.py         # Home page
│   ├── 2_🦠_MDR_Prediction.py       # MDR prediction interface
│   ├── 3_🔬_Species_Prediction.py   # Species prediction interface
│   ├── 4_📊_Batch_Prediction.py     # Batch processing
│   ├── 5_📈_Model_Insights.py       # Model performance
│   └── 6_🗺️_Data_Explorer.py       # Data visualization
├── examples/
│   ├── sample_input.csv      # Example CSV file
│   └── sample_single.json    # Example single profile
└── README.md                 # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Trained models (from Phase 0-2 notebooks)

### Setup

1. **Install dependencies:**

```bash
cd /path/to/final-thesis-project
pip install -r requirements.txt
```

2. **Ensure models are trained:**

The app requires trained models in the `models/` directory:
- `best_model_mar.pkl` - MDR prediction model
- `best_model_species.pkl` - Species prediction model

To train models, run the Jupyter notebooks in order:
- `notebooks/01_data_preprocessing.ipynb`
- `notebooks/03_supervised_learning.ipynb`

3. **Verify data files:**

Required data files in `data/processed/`:
- `feature_names.json` - List of feature names
- `encoding_mappings.json` - Encoding configuration
- `cleaned_data.csv` - Processed dataset (for Data Explorer)

## Running the Application

### Local Development

Run the app locally:

```bash
# From the project root directory
streamlit run app/streamlit_app.py

# Or from the app directory
cd app
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Command Line Options

```bash
# Specify a different port
streamlit run app/streamlit_app.py --server.port 8502

# Run in headless mode (for servers)
streamlit run app/streamlit_app.py --server.headless true

# Enable debug mode
streamlit run app/streamlit_app.py --logger.level=debug
```

## Deployment

### Option 1: Streamlit Community Cloud

1. **Push to GitHub:**
```bash
git add .
git commit -m "Add Streamlit app"
git push origin main
```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app/streamlit_app.py`
   - Click "Deploy"

3. **Configure secrets (if needed):**
   - Add any API keys or secrets in the Streamlit Cloud dashboard
   - Settings → Secrets

### Option 2: HuggingFace Spaces

1. **Create a new Space:**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Select "Streamlit" as SDK
   - Name your space

2. **Upload files:**
   - Clone the space repository
   - Copy app files to the repository
   - Create `requirements.txt` in the root
   - Push changes

3. **Required files for HuggingFace:**
```
your-space/
├── app.py (copy of streamlit_app.py)
├── config.py
├── utils.py
├── components.py
├── pages/
├── examples/
├── models/
├── data/processed/
└── requirements.txt
```

### Option 3: Docker

1. **Create `Dockerfile`:**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the app
ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. **Build and run:**

```bash
# Build image
docker build -t amr-app .

# Run container
docker run -p 8501:8501 amr-app

# Or with docker-compose
docker-compose up
```

### Option 4: Heroku

1. **Create required files:**

`Procfile`:
```
web: sh setup.sh && streamlit run app/streamlit_app.py
```

`setup.sh`:
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

2. **Deploy:**

```bash
heroku create your-app-name
git push heroku main
heroku open
```

## Usage Guide

### Making Single Predictions

1. Navigate to MDR Prediction or Species Prediction page
2. Choose input method:
   - **Manual Entry**: Select S/I/R for each antibiotic
   - **Upload CSV**: Upload a file with resistance data
   - **Example Data**: Use pre-defined profiles
3. Click "Predict" button
4. Review results and confidence scores

### Batch Processing

1. Prepare a CSV file with columns for each antibiotic
2. Each row should represent one isolate
3. Values: 0 (Susceptible), 1 (Intermediate), 2 (Resistant)
4. Go to Batch Prediction page
5. Upload your CSV file
6. Select prediction type
7. Download results

### CSV Format

```csv
isolate_id,ampicillin,gentamicin,ciprofloxacin,...
ISO001,2,0,1,...
ISO002,0,0,0,...
ISO003,2,2,2,...
```

**Accepted values:**
- `0` or `S` = Susceptible
- `1` or `I` = Intermediate  
- `2` or `R` = Resistant

### Exploring Model Insights

1. Go to Model Insights page
2. Select MDR or Species model
3. View feature importance chart
4. Review performance metrics
5. Understand which antibiotics drive predictions

### Visualizing Data

1. Go to Data Explorer page
2. Select dimensionality reduction method (UMAP/t-SNE/PCA)
3. Choose coloring (species, MAR status, cluster)
4. Explore patterns and clusters
5. Filter and download data

## Configuration

Edit `app/config.py` to customize:

- **Paths**: Model and data file locations
- **Antibiotics**: List of antibiotics to include
- **Species**: Species classes
- **Thresholds**: MAR threshold for classification
- **UI Settings**: Colors, chart sizes, page layout

## Troubleshooting

### Common Issues

**1. Models not found**
- Ensure models are trained and saved in `models/` directory
- Check file names match config (e.g., `best_model_mar.pkl`)
- Run training notebooks if models don't exist

**2. Import errors**
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+)
- Ensure you're running from the correct directory

**3. Data not loading**
- Verify `data/processed/` contains required files
- Run preprocessing notebook if files are missing
- Check file permissions

**4. Port already in use**
- Change port: `streamlit run app/streamlit_app.py --server.port 8502`
- Kill existing process: `pkill -f streamlit`

**5. Upload size limit**
- Streamlit default: 200MB
- To increase: Add to `.streamlit/config.toml`:
  ```toml
  [server]
  maxUploadSize = 500
  ```

### Debug Mode

Run with verbose logging:

```bash
streamlit run app/streamlit_app.py --logger.level=debug
```

### Clear Cache

If experiencing issues with cached data:

```bash
# Clear Streamlit cache
streamlit cache clear
```

Or use the app menu: ☰ → Settings → Clear cache

## Performance Optimization

### Caching

The app uses Streamlit caching decorators:

- `@st.cache_resource` - For loading models (persists across sessions)
- `@st.cache_data` - For loading data files

### Large Datasets

For datasets with > 1000 isolates:
- Consider pagination in data tables
- Use batch processing in chunks
- Implement sampling for visualizations

### Production Considerations

- Enable HTTPS in production
- Implement user authentication if needed
- Set up monitoring and logging
- Use environment variables for sensitive config
- Implement rate limiting for API calls

## Security

### Best Practices

- **Never commit sensitive data** to version control
- **Sanitize user inputs** before processing
- **Validate uploaded files** for correct format
- **Use HTTPS** in production
- **Implement authentication** for sensitive deployments
- **Set upload size limits** to prevent abuse

### HIPAA Compliance

If handling patient data:
- Deploy on HIPAA-compliant infrastructure
- Implement proper access controls
- Encrypt data at rest and in transit
- Maintain audit logs
- Ensure data retention policies

## Contributing

To add new features:

1. Create new page in `app/pages/`
2. Follow naming convention: `N_emoji_PageName.py`
3. Import shared components from `components.py`
4. Add helper functions to `utils.py`
5. Update this README

## Testing

Run basic tests:

```bash
# Test imports
python -c "from app import config, utils, components"

# Test model loading
python -c "from app.utils import load_model; from app.config import MAR_MODEL_PATH; load_model(MAR_MODEL_PATH)"

# Test app startup
streamlit run app/streamlit_app.py --server.headless true &
sleep 5
curl http://localhost:8501
```

## Support

For issues or questions:
- Check this README
- Review [Streamlit documentation](https://docs.streamlit.io)
- Check project issues on GitHub
- Contact the project maintainer

## License

Same as the main project (MIT License)

## Acknowledgments

- Streamlit for the framework
- Plotly for visualizations
- scikit-learn for ML models
- The thesis supervisor and advisors

---

**Version**: 1.0.0  
**Last Updated**: December 2024  
**Status**: Production Ready
