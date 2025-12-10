# Streamlit Deployment Guide

## Overview

This directory contains the Streamlit web application for the AMR Pattern Recognition system.

## Files

- `streamlit_app.py`: Main application file
- `utils.py`: Helper functions for the app
- `README.md`: This file

## Installation

1. Ensure you have installed all requirements:
```bash
pip install -r ../requirements.txt
```

2. Make sure trained models are available in the `../models/` directory

## Running the Application

### Local Development

Run the app locally:

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Production Deployment

#### Option 1: Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app/streamlit_app.py` as the main file
5. Deploy!

#### Option 2: Heroku

1. Create a `Procfile` in the root directory:
```
web: sh setup.sh && streamlit run app/streamlit_app.py
```

2. Create `setup.sh`:
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

3. Deploy to Heroku:
```bash
heroku create your-app-name
git push heroku main
```

#### Option 3: Docker

1. Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py"]
```

2. Build and run:
```bash
docker build -t amr-app .
docker run -p 8501:8501 amr-app
```

## Application Features

### 1. Home Page
- Overview of the system
- Key features and statistics
- Quick start guide

### 2. Data Upload
- Upload CSV files with AMR data
- View data preview and statistics
- Data quality checks

### 3. Make Predictions
- Choose from multiple trained models
- Manual input or batch prediction
- View prediction results and probabilities

### 4. Model Performance
- Compare all models
- View metrics and visualizations
- ROC curves and confusion matrices

### 5. About
- Project information
- Methodology overview
- Contact details

## Customization

### Adding New Features

1. Edit `streamlit_app.py` to add new pages or sections
2. Add helper functions to `utils.py`
3. Update this README with new features

### Styling

- Modify the CSS in the `st.markdown()` section of `streamlit_app.py`
- Customize colors, fonts, and layout

## Troubleshooting

### Common Issues

1. **Models not found**: Ensure models are trained and saved in `../models/`
2. **Import errors**: Check that all dependencies are installed
3. **Port already in use**: Change the port in Streamlit settings

### Debug Mode

Run with verbose logging:
```bash
streamlit run streamlit_app.py --logger.level=debug
```

## Performance Optimization

- Use `@st.cache_data` for data loading functions
- Use `@st.cache_resource` for model loading
- Limit data preview rows for large datasets
- Use pagination for large tables

## Security Considerations

- Validate all user inputs
- Sanitize uploaded files
- Don't expose sensitive model details
- Use HTTPS in production
- Implement authentication if needed

## Future Enhancements

- [ ] Add user authentication
- [ ] Implement model retraining interface
- [ ] Add real-time monitoring dashboard
- [ ] Export prediction reports as PDF
- [ ] Multi-language support
- [ ] Advanced filtering and search

## Support

For issues or questions:
- Check the main project README
- Review Streamlit documentation: https://docs.streamlit.io
- Contact the project maintainer

## License

Same as the main project (MIT License)
