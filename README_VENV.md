# Virtual Environment Setup - Life Expectancy ML Project

## ✅ Virtual Environment Created Successfully!

Your virtual environment is ready with all dependencies installed.

## How to use:

### Option 1: Use the activation script (Recommended)
```bash
./activate_env.sh
```

### Option 2: Manual activation
```bash
# Activate virtual environment
source venv/bin/activate

# Start Jupyter
jupyter notebook --no-browser --port=8888
```

### Option 3: Use VS Code
1. Open VS Code in this directory
2. Select the Python interpreter from `venv/bin/python`
3. Open `notebooks/ml_modeling.ipynb`

## What's included:

✅ **Virtual Environment**: `venv/` directory
✅ **All Dependencies**: Installed from `requirements.txt`
✅ **Jupyter Notebook**: Ready to use
✅ **ML Scripts**: `ml_modeling.py`, `create_plots.py`
✅ **Data**: Clean dataset and models
✅ **Plots**: 7 professional visualizations

## Files structure:

```
equipo5_proyecto5/
├── venv/                          # Virtual environment
├── data/
│   ├── Life Expectancy Data.csv  # Original data
│   └── clean_data.csv            # Cleaned data
├── models/
│   ├── best_life_expectancy_model.pkl
│   ├── feature_importance.csv
│   └── model_results.json
├── plots/                        # Generated visualizations
├── notebooks/
│   └── ml_modeling.ipynb        # Main notebook
├── ml_modeling.py               # Python script version
├── create_plots.py              # Plot generation script
├── activate_env.sh              # Environment activation script
└── requirements.txt             # Dependencies
```

## Next steps:

1. **Activate the environment**: `./activate_env.sh`
2. **Open Jupyter**: Go to http://localhost:8888
3. **Open notebook**: `notebooks/ml_modeling.ipynb`
4. **Run cells**: Execute step by step

## Troubleshooting:

- **If Jupyter doesn't start**: Try `jupyter lab` instead
- **If port 8888 is busy**: Use `jupyter notebook --port=8889`
- **If VS Code doesn't find Python**: Select `venv/bin/python` as interpreter

## Project Status:

✅ **Data Cleaning**: Completed by Person 1
✅ **ML Modeling**: Completed by Person 4
✅ **Visualizations**: 7 plots generated
✅ **Model Performance**: R² = 0.969 (96.9% accuracy)
✅ **Ready for**: Streamlit integration, production deployment

---

**Person 4 - ML Modeling Team**
