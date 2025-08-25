# Source-to-Target-Mapping-V2

# Data Mapping and Synonym Suggestion Tool

## Overview
This project provides a comprehensive solution for mapping and suggesting synonyms between data fields from different insurance data sources. It leverages machine learning and rule-based techniques to automate the mapping process, making it easier to integrate and analyze data from multiple systems.

## Features
- **Automated Field Mapping:** Suggests mappings between fields from different datasets using a trained ML model.
- **Synonym Generation:** Generates and manages synonyms for field names to improve mapping accuracy.
- **Configurable Pipeline:** Easily adjust data sources, model parameters, and synonym lists via configuration files.
- **Dashboard:** Visualize mapping suggestions and review results interactively.
- **Extensible:** Modular codebase for easy extension to new data sources or mapping strategies.

## Project Structure
```
dashboard.py                # Dashboard for visualizing mapping suggestions
DDL.sql                     # Example DDL for database schema
generate_synonyms.py        # Script to generate synonyms for field names
main.py                     # Main entry point for running the pipeline
requirements.txt            # Python dependencies
configs/
    config.yaml             # Main configuration file
    synonyms.json           # Synonym dictionary
models/
    matcher.pkl             # Trained ML model for field matching
outputs/
    mapping_suggestions.*   # Output files (CSV, JSON) with mapping results
data/
    guidewire/              # Guidewire data CSVs
    insurenow/              # InsureNow data CSVs
src/
    config.py               # Config loader
    data_loader.py          # Data loading utilities
    ddl_parser.py           # DDL parsing logic
    featurizer.py           # Feature engineering for ML model
    model.py                # ML model definition and utilities
    predict.py              # Prediction logic for mapping
    train.py                # Model training script
    utils.py                # Helper functions
```

## Setup
1. **Clone the repository**
   ```sh
   git clone <repo-url>
   cd <repo-folder>
   ```
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Configure the project**
   - Edit `configs/config.yaml` to set data paths and parameters.
   - Update `configs/synonyms.json` with custom synonyms if needed.

## Usage
### 1. Train the Model
Train the matcher model using the provided training script:
```sh
python src/train.py
```

### 2. Generate Synonyms
Generate or update synonyms for field names:
```sh
python generate_synonyms.py
```

### 3. Run the Mapping Pipeline
Run the main pipeline to generate mapping suggestions:
```sh
python main.py
```

### 4. View Results
- Check the `outputs/` directory for mapping suggestions in CSV and JSON formats.
- Launch the dashboard for interactive review:
  ```sh
  python dashboard.py
  ```

## Configuration
- **config.yaml:** Controls data paths, model parameters, and other settings.
- **synonyms.json:** Stores field name synonyms for improved matching.

## Data
- Place your source CSV files in the appropriate subfolders under `data/guidewire/` and `data/insurenow/`.
- Update the config file if you add new data sources.

## Extending the Project
- Add new data loaders in `src/data_loader.py` for additional formats.
- Extend feature engineering in `src/featurizer.py` for better model performance.
- Update or retrain the model using `src/train.py` as needed.

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies.


## Contact
For questions or contributions, please contact [Your Name] or open an issue in the repository.
