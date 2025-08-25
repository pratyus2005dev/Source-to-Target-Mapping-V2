# Source-to-Target-Mapping-V2

## Overview

**Source-to-Target-Mapping-V2** is an advanced Python-based framework designed to automate and visualize the mapping of data elements from source systems to target systems. This tool is particularly useful for data integration, ETL processes, and data migration projects, where understanding and documenting the relationships between source and target data structures is crucial.

## Features

* **Dynamic System Configuration**: Easily configure source and target system names through the sidebar interface.
* **Interactive Dashboard**: Visualize mappings, match scores, and data relationships in an intuitive dashboard.
* **CSV and JSON Support**: Import mapping suggestions from CSV and JSON files.
* **Confidence Thresholding**: Filter mappings based on confidence scores to focus on high-quality matches.
* **Alternate Suggestions**: View alternate mapping suggestions to explore potential matches.
* **Downloadable Reports**: Export mapping results and summaries for documentation and further analysis.

## Requirements

Ensure you have the following installed:

* Python 3.8+
* pip (Python package installer)

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/pratyus2005dev/Source-to-Target-Mapping-V2.git
cd Source-to-Target-Mapping-V2
pip install -r requirements.txt
```

## Usage

Run the application using Streamlit:

```bash
streamlit run dashboard.py
```

This command will start a local server, and you can access the dashboard by navigating to the provided URL in your web browser.

### Input Files

* **CSV File**: Upload a `mapping_suggestions.csv` file containing the source and target mappings.
* **JSON File**: Optionally, upload a `mapping_suggestions.json` file for additional mapping details.

### Configuration

* **Source System**: Specify the name of the source system (e.g., "Guidewire").
* **Target System**: Specify the name of the target system (e.g., "InsureNow").

### Features in the Dashboard

* **Column Mapping Explorer**: Search for source or target columns to find their mappings.
* **Metrics Overview**: View total columns, matched columns, and match percentages.
* **Charts**: Visual representations of matched vs. unmatched columns, match rates by source table, and score distributions.
* **Low-Confidence Matches**: Identify mappings with low confidence scores for further review.
* **Top Matches**: View the highest confidence mappings.
* **Best Score Table**: Analyze mappings with the highest combined scores.
* **Table-to-Table Summary**: Summarize mappings between source and target tables.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-name`).
6. Create a new Pull Request.

Please ensure your code adheres to the existing style and includes appropriate tests.



Feel free to adjust the content as needed to better fit your project's specifics.
