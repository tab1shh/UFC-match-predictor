# UFC-match-predictor
A machine learning-based UFC match predictor that scrapes fight data, processes relevant statistics, and predicts match outcomes using Scikit-learn. 
The model currently achieves 60% accuracy based on historical fight data.

To know how the data was collected, head over to this repository: https://github.com/tab1shh/UFC-scraper

## Features
- Data Preprocessing: Filters and structures relevant fighter attributes.
- Machine Learning Model: Uses Scikit-learn to train a predictor on past fight results.
- Prediction Output: Provides fight outcome predictions based on input fighter stats.

## Getting Started
1. Clone the repository
```bash
git clone https://github.com/yourusername/UFC-match-predictor.git
cd UFC-match-predictor
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Run the predictor
```bash
python predictor.py
```

## Tech Stack
- Python
- Scikit-learn (ML model)
- BeautifulSoup (Web scraping)
- Pandas (Data processing)
- NumPy (Numerical computations)

## Model Performance
- Accuracy: ~60%
- Trained on: Historical UFC fight data

## Future Improvements
- Improve feature selection & preprocessing for better accuracy.
- Integrate real-time fight data updates.

## Contributing 
Feel free to fork the repo and submit PRs!