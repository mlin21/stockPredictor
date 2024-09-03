# Stock Price Predictor for S&P 500

This project is a stock price predictor that uses machine learning to forecast the future prices of stocks in the S&P 500. The predictor is built with Python, using the `scikit-learn` library, and deployed using Django.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Model](#model)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The Stock Price Predictor is designed to analyze historical stock prices and predict future stock prices for companies listed in the S&P 500. The project uses a linear regression model to provide predictions based on various features such as stock prices, volume, and moving averages.

## Dataset
The project uses a combination of datasets from Kaggle, focusing on:
- **S&P 500 Companies**: Company-specific details such as sector, industry, and financial metrics.
- **S&P 500 Index**: Historical prices of the S&P 500 index.
- **S&P 500 Stocks**: Historical stock prices of individual companies in the S&P 500.

### Columns Used:
- **Stocks Dataset**: Date, Symbol, Adj Close, Close, High, Low, Open, Volume
- **Companies Dataset**: Symbol, Sector, Industry, Market Cap, etc.
- **Index Dataset**: Date, S&P 500 value

## Features
The project extracts and calculates the following features for prediction:
- **7-Day Moving Average** and **30-Day Moving Average** of stock prices.
- **Daily Price Change** and **Percentage Change**.
- **Price to S&P 500 Ratio**.

## Model
The machine learning model used in this project is a Linear Regression model. The model is trained on historical stock prices and predicts future prices based on the extracted features.

### Evaluation Metrics:
- **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
- **R-squared Score (R²)**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

## Usage
To use the stock price predictor, follow these steps:

1. Enter the stock symbol and date range for prediction.
2. Click "Predict" to get the future stock price based on the machine learning model.

## Future Improvements
- Implement more advanced models such as LSTM (Long Short-Term Memory) for time series prediction.
- Improve the UI/UX of the web application.

## Contributing
Contributions are welcome! Please create a pull request or open an issue to discuss any changes or improvements.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.