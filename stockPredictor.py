import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

companies_df = pd.read_csv('sp500_companies.csv')
index_df = pd.read_csv('sp500_index.csv')
stocks_df = pd.read_csv('sp500_stocks.csv')

# Merge companies_df with stocks_df on Symbol
company_stock_df = pd.merge(companies_df, stocks_df, on='Symbol', how='inner')

# Merge company_stock_df with index_df on Date
final_df = pd.merge(company_stock_df, index_df, on='Date', how='inner')

# Convert Date to datetime
final_df["Date"] = pd.to_datetime(final_df["Date"])

# Checks moving 7-day and 30-day averages 
final_df['7-Day MA'] = final_df['Close'].rolling(window=7).mean()
final_df['30-Day MA'] = final_df['Close'].rolling(window=30).mean()

# Checks daily price and percent changes
final_df['Daily Change'] = final_df['Close'].diff()
final_df['Pct Change'] = final_df['Close'].pct_change()

# Checks the value of individual stocks against the value of the S&P500
final_df['Price_to_SP500'] = final_df['Close'] / final_df['S&P500']

# Drop empty rows
final_df.dropna(inplace=True)

# Features and target
features = ['Open', 'High', 'Low', 'Volume', 'S&P500', '7-Day MA', '30-Day MA', 'Daily Change', 'Pct Change', 'Price_to_SP500']
X = final_df[features]
y = final_df['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the scaler on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Prints the first few predictions and actual values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison_df.head())

# Plot Actual vs. Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Line for perfect prediction
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()