import pandas as pd
import joblib
from django.shortcuts import render
from .forms import StockForm

# Load pre-trained model and scaler
def load_model():
    model = joblib.load('linear_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

def predict_stock(request):
    if request.method == 'POST':
        form = StockForm(request.POST)
        if form.is_valid():
            # Extract cleaned data from form
            # symbol = form.cleaned_data['symbol']
            # start_date = form.cleaned_data['start_date']
            # end_date = form.cleaned_data['end_date']
            
            # Load the model and scaler
            model, scaler = load_model()

            # Retrieve input values from form data
            try:
                open_value = float(request.POST.get('open'))
                high_value = float(request.POST.get('high'))
                low_value = float(request.POST.get('low'))
                volume_value = float(request.POST.get('volume'))
                sp500_value = float(request.POST.get('sp500'))
                
                # Prepare input data for prediction
                input_data = pd.DataFrame([[open_value, high_value, low_value, volume_value, sp500_value]], 
                                          columns=['Open', 'High', 'Low', 'Volume', 'S&P500'])

                # Scale the input data
                input_data_scaled = scaler.transform(input_data)

                # Predict using the model
                prediction = model.predict(input_data_scaled)[0]
                
                return render(request, 'predictor/results.html', {'prediction': prediction})
            except Exception as e:
                # Handle any errors that occur during prediction
                return render(request, 'predictor/results.html', {'prediction': 'Error in prediction: {}'.format(e)})
    else:
        form = StockForm()

    return render(request, 'predictor/predict.html', {'form': form})
