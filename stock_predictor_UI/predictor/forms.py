from django import forms

class StockForm(forms.Form):
    # symbol = forms.CharField(label='Stock Symbol', max_length=10)
    # start_date = forms.DateField(label='Start Date')
    # end_date = forms.DateField(label='End Date')
    open = forms.FloatField(label='Open Value')
    high = forms.FloatField(label='High Value')
    low = forms.FloatField(label='Low Value')
    volume = forms.FloatField(label='Volume')
    sp500 = forms.FloatField(label='S&P500 Value')