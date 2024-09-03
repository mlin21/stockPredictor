from django import forms

class StockForm(forms.Form):
    symbol = forms.CharField(label='Stock Symbol', max_length=10)
    start_date = forms.DateField(label='Start Date')
    end_date = forms.DateField(label='End Date')
