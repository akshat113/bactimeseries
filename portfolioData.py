# Import Data
import pandas as pd
import numpy as np
#ETF Selected
#BCOMTR - Bloomberg Commodity Index Total Return
#LBUSTRUU - Bloomberg Barclays US Aggregate Bond Index
#RU20INTR - Russell 2000 Total Return
#RU10VATR - iShares Russell 1000 Value ETF

#Get etf names by ISIN

import investpy

BCOMTR = investpy.search_etfs(by="isin", value="US06738C7781")
LBUSTRUU = investpy.search_etfs(by="isin", value="IE00B459R192")
RU20INTR = investpy.search_etfs(by="isin", value="IE00B60SX402")
RU10VATR = investpy.search_etfs(by="isin", value="US4642876308")

#Downloading data
df1 = investpy.get_etf_historical_data(etf=BCOMTR['name'].values[0], country="United States",from_date="16/09/2011",to_date="26/09/2021")
df2 = investpy.get_etf_historical_data(etf=LBUSTRUU['name'].values[2], country="United Kingdom",from_date="16/09/2011",to_date="26/09/2021")
df3 = investpy.get_etf_historical_data(etf=RU20INTR['name'].values[2], country="United Kingdom",from_date="16/09/2011",to_date="26/09/2021")
df4 = investpy.get_etf_historical_data(etf=RU10VATR['name'].values[1], country="United States",from_date="16/09/2011",to_date="26/09/2021")

# Construction of portfolio
df = pd.concat([df1['Close'], df2['Close'], df3['Close'], df4['Close']], axis=1)
print(df.head())

returns = df['Close'].pct_change()

weights = np.full((4, 1), 0.25, dtype=float)
returns['portfolio'] = np.dot(returns, weights)

df = returns['portfolio'].to_frame()
df.to_csv("portfolio_returns.csv")
# Visualize data

import plotly.graph_objects as go

trace1 = go.Scatter(
    x=df.index,
    y=df['portfolio'],
    mode='lines',
    name='Data'
)

layout = go.Layout(
    title="Portfolio Returns",
    xaxis={'title': "Date"},
    yaxis={'title': "Return"}
)

fig = go.Figure(data=[trace1], layout=layout)

fig.update_layout(title_x=0.5)

fig.show()

# Create train and test data
return_data = df['portfolio'].values
return_data = return_data.reshape((-1, 1))

split_percent = 0.80
split = int(split_percent * len(return_data))

return_train = return_data[:split]
return_test = return_data[split:]

date_train = df.index[:split]
date_test = df.index[split:]

print(len(return_train))
print(len(return_test))