import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly

from plotly import graph_objs as go

#Timeline
START = "2016-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

#App Title
st.title("Stock Predition SOLFINTECH")
st.markdown('''
**Credits**
- App built by João Pedro Soares Bota
- Built in `Python` using `streamlit`,`datetime`,`yfinance`,`fbprophet` and `plotly`

''')


#Stocks Selection
stocks = ("AAPL","GOOGL","MSFT","NIO","TSLA","NVDA","LCID","CRSR","MRNA","PFE","BA","COIN","BABA","NFLX","AMC","RIVN","PTON","FB","NKE","KO")
select_stock = st.selectbox("Select Stock to predict", stocks)
period = st.slider("Days to predict:",1,365)

#Selected Stock Information
stockData = yf.Ticker(select_stock)
stock_logo = '<img src=%s>' %stockData.info['logo_url']
st.markdown(stock_logo, unsafe_allow_html=True)
stock_name = stockData.info['longName']
st.header(stock_name)

#Setup cache and download data from Yahoo Finance of the selected stock
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

#Load data as a table
data = load_data(select_stock)

st.subheader('Updated Data')
st.write(data.tail())

#Setup Graph for the highest and Lowest price of the selected stock
def graph_updated_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['High'], name='Highest Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Low'], name='Lowest Price'))
    fig.layout.update(title_text="Updated Data Graph", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

#Run the graph
graph_updated_data()


#Forecasting

dataframe_train = data[['Date', 'Close']]
dataframe_train = dataframe_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(dataframe_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Predicted Data')
st.write(forecast.tail())

fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

fig2 = m.plot_components(forecast)
st.write(fig2)
