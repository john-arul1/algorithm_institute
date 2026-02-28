# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 21:36:37 2025

@author: DELLL

Deepseek generated

Relative Strength Index (RSI), Money Flow Index (MFI)
Moving MCAD
app.y 
components.py
requirements.txt

pip install -r requirements.txt
streamlit run app.py

"""

import streamlit as st

#st.markdown('<a href="/my_page" target="_self">Go to my page in the same tab</a>', unsafe_allow_html=True)

import pandas as pd
import numpy as np
import json
import yfinance as yf
from datetime import datetime

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import the components
from core.components import portfolio_allocator, alert_system, performance_tracker

# ... (keep all the existing TradingPlatform class and functions)


# Set page configuration
st.set_page_config(
    page_title="Stock Trading Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


"""
Stock list is loaded from json file "tickers.json", modify if required

"""

class TradingPlatform:
    name= "DeepSeek"   # class variable  Classname.name or self.__class__.name
    
    def __init__(self):
    
        self.mfi_overbought=70    # instance variables
        self.mfi_oversold=30
        self.name_data_dict={}
        #self.window = 14 # averaging window
        self.tickers = {
            'nifty': '^NSEI',
            'gold': 'GC=F',
            'stocks':{}   # dictionary of stock and ticker to be read from a file 
        }
    
    def load_tickers(self,file):
         with open(file,'r') as f:
             data=json.load(f)
             if data is not None:
                self.tickers.update(data)        
         return
     
        
    def fetch_data(self, ticker, period='1y',interval='1d'):
        """Fetch historical data for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period,interval=interval)
            #data = stock.download(period=period,interval=interval)
            return data
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def calculate_rsi(self, prices, window=5):
        """Calculate RSI indicator"""
        avg = (prices["High"] + prices["Low"] + prices["Close"]) / 3
        delta = avg.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 1.0E-5)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_mfi(self, data, window=7):
        """Calculate Money Flow Index"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
        
        positive_mf = pd.Series(positive_flow).rolling(window=window).sum()
        negative_mf = pd.Series(negative_flow).rolling(window=window).sum()
        positive_mf.index = typical_price.index.copy()
        negative_mf.index = typical_price.index.copy()
        
        mfr = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + mfr))
        return mfi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_gold_nifty_ratio(self, gold_data, nifty_data):
        """Calculate Gold to NIFTY RSI ratio"""
        gold_rsi = self.calculate_rsi(gold_data)
        nifty_rsi = self.calculate_rsi(nifty_data)
        
        ratio = gold_rsi / nifty_rsi
        
        return ratio, gold_rsi, nifty_rsi
    
    def generate_signals(self, stock_data, gold_nifty_ratio):
        """Generate trading signals based on all indicators"""
        signals = pd.DataFrame(index=stock_data.index)
        
        signals['RSI'] = self.calculate_rsi(stock_data)
        signals['MFI'] = self.calculate_mfi(stock_data)
        signals['MACD'], signals['MACD_Signal'], signals['MACD_Hist'] = self.calculate_macd(stock_data['Close'])
        signals['Gold_Nifty_Ratio'] = gold_nifty_ratio
        
        signals['Signal'] = 'HOLD'
        
        # Enhanced signal logic          mfi_oversold/bought set by selector
        buy_condition = (
            (signals['RSI'] < 35) &
            (signals['MFI'] < self.mfi_oversold) &
            (signals['MACD'] > signals['MACD_Signal']) &
            (signals['Gold_Nifty_Ratio'] < 1.2)
        )
        
        sell_condition = (
            (signals['RSI'] > 70) |
            (signals['MFI'] > self.mfi_overbought) |
            (signals['MACD'] < signals['MACD_Signal']) |
            (signals['Gold_Nifty_Ratio'] > 1.8)
        )
        
        signals.loc[buy_condition, 'Signal'] = 'BUY'
        signals.loc[sell_condition, 'Signal'] = 'SELL'
        
        # Add signal strength
        signals['Signal_Strength'] = self.calculate_signal_strength(signals)
        
        return signals
    
    #@st.cache_data    
    def calculate_signal_strength(self, nsignals):
        """Calculate signal strength from 0 to 100"""
        strength = np.zeros(len(nsignals))

        signals=nsignals.reset_index()  # index is RangeIndex 
        
        for i,row in signals.iterrows():
            score = 50  # Neutral starting point
            
            # RSI contribution
            if row['RSI'] < 30:
                score += 20
            elif row['RSI'] > 70:
                score -= 20
            elif 30 <= row['RSI'] <= 50:
                score += 10
                
            # MFI contribution  
            if row['MFI'] < 25:
                score += 15
            elif row['MFI'] > 75:
                score -= 15
                
            # MACD contribution
            if row['MACD'] > row['MACD_Signal']:
                score += 10
            else:
                score -= 10
                
            # Gold/Nifty ratio contribution
            if row['Gold_Nifty_Ratio'] < 0.9:
                score += 5
            elif row['Gold_Nifty_Ratio'] > 1.5:
                score -= 5
                
            strength[i] = max(0, min(100, score))
            
        return strength



def main():
    # Initialize platform
    platform = TradingPlatform()
    
    platform.load_tickers("./data/tickers.json") 
    
    # Sidebar for user inputs
    st.sidebar.title("🎯 Trading Configuration")
    
    # Stock selection
    selected_stocks = st.sidebar.multiselect(
        "Select Stocks to Analyze:",
        options=list(platform.tickers['stocks'].keys()),
        default=['RELIANCE', 'TCS', 'HDFC BANK']
    )
    
    # Time period selection
    time_period = st.sidebar.selectbox(
        "Select Time Period:",
        options=['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y'],
        index=2
    )
    
    # Analysis parameters
    st.sidebar.subheader("Money Flow Index Parameters")
    platform.mfi_overbought = st.sidebar.slider("MFI Level:", 70, 100, 70,step=1)
    platform.mfi_oversold = st.sidebar.slider("MFI Oversold Level:", 0, 30, 30,step=1)
    
    # Risk appetite
    risk_appetite = st.sidebar.radio(
        "Risk Appetite:",
        ['Conservative', 'Moderate', 'Aggressive'],
        index=1
    )
    
    # ✅ INTEGRATE THE COMPONENTS
    st.sidebar.markdown("---")
    portfolio_config = portfolio_allocator()
    alert_enabled = alert_system()
    performance_config = performance_tracker()
    
    # Main content area
    st.title("📈 Stock Trading Advisor Platform")
    st.markdown("---")
    
    # Display portfolio info if configured
    if portfolio_config:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Size", f"₹{portfolio_config['initial_capital']:,}")
        with col2:
            st.metric("Max Position", f"{portfolio_config['max_position_size']}%")
        with col3:
            st.metric("Stop Loss", f"{portfolio_config['stop_loss']}%")
    
    # Analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Analyzing market data..."):
            analyze_stocks(platform, selected_stocks, time_period,risk_appetite, 
                          portfolio_config, alert_enabled, performance_config)
            
            
#End of main            
            



def analyze_stocks(platform, selected_stocks, time_period, risk_appetite, 
                  portfolio_config=None, alert_enabled=False, performance_config=None):
    """Perform analysis and display results"""
    
    # Fetch market data
    nifty_data = platform.fetch_data(platform.tickers['nifty'], time_period,"1d")
    gold_data = platform.fetch_data(platform.tickers['gold'], time_period,"1d")
    
    if nifty_data is None or gold_data is None:
        st.error("Failed to fetch market data. Please try again.")
        return
    
    # this function changes the key format
    df_nifty,df_gold=robust_data_alignment(nifty_data,gold_data, primary_column='Close')

    # Calculate Gold to NIFTY ratio
    gold_nifty_ratio, gold_rsi, nifty_rsi = platform.calculate_gold_nifty_ratio(df_gold, df_nifty)
    
    # Market Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("NIFTY 50 RSI", f"{nifty_rsi.iloc[-1]:.2f}")
    with col2:
        st.metric("Gold RSI", f"{gold_rsi.iloc[-1]:.2f}")
    with col3:
        ratio_value = gold_nifty_ratio.iloc[-1]
        st.metric("Gold/NIFTY RSI Ratio", f"{ratio_value:.2f}")
    with col4:
        market_sentiment = "Risk-Off" if ratio_value > 1.2 else "Risk-On"
        st.metric("Market Sentiment", market_sentiment)
    
    st.markdown("---")
    
    # Stock Analysis
    st.subheader("📊 Stock Analysis Results")
    
    results = []
    for stock_name in selected_stocks:
        stock_ticker = platform.tickers['stocks'][stock_name]
        stock_data = platform.fetch_data(stock_ticker, time_period,'1d')
        platform.name_data_dict[stock_name]=stock_data  # store fetched data of stocks

        if stock_data is not None:
            signals = platform.generate_signals(stock_data, gold_nifty_ratio)
            current_signal = signals['Signal'].iloc[-1]
            current_strength = signals['Signal_Strength'].iloc[-1]
            current_price = stock_data['Close'].iloc[-1]
            
            # Calculate position size if portfolio config exists
            position_size = 0
            if portfolio_config and current_signal == 'BUY':
                position_size = (portfolio_config['initial_capital'] * 
                               portfolio_config['max_position_size'] / 100) / current_price
            
            results.append({
                'Stock': stock_name,
                'Current Price': f"₹{current_price:.2f}",
                'RSI': f"{signals['RSI'].iloc[-1]:.2f}",
                'MFI': f"{signals['MFI'].iloc[-1]:.2f}",
                'MACD': f"{signals['MACD'].iloc[-1]:.3f}",
                'Signal': current_signal,
                'Strength': current_strength,
                'Position Size': f"{int(position_size)} shares" if position_size > 0 else "N/A"
            })
    
    # Display results in a table
    if results:
        results_df = pd.DataFrame(results)
        
        # Color coding for signals
        def color_signal(val):
            if val == 'BUY':
                return 'background-color: #90EE90'
            elif val == 'SELL':
                return 'background-color: #FFB6C1'
            else:
                return 'background-color: #FFFFE0'
        
        styled_df = results_df.style.applymap(color_signal, subset=['Signal'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Show alerts if enabled
        if alert_enabled:
            st.info("🔔 Alerts are enabled. You will be notified of significant signals.")
        
        # Visualizations for each selected stock
        st.subheader("📈 Detailed Analysis Charts")
        
        tabs = st.tabs(selected_stocks)
        
        for i, stock_name in enumerate(selected_stocks):
            with tabs[i]:
        #         stock_ticker = platform.tickers['stocks'][stock_name]
        #         stock_data = platform.fetch_data(stock_ticker, time_period)              
        #        if stock_data is not None:
        
                stock_data=platform.name_data_dict[stock_name]
                create_interactive_charts(platform, stock_data, stock_name, gold_nifty_ratio)
    
    # Trading Recommendations with portfolio context
    st.markdown("---")
    st.subheader("🎯 Trading Recommendations")
    
    if results and portfolio_config:
        st.write(f"**Portfolio Context:** ₹{portfolio_config['initial_capital']:,} | Max Position: {portfolio_config['max_position_size']}% | Stop Loss: {portfolio_config['stop_loss']}%")
    
    if results:
        buy_recommendations = [r for r in results if r['Signal'] == 'BUY']
        sell_recommendations = [r for r in results if r['Signal'] == 'SELL']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🟢 Strong Buy Opportunities")
            if buy_recommendations:
                for rec in sorted(buy_recommendations, key=lambda x: x['Strength'], reverse=True):
                    st.write(f"**{rec['Stock']}** - Strength: {rec['Strength']:.0f}/100")
                    st.write(f"  Price: {rec['Current Price']} | RSI: {rec['RSI']} | MFI: {rec['MFI']}")
                    if portfolio_config:
                        st.write(f"  Recommended: {rec['Position Size']}")
            else:
                st.write("No strong buy signals detected.")
        
        with col2:
            st.write("### 🔴 Consider Selling")
            if sell_recommendations:
                for rec in sorted(sell_recommendations, key=lambda x: x['Strength']):
                    st.write(f"**{rec['Stock']}** - Strength: {rec['Strength']:.0f}/100")
                    st.write(f"  Price: {rec['Current Price']} | RSI: {rec['RSI']} | MFI: {rec['MFI']}")
            else:
                st.write("No strong sell signals detected.")




# ... (keep the existing create_interactive_charts function)
def create_interactive_charts(platform, stock_data, stock_name, gold_nifty_ratio):
    """Create interactive Plotly charts for a stock"""
    
    signals = platform.generate_signals(stock_data, gold_nifty_ratio)
    
    # Create subplots from plotly
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            f'{stock_name} - Price Chart with Signals',
            'RSI & MFI Indicators',
            'MACD Indicator',
            'Gold to NIFTY RSI Ratio'
        ),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(x=stock_data.index, y=stock_data['Close'], 
                  name='Price', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Add buy/sell signals
    buy_signals = signals[signals['Signal'] == 'BUY']
    sell_signals = signals[signals['Signal'] == 'SELL']
    
    fig.add_trace(
        go.Scatter(x=buy_signals.index, 
                  y=stock_data.loc[buy_signals.index, 'Close'],
                  mode='markers', name='Buy Signal',
                  marker=dict(color='green', size=10, symbol='triangle-up')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=sell_signals.index, 
                  y=stock_data.loc[sell_signals.index, 'Close'],
                  mode='markers', name='Sell Signal',
                  marker=dict(color='red', size=10, symbol='triangle-down')),
        row=1, col=1
    )
    
    # RSI and MFI
    fig.add_trace(
        go.Scatter(x=signals.index, y=signals['RSI'], 
                  name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=signals.index, y=signals['MFI'], 
                  name='MFI', line=dict(color='orange')),
        row=2, col=1
    )
    
    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(
        go.Scatter(x=signals.index, y=signals['MACD'], 
                  name='MACD', line=dict(color='blue')),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=signals.index, y=signals['MACD_Signal'], 
                  name='Signal', line=dict(color='red')),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Bar(x=signals.index, y=signals['MACD_Hist'], 
               name='Histogram', marker_color='gray'),
        row=3, col=1
    )
    
    # Gold/NIFTY Ratio
    # fig.add_trace(
    #     go.Scatter(x=signals.index, y=signals['Gold_Nifty_Ratio'], 
    #               name='Gold/NIFTY Ratio', line=dict(color='gold')),
    #     row=4, col=1
    # )
    fig.add_trace(
         go.Scatter(x=gold_nifty_ratio.index, y=gold_nifty_ratio, 
                   name='Gold/NIFTY Ratio', line=dict(color='gold')),
         row=4, col=1
    )

    fig.add_hline(y=1.0, line_dash="solid", line_color="black", row=4, col=1)
    fig.add_hline(y=1.5, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=0.8, line_dash="dash", line_color="green", row=4, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"Technical Analysis - {stock_name}",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Current indicator values
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Current RSI", f"{signals['RSI'].iloc[-1]:.2f}")
    with col2:
        st.metric("Current MFI", f"{signals['MFI'].iloc[-1]:.2f}")
    with col3:
        st.metric("MACD", f"{signals['MACD'].iloc[-1]:.3f}")
    with col4:
        st.metric("Signal", f"{signals['Signal'].iloc[-1]}")
    with col5:
        st.metric("Strength", f"{signals['Signal_Strength'].iloc[-1]:.0f}/100")



# More robust version that handles different scenarios
# def robust_data_alignment(df1, df2, primary_column='Close'):
#     """
#     Robust alignment that handles various data quality issues
#     """
#     # Ensure we're working with copies
#     df1 = df1.copy()
#     df2 = df2.copy()
    
#     # Convert indices to datetime
#     df1.index = pd.to_datetime(df1.index)
#     df2.index = pd.to_datetime(df2.index)
    
#     # Normalize times to remove time component for daily data
#     df1.index = df1.index.normalize()
#     df2.index = df2.index.normalize()
    
#     # Find common dates
#     common_dates = df1.index.intersection(df2.index)
    
#     if len(common_dates) == 0:
#         print("Warning: No common dates found. Trying date range alignment...")
#         # Try to align by date range
#         start_date = max(df1.index.min(), df2.index.min())
#         end_date = min(df1.index.max(), df2.index.max())
        
#         if start_date > end_date:
#             raise ValueError("No overlapping date range between datasets")
            
#         common_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
#     # Align both datasets to common dates
#     aligned_df1 = df1.loc[common_dates]
#     aligned_df2 = df2.loc[common_dates]
    
#     # Drop any remaining NaN values
#     mask = ~(aligned_df1[primary_column].isna() | aligned_df2[primary_column].isna())
#     aligned_df1 = aligned_df1[mask]
#     aligned_df2 = aligned_df2[mask]
    
#     return aligned_df1, aligned_df2


# Fixed version that handles timezone issues
def robust_data_alignment(df1, df2, primary_column='Close'):
    """
    Robust alignment that handles various data quality issues including timezones
    """
    # Ensure we're working with copies
    df1 = df1.copy()
    df2 = df2.copy()
    
    # Convert indices to datetime and remove timezone information
    df1.index = pd.to_datetime(df1.index).tz_localize(None)
    df2.index = pd.to_datetime(df2.index).tz_localize(None)
    
    # Normalize times to remove time component for daily data
    df1.index = df1.index.normalize()
    df2.index = df2.index.normalize()
    
    # Find common dates
    common_dates = df1.index.intersection(df2.index)
    
    if len(common_dates) == 0:
        print("Warning: No common dates found. Trying date range alignment...")
        # Try to align by date range - ensure no timezone issues
        start_date = max(df1.index.min(), df2.index.min())
        end_date = min(df1.index.max(), df2.index.max())
        
        if start_date > end_date:
            raise ValueError("No overlapping date range between datasets")
        
        # Create date range without timezone issues
        common_dates = pd.date_range(
            start=start_date.replace(tzinfo=None), 
            end=end_date.replace(tzinfo=None), 
            freq='D'
        )
    
    # Align both datasets to common dates
    aligned_df1 = df1.reindex(common_dates)
    aligned_df2 = df2.reindex(common_dates)
    
    # Drop any remaining NaN values
    # mask = ~(aligned_df1[primary_column].isna() | aligned_df2[primary_column].isna())
    # aligned_df1 = aligned_df1[mask]
    # aligned_df2 = aligned_df2[mask]
    
    return aligned_df1, aligned_df2



if __name__ == "__main__":
    main()



# # Usage example
# if __name__ == "__main__":
#     # Example with Gold and Nifty
#     gold_data, nifty_data = calculate_aligned_rsi_ratio("GC=F", "^NSEI", period="6mo")
    
#     if gold_data is not None:
#         print(f"\nFirst few aligned dates:")
#         print("Gold dates:", gold_data.index[:5].tolist())
#         print("Nifty dates:", nifty_data.index[:5].tolist())

# # Complete example with RSI ratio calculation
# def calculate_aligned_rsi_ratio(symbol1, symbol2, period="1y", rsi_period=14):
#     """
#     Complete workflow: fetch, align, and calculate RSI ratio
#     """
#     try:
#         # Fetch data
#         print("Fetching data...")
#         data1 = yf.download(symbol1, period=period, interval="1d")
#         data2 = yf.download(symbol2, period=period, interval="1d")
        
#         print(f"\nOriginal data points:")
#         print(f"{symbol1}: {len(data1)}")
#         print(f"{symbol2}: {len(data2)}")
        
#         # Align data
#         aligned_data1, aligned_data2 = robust_data_alignment(data1, data2)
        
#         print(f"\nAfter alignment:")
#         print(f"{symbol1}: {len(aligned_data1)}")
#         print(f"{symbol2}: {len(aligned_data2)}")
        
#         if len(aligned_data1) == 0:
#             raise ValueError("No common data points after alignment")
        
#         # Calculate RSI for both assets (you can add your RSI function here)
#         # rsi1 = calculate_rsi(aligned_data1['Close'], rsi_period)
#         # rsi2 = calculate_rsi(aligned_data2['Close'], rsi_period)
#         # rsi_ratio = rsi1 / rsi2
        
#         print("Data successfully aligned for RSI ratio calculation!")
#         return aligned_data1, aligned_data2
        
#     except Exception as e:
#         print(f"Error in data alignment: {e}")
#         return None, None


