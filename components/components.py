# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 21:34:53 2025

@author: DELL

Deepseek

components.py


Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max Default: 1mo Either Use period parameter or use start and end

Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo Intraday data cannot extend last 60 days


"""

import streamlit as st
import pandas as pd

def portfolio_allocator():
    """Interactive portfolio allocation component"""
    st.sidebar.subheader("💰 Portfolio Allocator")
    
    initial_capital = st.sidebar.number_input(
        "Initial Capital (₹):",
        min_value=10000,
        max_value=10000000,
        value=100000,
        step=5000
    )
    
    max_position_size = st.sidebar.slider(
        "Max Position Size (%):",
        min_value=5,
        max_value=50,
        value=20
    )
    
    stop_loss = st.sidebar.slider(
        "Stop Loss (%):",
        min_value=1,
        max_value=20,
        value=7
    )
    
    return {
        'initial_capital': initial_capital,
        'max_position_size': max_position_size,
        'stop_loss': stop_loss
    }

def alert_system():
    """Interactive alert system component"""
    st.sidebar.subheader("🔔 Alert System")
    
    enable_alerts = st.sidebar.checkbox("Enable Price Alerts", value=False)
    
    if enable_alerts:
        alert_type = st.sidebar.selectbox(
            "Alert Type:",
            ["RSI Crossover", "MACD Crossover", "Price Target", "Volume Spike", "MFI Crossover"]
        )
        
        if alert_type == "RSI Crossover":
            rsi_level = st.sidebar.slider("RSI Alert Level:", 20, 80, 70)
            st.sidebar.write(f"Alert when RSI crosses {rsi_level}")
        
        return True
    return False

def performance_tracker():
    """Interactive performance tracking component"""
    st.sidebar.subheader("📊 Performance Tracker")
    
    show_performance = st.sidebar.checkbox("Show Portfolio Performance", value=True)
    
    if show_performance:
        period = st.sidebar.selectbox(
            "Performance Period:",
            
            ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd"], index=2
           
        )
        # ["1W", "1M", "3M", "6M", "YTD", "1Y"]
        
        benchmark = st.sidebar.selectbox(
            "Benchmark:",
            ["NIFTY 50", "SENSEX", "None"]
        )
        
        return {
            'period': period,
            'benchmark': benchmark
        }
    return None