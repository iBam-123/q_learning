import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import os
import matplotlib.pyplot as plt
from util.algo_dataset import get_algo_dataset

def read_nav_file(path):
    """Read NAV file safely"""
    try:
        df = pd.read_csv(path, parse_dates=['Date'])
        return df
    except Exception as e:
        print(f"Error reading file {path}: {str(e)}")
        return None

def calculate_nav_metrics(df):
    """Calculate Total Return, Max Drawdown, Sharpe Ratio, and Return-Drawdown Correlation."""
    if df is None:
        return None
    
    # Ensure 'Net' column exists
    if 'Net' not in df.columns:
        df['Net'] = df['Close'] if 'Close' in df.columns else df['NAV']

    #Daily Return and Total Return
    df['Daily Return'] = df['Net'].pct_change()
    total_return = (df['Net'].iloc[-1] / df['Net'].iloc[0] - 1) * 100
    
    #Max Drawdown
    df['Cumulative Return'] = (1 + df['Daily Return']).cumprod()
    df['Drawdown'] = (df['Cumulative Return'].cummax() - df['Cumulative Return']) / df['Cumulative Return'].cummax()
    max_drawdown = df['Drawdown'].max() * 100
    
    #Sharpe Ratio
    daily_rf = 0.02/252  # Risk-free rate
    excess_returns = df['Daily Return'] - daily_rf
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    #Correlation Return-Drawdown
    correlation = np.corrcoef(df['Daily Return'].dropna(), df['Drawdown'].iloc[1:])[0,1]
    
    return {
        'Total Return (%)': total_return,
        'Max Drawdown (%)': max_drawdown,
        'Sharpe Ratio': sharpe_ratio,
        'Return-DD Correlation': correlation
    }

def print_performance_table(results, title):
    """Print performance table with formatted output"""
    print(f"\n{title}")
    print("-" * 80)
    print(f"{'Asset/Strategy':<20} {'Total Return (%)':<20} {'Max Drawdown (%)':<20} {'Sharpe Ratio':<20} {'Correlation Return-Drwadown':<20}")
    print("-" * 80)
    
    for asset, metrics in results.items():
        if metrics is not None:
            print(f"{asset:<20} {metrics['Total Return (%)']:>18.2f} {metrics['Max Drawdown (%)']:>18.2f} "
                  f"{metrics['Sharpe Ratio']:>18.2f} {metrics['Return-DD Correlation']:>18.2f}")

def analyze_portfolio_performance(portfolio_num):
    """Comprehensive portfolio performance analysis"""
    # Define paths for different approaches and prediction modes
    approaches = ['gradual', 'full_swing']
    prediction_modes = [False, True]
    
    for approach in approaches:
        for predict in prediction_modes:
            # Determine correct subfolder
            subfolder = 'non_lagged' if predict else 'lagged'
            if approach == 'full_swing':
                subfolder = f'fs_{subfolder}'
            
            base_path = f'data/rl/portfolio{portfolio_num+1}/{subfolder}'
            

            results = {}
            
            # Get dataset for individual assets
            df_list, date_range, trend_list, stocks = get_algo_dataset(portfolio_num)
            
            # Analyze individual assets
            for i, stock in enumerate(stocks):
                df = df_list[i]
                df = df[df['Date'].isin(date_range)]
                df = df.sort_values('Date')
                results[stock] = calculate_nav_metrics(df)
            
            # portfolio NAV
#            portfolio_nav_path = os.path.join(base_path, 'daily_nav.csv')
#            portfolio_nav = read_nav_file(portfolio_nav_path)
#            results['Portfolio'] = calculate_nav_metrics(portfolio_nav)
            
            # RL strategy NAV
            rl_nav_path = os.path.join(base_path, 'daily_nav.csv')
            rl_nav = read_nav_file(rl_nav_path)
            results['RL Strategy'] = calculate_nav_metrics(rl_nav)
            
            title = f"Portfolio {portfolio_num+1} - {approach.replace('_', ' ').title()} Approach, " \
                    f"{'With' if predict else 'Without'} Prediction"
            
            print_performance_table(results, title)

for i in range(3):
    analyze_portfolio_performance(i)