import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import os
import matplotlib.pyplot as plt
from util.algo_dataset import get_algo_dataset

def read_rl_results(portfolio_num):
    """Read RL results for the given portfolio."""
    rl_path = f'data/rl/portfolio{portfolio_num+1}/lagged/daily_nav_comp_gradual_non_predicted.csv'
    if os.path.exists(rl_path):
        try:
            rl_df = pd.read_csv(rl_path, parse_dates=['Date'])
            return rl_df
        except Exception as e:
            print(f"Error reading RL results for portfolio {portfolio_num+1}: {str(e)}")
    else:
        print(f"RL results file not found: {rl_path}")
    return None

def calculate_nav_metrics(df):
    """Calculate Total Return and Max Drawdown for a given DataFrame."""
    if 'Net' not in df.columns:
        if 'Close' in df.columns:
            df['Net'] = df['Close']
        elif 'NAV' in df.columns:
            df['Net'] = df['NAV']
        else:
            raise ValueError("DataFrame does not contain 'Net', 'Close', or 'NAV' column")

    df['Daily Return'] = df['Net'].pct_change()
    total_return = (df['Net'].iloc[-1] / df['Net'].iloc[0] - 1) * 100
    
    df['Cumulative Return'] = (1 + df['Daily Return']).cumprod()
    df['Drawdown'] = (df['Cumulative Return'].cummax() - df['Cumulative Return']) / df['Cumulative Return'].cummax()
    max_drawdown = df['Drawdown'].max() * 100
    
    return {
        'Total Return (%)': total_return,
        'Max Drawdown (%)': max_drawdown
    }

def compare_portfolio_assets(portfolio_num):
    """Compare performance metrics for each asset in the specified portfolio, including RL results."""
    try:
        df_list, date_range, trend_list, stocks = get_algo_dataset(portfolio_num)
    except Exception as e:
        print(f"Error getting dataset for portfolio {portfolio_num+1}: {str(e)}")
        return

    results = {}
    
    # Calculate metrics for each asset
    for i, stock in enumerate(stocks):
        df = df_list[i]
        df = df[df['Date'].isin(date_range)]
        df = df.sort_values('Date')
        df = df.reset_index(drop=True)
        
        try:
            metrics = calculate_nav_metrics(df)
            results[stock] = metrics
        except Exception as e:
            print(f"Error calculating metrics for {stock}: {str(e)}")
    
    # Calculate metrics for the combined portfolio
    portfolio_path = f'data/rl/portfolio{portfolio_num+1}/non_lagged/daily_nav.csv'
    if os.path.exists(portfolio_path):
        try:
            portfolio_df = pd.read_csv(portfolio_path, parse_dates=['Date'])
            portfolio_metrics = calculate_nav_metrics(portfolio_df)
            results['RL'] = portfolio_metrics
        except Exception as e:
            print(f"Error reading or processing portfolio data: {str(e)}")
    else:
        print(f"Portfolio data file not found: {portfolio_path}")
    
    # Calculate metrics for RL results
    rl_df = read_rl_results(portfolio_num)
    if rl_df is not None:
        try:
            rl_metrics = calculate_nav_metrics(rl_df)
            results['RL Strategy'] = rl_metrics
        except Exception as e:
            print(f"Error calculating metrics for RL results: {str(e)}")
    
    # Print results
    print(f"\nPerformance Comparison for Portfolio {portfolio_num+1}")
    print("-" * 60)
    print(f"{'Asset/Strategy':<15} {'Total Return (%)':<20} {'Max Drawdown (%)':<20}")
    print("-" * 60)
    
    for asset, metrics in results.items():
        print(f"{asset:<15} {metrics['Total Return (%)']:>18.2f} {metrics['Max Drawdown (%)']:>19.2f}")
    
    # Plotting
    if results:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Total Return comparison
        returns = [metrics['Total Return (%)'] for metrics in results.values()]
        ax1.bar(results.keys(), returns)
        ax1.set_title(f'Total Return Comparison - Portfolio {portfolio_num+1}')
        ax1.set_ylabel('Total Return (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Max Drawdown comparison
        drawdowns = [metrics['Max Drawdown (%)'] for metrics in results.values()]
        ax2.bar(results.keys(), drawdowns)
        ax2.set_title(f'Max Drawdown Comparison - Portfolio {portfolio_num+1}')
        ax2.set_ylabel('Max Drawdown (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    else:
        print("No data to plot.")

# Run comparison for all portfolios
for i in range(3):
    compare_portfolio_assets(i)