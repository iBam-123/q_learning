import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import os
import matplotlib.pyplot as plt
from util.algo_dataset import get_algo_dataset

def read_rl_results(portfolio_num):
    """Read RL results for the given portfolio."""
    rl_path = f'data/rl/portfolio{portfolio_num+1}/lagged/daily_nav.csv'
    if os.path.exists(rl_path):
        try:
            rl_df = pd.read_csv(rl_path, parse_dates=['Date'])
            return rl_df
        except Exception as e:
            print(f"Error reading RL results for portfolio {portfolio_num+1}: {str(e)}")
    else:
        print(f"RL results file not found: {rl_path}")
    return None

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate annualized Sharpe Ratio
    """
    daily_rf = risk_free_rate/252
    excess_returns = returns - daily_rf
    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

def calculate_return_drawdown_correlation(returns, drawdowns):
    """
    Calculate correlation between returns and drawdowns
    """
    return np.corrcoef(returns, drawdowns)[0,1]


def calculate_nav_metrics(df):
    """Calculate Total Return, Max Drawdown, Sharpe Ratio, and Return-Drawdown Correlation."""
    if 'Net' not in df.columns:
        if 'Close' in df.columns:
            df['Net'] = df['Close']
        elif 'NAV' in df.columns:
            df['Net'] = df['NAV']
        else:
            raise ValueError("DataFrame does not contain 'Net', 'Close', or 'NAV' column")

    # Calculate Daily Return and Total Return
    df['Daily Return'] = df['Net'].pct_change()
    total_return = (df['Net'].iloc[-1] / df['Net'].iloc[0] - 1) * 100
    
    # Calculate Max Drawdown
    df['Cumulative Return'] = (1 + df['Daily Return']).cumprod()
    df['Drawdown'] = (df['Cumulative Return'].cummax() - df['Cumulative Return']) / df['Cumulative Return'].cummax()
    max_drawdown = df['Drawdown'].max() * 100
    
    # Calculate Sharpe Ratio
    daily_returns = df['Daily Return'].dropna()
    sharpe_ratio = calculate_sharpe_ratio(daily_returns)
    
    # Calculate Return-Drawdown Correlation
    correlation = calculate_return_drawdown_correlation(
        df['Daily Return'].dropna(),
        df['Drawdown'].iloc[1:].values
    )
    
    return {
        'Total Return (%)': total_return,
        'Max Drawdown (%)': max_drawdown,
        'Sharpe Ratio': sharpe_ratio,
        'Return-DD Correlation': correlation
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
    print("-" * 80)
    print(f"{'Asset/Strategy':<15} {'Total Return (%)':<15} {'Max Drawdown (%)':<15} {'Sharpe Ratio':<15} {'Return-DD Corr':<15}")
    print("-" * 80)
    
    for asset, metrics in results.items():
        print(f"{asset:<15} {metrics['Total Return (%)']:>13.2f} {metrics['Max Drawdown (%)']:>14.2f} "
              f"{metrics['Sharpe Ratio']:>14.2f} {metrics['Return-DD Correlation']:>14.2f}")
    
    # Update plotting
    if results:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
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
        
        # Sharpe Ratio comparison
        sharpe_ratios = [metrics['Sharpe Ratio'] for metrics in results.values()]
        ax3.bar(results.keys(), sharpe_ratios)
        ax3.set_title('Sharpe Ratio Comparison')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.tick_params(axis='x', rotation=45)
        
        # Return-DD Correlation comparison
        correlations = [metrics['Return-DD Correlation'] for metrics in results.values()]
        ax4.bar(results.keys(), correlations)
        ax4.set_title('Return-Drawdown Correlation')
        ax4.set_ylabel('Correlation')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    else:
        print("No data to plot.")

# Run comparison for all portfolios
for i in range(3):
    compare_portfolio_assets(i)