import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os

# make folders if they don't exist
os.makedirs('plots', exist_ok=True)
warnings.filterwarnings('ignore')

# styling stuff
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# load the data
print("loading data...")

fear_greed_df = pd.read_csv('fear_greed_index.csv')
print("fear & greed data:")
print(fear_greed_df.head())
print(f"\nshape: {fear_greed_df.shape}")
print(f"\ncolumns: {fear_greed_df.columns.tolist()}")

trader_df = pd.read_csv('historical_data.csv')
print("\n\ntrader data:")
print(trader_df.head())
print(f"\nshape: {trader_df.shape}")
print(f"\ncolumns: {trader_df.columns.tolist()}")

# clean up the data
def preprocess_data(fear_greed_df, trader_df):
    # figure out the timestamp column
    timestamp_col = None
    for col in ['timestamp', 'Timestamp', 'time', 'date']:
        if col in fear_greed_df.columns:
            timestamp_col = col
            break
    
    if timestamp_col:
        if timestamp_col == 'date':
            fear_greed_df['Date'] = pd.to_datetime(fear_greed_df['date'])
        else:
            # check if it's seconds or milliseconds
            sample_val = fear_greed_df[timestamp_col].iloc[0]
            if sample_val > 1e12:  
                fear_greed_df['Date'] = pd.to_datetime(fear_greed_df[timestamp_col], unit='ms')
            else:  
                fear_greed_df['Date'] = pd.to_datetime(fear_greed_df[timestamp_col], unit='s')
    
    fear_greed_df = fear_greed_df.sort_values('Date')
    
    # find classification column
    classification_col = None
    for col in ['classification', 'Classification', 'class', 'sentiment']:
        if col in fear_greed_df.columns:
            classification_col = col
            break
    
    if classification_col and classification_col != 'Classification':
        fear_greed_df.rename(columns={classification_col: 'Classification'}, inplace=True)
    
    # make classification from value if needed
    if 'value' in fear_greed_df.columns and 'Classification' not in fear_greed_df.columns:
        def classify_sentiment(value):
            if value <= 25:
                return 'Extreme Fear'
            elif value <= 45:
                return 'Fear'
            elif value <= 55:
                return 'Neutral'
            elif value <= 75:
                return 'Greed'
            else:
                return 'Extreme Greed'
        
        fear_greed_df['Classification'] = fear_greed_df['value'].apply(classify_sentiment)
    
    # rename columns to something more standard
    column_mapping = {
        'Account': 'account',
        'Coin': 'symbol',
        'Execution Price': 'execution_price',
        'Size Tokens': 'size',
        'Size USD': 'size_usd',
        'Side': 'side',
        'Timestamp': 'time',
        'Timestamp IST': 'time_ist',
        'Start Position': 'start_position',
        'Direction': 'direction',
        'Closed PnL': 'closedPnL',
        'Fee': 'fee',
        'Trade ID': 'trade_id'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in trader_df.columns:
            trader_df.rename(columns={old_col: new_col}, inplace=True)
    
    # fix timestamps
    if 'time' in trader_df.columns:
        sample_timestamp = trader_df['time'].iloc[0]
        
        if isinstance(sample_timestamp, (int, float, np.integer, np.floating)):
            if sample_timestamp > 1e12:  
                trader_df['time'] = pd.to_datetime(trader_df['time'], unit='ms')
            elif sample_timestamp > 1e9:  
                trader_df['time'] = pd.to_datetime(trader_df['time'], unit='s')
        else:
            trader_df['time'] = pd.to_datetime(trader_df['time'])
    
    # make date column for joining
    trader_df['date'] = trader_df['time'].dt.date
    trader_df['date'] = pd.to_datetime(trader_df['date'])
    
    # calculate some extra stuff
    if 'execution_price' in trader_df.columns and 'size' in trader_df.columns:
        trader_df['trade_value'] = trader_df['execution_price'] * abs(trader_df['size'])
    
    if 'closedPnL' in trader_df.columns:
        trader_df['is_profitable'] = trader_df['closedPnL'] > 0
        trader_df['pnl_category'] = pd.cut(trader_df['closedPnL'], 
                                          bins=[-np.inf, -1000, -100, 0, 100, 1000, np.inf],
                                          labels=['Large Loss', 'Medium Loss', 'Small Loss', 
                                                 'Small Profit', 'Medium Profit', 'Large Profit'])
    
    # add leverage estimate if not there
    if 'leverage' not in trader_df.columns:
        trader_df['leverage'] = 1
    
    return fear_greed_df, trader_df

print("\ncleaning data...")
fear_greed_df, trader_df = preprocess_data(fear_greed_df, trader_df)

print("\n=== cleaned fear & greed data ===")
print(fear_greed_df[['Date', 'value', 'Classification']].head())
print(f"\nunique classifications: {fear_greed_df['Classification'].unique()}")
print(f"date range: {fear_greed_df['Date'].min()} to {fear_greed_df['Date'].max()}")

print("\n=== cleaned trader data ===")
cols_to_show = [col for col in ['account', 'symbol', 'execution_price', 'size', 'side', 'time', 'closedPnL'] 
                if col in trader_df.columns]
print(trader_df[cols_to_show].head())
print(f"\navailable columns: {trader_df.columns.tolist()}")
print(f"date range: {trader_df['date'].min()} to {trader_df['date'].max()}")

# some basic visualizations
def perform_eda(fear_greed_df, trader_df):
    print("\n=== exploratory data analysis ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # sentiment distribution pie chart
    sentiment_counts = fear_greed_df['Classification'].value_counts()
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                   autopct='%1.1f%%', colors=colors)
    axes[0, 0].set_title('Fear & Greed Index Distribution')
    
    # pnl histogram
    if 'closedPnL' in trader_df.columns:
        pnl_data = trader_df['closedPnL']
        lower_bound = pnl_data.quantile(0.01)
        upper_bound = pnl_data.quantile(0.99)
        filtered_pnl = pnl_data[(pnl_data >= lower_bound) & (pnl_data <= upper_bound)]
        
        axes[0, 1].hist(filtered_pnl, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Closed PnL')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Trader PnL (99% data)')
        axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
        axes[0, 1].legend()
    
    # buy vs sell
    if 'side' in trader_df.columns:
        side_counts = trader_df['side'].value_counts()
        axes[0, 2].bar(side_counts.index, side_counts.values, color=['green', 'red'])
        axes[0, 2].set_title('Trade Count by Side')
        axes[0, 2].set_ylabel('Number of Trades')
        
        for i, v in enumerate(side_counts.values):
            axes[0, 2].text(i, v + max(side_counts.values)*0.01, 
                           f'{v:,}\n({v/len(trader_df)*100:.1f}%)', 
                           ha='center')
    
    # volume over time
    if 'trade_value' in trader_df.columns:
        daily_volume = trader_df.groupby('date')['trade_value'].sum().rolling(7).mean()
        axes[1, 0].plot(daily_volume.index, daily_volume.values, linewidth=2)
        axes[1, 0].set_title('7-Day Moving Average Trading Volume')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Volume (USD)')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # top traders
    if 'account' in trader_df.columns and 'trade_value' in trader_df.columns:
        top_traders = trader_df.groupby('account')['trade_value'].sum().nlargest(10)
        axes[1, 1].barh(range(len(top_traders)), top_traders.values)
        axes[1, 1].set_yticks(range(len(top_traders)))
        axes[1, 1].set_yticklabels([f"{str(addr)[:8]}..." for addr in top_traders.index])
        axes[1, 1].set_xlabel('Total Trade Volume (USD)')
        axes[1, 1].set_title('Top 10 Traders by Volume')
    
    # profitability by size
    if 'size' in trader_df.columns and 'is_profitable' in trader_df.columns:
        trader_df['size_category'] = pd.qcut(abs(trader_df['size']), q=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
        profitability_by_size = trader_df.groupby('size_category')['is_profitable'].mean() * 100
        
        axes[1, 2].bar(profitability_by_size.index, profitability_by_size.values)
        axes[1, 2].set_title('Profitability Rate by Trade Size Category')
        axes[1, 2].set_ylabel('Profitability Rate (%)')
        axes[1, 2].set_xlabel('Trade Size Category')
    
    plt.tight_layout()
    plt.savefig('plots/eda_overview.png', dpi=300, bbox_inches='tight')
    print("saved eda overview to plots/eda_overview.png")
    plt.close()
    
    # print some stats
    print("\n=== trader stats ===")
    print(f"total trades: {len(trader_df):,}")
    
    if 'account' in trader_df.columns:
        print(f"unique traders: {trader_df['account'].nunique():,}")
    
    if 'symbol' in trader_df.columns:
        print(f"unique symbols: {trader_df['symbol'].nunique():,}")
        print(f"top 5 symbols:")
        print(trader_df['symbol'].value_counts().head())
    
    if 'date' in trader_df.columns:
        print(f"time period: {trader_df['date'].min().date()} to {trader_df['date'].max().date()}")
    
    if 'closedPnL' in trader_df.columns:
        print(f"total pnl: ${trader_df['closedPnL'].sum():,.2f}")
        print(f"avg pnl per trade: ${trader_df['closedPnL'].mean():,.2f}")
        print(f"median pnl: ${trader_df['closedPnL'].median():,.2f}")
        
        if 'is_profitable' in trader_df.columns:
            win_rate = trader_df['is_profitable'].mean()
            print(f"win rate: {win_rate*100:.2f}%")
            
            winning_trades = trader_df[trader_df['is_profitable']]
            losing_trades = trader_df[~trader_df['is_profitable']]
            
            if len(winning_trades) > 0:
                avg_win = winning_trades['closedPnL'].mean()
                print(f"avg win: ${avg_win:,.2f}")
            
            if len(losing_trades) > 0:
                avg_loss = losing_trades['closedPnL'].mean()
                print(f"avg loss: ${avg_loss:,.2f}")
            
            if len(winning_trades) > 0 and len(losing_trades) > 0 and losing_trades['closedPnL'].sum() != 0:
                profit_factor = abs(winning_trades['closedPnL'].sum() / losing_trades['closedPnL'].sum())
                print(f"profit factor: {profit_factor:.2f}")

print("\nrunning eda...")
perform_eda(fear_greed_df, trader_df)

# merge and analyze
def analyze_sentiment_impact(fear_greed_df, trader_df):
    print("\n=== sentiment impact analysis ===")
    
    fear_greed_df['date_only'] = fear_greed_df['Date'].dt.date
    trader_df['date_only'] = trader_df['date'].dt.date
    
    merged_df = pd.merge(trader_df, fear_greed_df[['date_only', 'value', 'Classification']], 
                         on='date_only', how='left')
    
    print(f"rows before merge: {len(trader_df)}")
    print(f"rows after merge: {len(merged_df)}")
    print(f"rows with sentiment: {merged_df['Classification'].notna().sum()}")
    print(f"rows without sentiment: {merged_df['Classification'].isna().sum()}")
    
    if merged_df['Classification'].isna().all():
        print("warning: no matching sentiment data found")
        return merged_df, None
    
    sentiment_performance = merged_df.groupby('Classification').agg({
        'closedPnL': ['mean', 'median', 'sum', 'std', 'count'],
        'is_profitable': 'mean',
        'size': 'mean',
        'trade_value': 'mean'
    }).round(2)
    
    sentiment_performance.columns = ['_'.join(col).strip() for col in sentiment_performance.columns.values]
    
    print("\n=== performance by sentiment ===")
    print(sentiment_performance)
    
    # make some plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    
    # avg pnl by sentiment
    avg_pnl = merged_df.groupby('Classification')['closedPnL'].mean()
    avg_pnl = avg_pnl.reindex(sentiment_order)
    axes[0, 0].bar(avg_pnl.index, avg_pnl.values, color=['darkred', 'red', 'gray', 'lightgreen', 'green'])
    axes[0, 0].set_title('Average PnL by Market Sentiment')
    axes[0, 0].set_ylabel('Average PnL (USD)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # win rate by sentiment
    profitability = merged_df.groupby('Classification')['is_profitable'].mean() * 100
    profitability = profitability.reindex(sentiment_order)
    axes[0, 1].bar(profitability.index, profitability.values, color=['darkred', 'red', 'gray', 'lightgreen', 'green'])
    axes[0, 1].set_title('Profitability Rate by Market Sentiment')
    axes[0, 1].set_ylabel('Profitability Rate (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% baseline')
    
    # trade count by sentiment
    trade_count = merged_df.groupby('Classification').size()
    trade_count = trade_count.reindex(sentiment_order)
    axes[1, 0].bar(trade_count.index, trade_count.values, color=['darkred', 'red', 'gray', 'lightgreen', 'green'])
    axes[1, 0].set_title('Number of Trades by Market Sentiment')
    axes[1, 0].set_ylabel('Number of Trades')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # avg size by sentiment
    trade_size = merged_df.groupby('Classification')['size'].mean()
    trade_size = trade_size.reindex(sentiment_order)
    axes[1, 1].bar(trade_size.index, abs(trade_size.values), color=['darkred', 'red', 'gray', 'lightgreen', 'green'])
    axes[1, 1].set_title('Average Trade Size by Market Sentiment')
    axes[1, 1].set_ylabel('Average Size (Tokens)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('plots/sentiment_analysis.png', dpi=300, bbox_inches='tight')
    print("saved sentiment analysis to plots/sentiment_analysis.png")
    plt.close()
    
    return merged_df, sentiment_performance

print("\nanalyzing sentiment impact...")
merged_df, sentiment_performance = analyze_sentiment_impact(fear_greed_df, trader_df)

# deeper analysis
def advanced_pattern_analysis(merged_df):
    if merged_df is None or 'Classification' not in merged_df.columns:
        print("no merged data for advanced analysis")
        return
    
    print("\n=== advanced patterns ===")
    
    # time patterns
    merged_df['hour'] = merged_df['time'].dt.hour
    merged_df['day_of_week'] = merged_df['time'].dt.day_name()
    merged_df['month'] = merged_df['time'].dt.month_name()
    
    # hourly profitability
    plt.figure(figsize=(12, 6))
    profitability_by_hour = merged_df.groupby('hour')['is_profitable'].mean() * 100
    plt.bar(profitability_by_hour.index, profitability_by_hour.values)
    plt.xlabel('Hour of Day (UTC)')
    plt.ylabel('Profitability Rate (%)')
    plt.title('Profitability by Hour of Day')
    plt.xticks(range(0, 24))
    plt.axhline(y=profitability_by_hour.mean(), color='r', linestyle='--', 
                label=f'avg: {profitability_by_hour.mean():.1f}%')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('plots/profitability_by_hour.png', dpi=300, bbox_inches='tight')
    print("saved hourly profitability to plots/profitability_by_hour.png")
    plt.close()
    
    # day of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    merged_df['day_of_week'] = pd.Categorical(merged_df['day_of_week'], categories=day_order, ordered=True)
    
    plt.figure(figsize=(10, 6))
    profitability_by_day = merged_df.groupby('day_of_week')['is_profitable'].mean() * 100
    plt.bar(profitability_by_day.index, profitability_by_day.values)
    plt.xlabel('Day of Week')
    plt.ylabel('Profitability Rate (%)')
    plt.title('Profitability by Day of Week')
    plt.xticks(rotation=45)
    plt.axhline(y=profitability_by_day.mean(), color='r', linestyle='--',
                label=f'avg: {profitability_by_day.mean():.1f}%')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('plots/profitability_by_day.png', dpi=300, bbox_inches='tight')
    print("saved daily profitability to plots/profitability_by_day.png")
    plt.close()
    
    # sentiment changes
    if 'Classification' in merged_df.columns:
        sentiment_changes = fear_greed_df.copy()
        sentiment_changes = sentiment_changes.sort_values('Date')
        sentiment_changes['prev_sentiment'] = sentiment_changes['Classification'].shift(1)
        sentiment_changes['sentiment_change'] = sentiment_changes['Classification'] != sentiment_changes['prev_sentiment']
        
        sentiment_changes['date_only'] = sentiment_changes['Date'].dt.date
        merged_with_changes = pd.merge(merged_df, sentiment_changes[['date_only', 'sentiment_change']], 
                                      on='date_only', how='left')
        
        if 'sentiment_change' in merged_with_changes.columns:
            change_performance = merged_with_changes.groupby('sentiment_change').agg({
                'is_profitable': 'mean',
                'closedPnL': 'mean'
            })
            
            print("\n=== performance during sentiment changes ===")
            print(change_performance)
    
    # correlations
    numeric_cols = ['closedPnL', 'size', 'execution_price', 'fee']
    numeric_cols = [col for col in numeric_cols if col in merged_df.columns]
    
    if len(numeric_cols) > 1:
        correlation_matrix = merged_df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
        plt.title('Correlation Matrix of Trading Metrics')
        plt.tight_layout()
        plt.savefig('plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("saved correlation matrix to plots/correlation_matrix.png")
        plt.close()
    
    # top performers
    print("\n=== top traders ===")
    
    if 'account' in merged_df.columns:
        trader_performance = merged_df.groupby('account').agg({
            'closedPnL': ['sum', 'mean', 'count'],
            'is_profitable': 'mean',
            'trade_value': 'sum'
        }).round(2)
        
        trader_performance.columns = ['total_pnl', 'avg_pnl', 'trade_count', 'win_rate', 'total_volume']
        
        active_traders = trader_performance[trader_performance['trade_count'] >= 10]
        
        print(f"total traders: {len(trader_performance)}")
        print(f"active traders (10+ trades): {len(active_traders)}")
        
        top_traders = active_traders.nlargest(10, 'total_pnl')
        print("\ntop 10 by pnl:")
        print(top_traders[['total_pnl', 'win_rate', 'trade_count', 'total_volume']])
        
        bottom_traders = active_traders.nsmallest(10, 'total_pnl')
        print("\nbottom 10 by pnl:")
        print(bottom_traders[['total_pnl', 'win_rate', 'trade_count', 'total_volume']])

print("\nrunning advanced analysis...")
advanced_pattern_analysis(merged_df)

# wrap up with insights
def generate_insights(fear_greed_df, trader_df, merged_df, sentiment_performance):
    print("\n" + "="*80)
    print("insights & recommendations")
    print("="*80)
    
    print("\ndata overview:")
    print(f"fear & greed period: {fear_greed_df['Date'].min().date()} to {fear_greed_df['Date'].max().date()}")
    print(f"trading period: {trader_df['date'].min().date()} to {trader_df['date'].max().date()}")
    print(f"total trades: {len(trader_df):,}")
    print(f"unique traders: {trader_df['account'].nunique():,}")
    
    if 'closedPnL' in trader_df.columns:
        total_pnl = trader_df['closedPnL'].sum()
        avg_pnl = trader_df['closedPnL'].mean()
        win_rate = trader_df['is_profitable'].mean() * 100
        print(f"total pnl: ${total_pnl:,.2f}")
        print(f"avg pnl per trade: ${avg_pnl:,.2f}")
        print(f"overall win rate: {win_rate:.1f}%")
    
    print("\nkey findings:")
    
    if sentiment_performance is not None:
        if 'closedPnL_mean' in sentiment_performance.columns:
            best_idx = sentiment_performance['closedPnL_mean'].idxmax()
            worst_idx = sentiment_performance['closedPnL_mean'].idxmin()
            best_pnl = sentiment_performance.loc[best_idx, 'closedPnL_mean']
            worst_pnl = sentiment_performance.loc[worst_idx, 'closedPnL_mean']
            
            print(f"best performing regime: {best_idx}")
            print(f"  avg pnl: ${best_pnl:.2f}")
            print(f"  win rate: {sentiment_performance.loc[best_idx, 'is_profitable_mean']*100:.1f}%")
            print(f"\nworst performing regime: {worst_idx}")
            print(f"  avg pnl: ${worst_pnl:.2f}")
            print(f"  win rate: {sentiment_performance.loc[worst_idx, 'is_profitable_mean']*100:.1f}%")
        
        if 'closedPnL_count' in sentiment_performance.columns:
            highest_volume_idx = sentiment_performance['closedPnL_count'].idxmax()
            volume = sentiment_performance.loc[highest_volume_idx, 'closedPnL_count']
            print(f"\nmost active during: {highest_volume_idx}")
            print(f"  trades: {volume:,.0f}")
            print(f"  percentage: {volume/len(trader_df)*100:.1f}%")
    
    print("\ntrading patterns:")
    
    if 'hour' in merged_df.columns and 'is_profitable' in merged_df.columns:
        profitability_by_hour = merged_df.groupby('hour')['is_profitable'].mean() * 100
        best_hour = profitability_by_hour.idxmax()
        worst_hour = profitability_by_hour.idxmin()
        print(f"best hour: {best_hour}:00 UTC ({profitability_by_hour.max():.1f}% win rate)")
        print(f"worst hour: {worst_hour}:00 UTC ({profitability_by_hour.min():.1f}% win rate)")
    
    if 'account' in trader_df.columns:
        trader_volumes = trader_df.groupby('account')['trade_value'].sum().sort_values(ascending=False)
        top_10_pct = trader_volumes.head(int(len(trader_volumes) * 0.1)).sum() / trader_volumes.sum()
        print(f"\ntop 10% of traders control {top_10_pct*100:.1f}% of volume")
    
    print("\nrecommendations:")
    print("1. use sentiment as a contrarian indicator")
    print("   - buy during extreme fear")
    print("   - sell during extreme greed")
    print("\n2. optimize timing")
    print("   - trade during high-probability hours")
    print("   - avoid low-performance times")
    print("\n3. risk management")
    print("   - adjust position sizes based on sentiment")
    print("   - use sentiment-based stop losses")
    print("   - track performance by regime")
    print("\n4. monitor key metrics")
    print("   - daily sentiment levels")
    print("   - regime changes")
    print("   - volume patterns")

print("\ngenerating insights...")
generate_insights(fear_greed_df, trader_df, merged_df, sentiment_performance)

# save everything
os.makedirs('output', exist_ok=True)

merged_df.to_csv('output/merged_trading_sentiment_data.csv', index=False)
print("\nsaved merged data to output/merged_trading_sentiment_data.csv")

if sentiment_performance is not None:
    sentiment_performance.to_csv('output/sentiment_performance_summary.csv')
    print("saved sentiment performance to output/sentiment_performance_summary.csv")

stats = {
    'total_trades': len(trader_df),
    'unique_traders': trader_df['account'].nunique(),
    'total_pnl': trader_df['closedPnL'].sum() if 'closedPnL' in trader_df.columns else None,
    'win_rate': trader_df['is_profitable'].mean() if 'is_profitable' in trader_df.columns else None,
    'analysis_period_start': trader_df['date'].min(),
    'analysis_period_end': trader_df['date'].max()
}

stats_df = pd.DataFrame([stats])
stats_df.to_csv('output/analysis_statistics.csv', index=False)
print("saved stats to output/analysis_statistics.csv")

print("\n" + "="*80)
print("analysis complete")
print("="*80)
print("\ncheck the plots/ folder for visualizations")
print("check the output/ folder for data exports")