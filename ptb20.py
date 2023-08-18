import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Loading

aapl_data = pd.read_csv('AAPL_data.csv')[:500]
spy_data = pd.read_csv('XLK_data.csv')[:500]

aapl_data_untrained = pd.read_csv('AAPL_data.csv')[501:800]
spy_data_untrained = pd.read_csv('SPY_data.csv')[501:800]

aapl_data_untrained.reset_index()
spy_data_untrained.reset_index()

aapl_data['p_change'] = aapl_data['Close'].pct_change() * 100
spy_data['p_change'] = spy_data['Close'].pct_change() * 100

aapl_data_untrained['p_change'] = aapl_data_untrained['Close'].pct_change() * 100
spy_data_untrained['p_change'] = spy_data_untrained['Close'].pct_change() * 100

ma_window = 20

# Z-Score and Signal Logic (as in original script)
def z_score_signals(upper_thresh=1, lower_thresh=-1, data=None):
    data['z_scores'] = (data['spread'] - data['ma']) / data['std']
    data['above_z'] = data['z_scores'] > upper_thresh
    data['below_z'] = data['z_scores'] < lower_thresh

# ATR Calculation (as in original script)
def compute_atr(data, window=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr1 = abs(high - low)
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr


def plot_trading_signals(data_aapl, data_spy, pairs_data):
    fig, axs = plt.subplots(2, figsize=(14, 10), sharex=True)

    # For AAPL data
    axs[0].plot(data_aapl['Close'], label='AAPL Close Price', color='black')
    long_aapl_entries = pairs_data[pairs_data['aapl_position'] == 1].index
    short_aapl_entries = pairs_data[pairs_data['aapl_position'] == -1].index
    long_aapl_exits = pairs_data[(pairs_data['aapl_position'] == 1) & (pairs_data['aapl_daily_pl'] != 0)].index
    short_aapl_exits = pairs_data[(pairs_data['aapl_position'] == -1) & (pairs_data['aapl_daily_pl'] != 0)].index

    entry_counter = 0
    for entry, exit in zip(long_aapl_entries, long_aapl_exits):
        axs[0].scatter(entry, data_aapl.loc[entry, 'Close'], marker='^', color='g', alpha=1)
        axs[0].scatter(exit, data_aapl.loc[exit, 'Close'], marker='x', color='g', alpha=1)
        entry_counter += 1
        axs[0].text(entry, data_aapl.loc[entry, 'Close'], str(entry_counter), fontsize=9, verticalalignment='bottom',
                    horizontalalignment='right')
        axs[0].text(exit, data_aapl.loc[exit, 'Close'], str(entry_counter), fontsize=9, verticalalignment='bottom',
                    horizontalalignment='right')

    for entry, exit in zip(short_aapl_entries, short_aapl_exits):
        axs[0].scatter(entry, data_aapl.loc[entry, 'Close'], marker='v', color='r', alpha=1)
        axs[0].scatter(exit, data_aapl.loc[exit, 'Close'], marker='x', color='r', alpha=1)
        entry_counter += 1
        axs[0].text(entry, data_aapl.loc[entry, 'Close'], str(entry_counter), fontsize=9, verticalalignment='top',
                    horizontalalignment='right')
        axs[0].text(exit, data_aapl.loc[exit, 'Close'], str(entry_counter), fontsize=9, verticalalignment='top',
                    horizontalalignment='right')

    axs[0].set_title('AAPL Trading Signals')
    axs[0].set_ylabel('Price')

    # For SPY data
    axs[1].plot(data_spy['Close'], label='SPY Close Price', color='black')
    long_spy_entries = pairs_data[pairs_data['spy_position'] == 1].index
    short_spy_entries = pairs_data[pairs_data['spy_position'] == -1].index
    long_spy_exits = pairs_data[(pairs_data['spy_position'] == 1) & (pairs_data['spy_daily_pl'] != 0)].index
    short_spy_exits = pairs_data[(pairs_data['spy_position'] == -1) & (pairs_data['spy_daily_pl'] != 0)].index

    entry_counter = 0
    for entry, exit in zip(long_spy_entries, long_spy_exits):
        axs[1].scatter(entry, data_spy.loc[entry, 'Close'], marker='^', color='g', alpha=1)
        axs[1].scatter(exit, data_spy.loc[exit, 'Close'], marker='x', color='g', alpha=1)
        entry_counter += 1
        axs[1].text(entry, data_spy.loc[entry, 'Close'], str(entry_counter), fontsize=9, verticalalignment='bottom',
                    horizontalalignment='right')
        axs[1].text(exit, data_spy.loc[exit, 'Close'], str(entry_counter), fontsize=9, verticalalignment='bottom',
                    horizontalalignment='right')

    for entry, exit in zip(short_spy_entries, short_spy_exits):
        axs[1].scatter(entry, data_spy.loc[entry, 'Close'], marker='v', color='r', alpha=1)
        axs[1].scatter(exit, data_spy.loc[exit, 'Close'], marker='x', color='r', alpha=1)
        entry_counter += 1
        axs[1].text(entry, data_spy.loc[entry, 'Close'], str(entry_counter), fontsize=9, verticalalignment='top',
                    horizontalalignment='right')
        axs[1].text(exit, data_spy.loc[exit, 'Close'], str(entry_counter), fontsize=9, verticalalignment='top',
                    horizontalalignment='right')

    axs[1].set_title('SPY Trading Signals')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Price')

    plt.tight_layout()
    plt.show()

# Modified Backtesting Function
def modified_backtest_strategy(data_aapl, data_spy, upper_thresh, lower_thresh, atr_window, risk_reward_ratio,
                               return_pairs_data=False):
    # Initial ATR computation
    data_aapl['atr'] = compute_atr(data_aapl, window=atr_window)
    data_spy['atr'] = compute_atr(data_spy, window=atr_window)

    # Pairs data computation
    pairs_data = pd.DataFrame({
        'spread': data_aapl['p_change'] - data_spy['p_change'],
        'ma': (data_aapl['p_change'] - data_spy['p_change']).rolling(window=ma_window).mean(),
        'std': (data_aapl['p_change'] - data_spy['p_change']).rolling(window=ma_window).std(),
    })
    pairs_data['std_up'] = pairs_data['ma'] + pairs_data['std']
    pairs_data['std_low'] = pairs_data['ma'] - pairs_data['std']

    z_score_signals(upper_thresh, lower_thresh, data=pairs_data)

    # Stop-loss and take-profit computation
    data_aapl['sl'] = data_aapl['atr']
    data_aapl['tp'] = data_aapl['atr'] * risk_reward_ratio
    data_spy['sl'] = data_spy['atr']
    data_spy['tp'] = data_spy['atr'] * risk_reward_ratio

    # Initialize trading columns
    pairs_data['aapl_position'] = 0
    pairs_data['spy_position'] = 0
    pairs_data['aapl_daily_pl'] = 0
    pairs_data['spy_daily_pl'] = 0

    # Setting trading positions based on z-scores
    pairs_data.loc[pairs_data['z_scores'] > upper_thresh, 'aapl_position'] = -1
    pairs_data.loc[pairs_data['z_scores'] > upper_thresh, 'spy_position'] = 1
    pairs_data.loc[pairs_data['z_scores'] < lower_thresh, 'aapl_position'] = 1
    pairs_data.loc[pairs_data['z_scores'] < lower_thresh, 'spy_position'] = -1

    aapl_open_position = None
    spy_open_position = None

    # For keeping track of entry and exit points
    aapl_entries = []
    aapl_exits = []
    spy_entries = []
    spy_exits = []

    for idx, row in pairs_data.iterrows():
        # AAPL trading logic
        if row['aapl_position'] != 0 and aapl_open_position is None:
            aapl_open_position = {'entry_price': data_aapl.loc[idx, 'Close'], 'position': row['aapl_position']}
            aapl_entries.append((idx, data_aapl.loc[idx, 'Close']))
        elif aapl_open_position:
            pnl_pct = (data_aapl.loc[idx, 'Close'] - aapl_open_position['entry_price']) / aapl_open_position[
                'entry_price'] * 100 * aapl_open_position['position']
            if pnl_pct <= -data_aapl.loc[idx, 'sl'] or pnl_pct >= data_aapl.loc[idx, 'tp']:
                aapl_exits.append((idx, data_aapl.loc[idx, 'Close']))
                aapl_open_position = None

        # SPY trading logic
        if row['spy_position'] != 0 and spy_open_position is None:
            spy_open_position = {'entry_price': data_spy.loc[idx, 'Close'], 'position': row['spy_position']}
            spy_entries.append((idx, data_spy.loc[idx, 'Close']))
        elif spy_open_position:
            pnl_pct = (data_spy.loc[idx, 'Close'] - spy_open_position['entry_price']) / spy_open_position[
                'entry_price'] * 100 * spy_open_position['position']
            if pnl_pct <= -data_spy.loc[idx, 'sl'] or pnl_pct >= data_spy.loc[idx, 'tp']:
                spy_exits.append((idx, data_spy.loc[idx, 'Close']))
                spy_open_position = None

        # Compute daily P&L
        if aapl_open_position:
            pairs_data.loc[idx, 'aapl_daily_pl'] = (data_aapl.loc[idx, 'Close'] - aapl_open_position['entry_price']) / \
                                                   aapl_open_position['entry_price'] * 100 * aapl_open_position[
                                                       'position']
        if spy_open_position:
            pairs_data.loc[idx, 'spy_daily_pl'] = (data_spy.loc[idx, 'Close'] - spy_open_position['entry_price']) / \
                                                  spy_open_position['entry_price'] * 100 * spy_open_position['position']

    # Compute cumulative P&L
    pairs_data['aapl_cum_pl'] = pairs_data['aapl_daily_pl'].cumsum()
    pairs_data['spy_cum_pl'] = pairs_data['spy_daily_pl'].cumsum()
    pairs_data['total_cum_pl'] = pairs_data['aapl_cum_pl'] + pairs_data['spy_cum_pl']



    return pairs_data

# Grid Search Logic (as in original script)
z_score_lower_thresholds = [-3, -2.5, -2, -1.5]
z_score_upper_thresholds = [1.5, 2, 2.5, 3]
atr_windows = [10, 14, 20, 30]
risk_reward_ratios = [1, 1.5, 2, 2.5, 3]

results = []

for lower_thresh in z_score_lower_thresholds:
    for upper_thresh in z_score_upper_thresholds:
        for atr_window in atr_windows:
            for rr_ratio in risk_reward_ratios:
                data = modified_backtest_strategy(aapl_data.copy(), spy_data.copy(), upper_thresh, lower_thresh, atr_window, rr_ratio)
                total_cum_pl = data['total_cum_pl'].iloc[-1]
                results.append({
                    'lower_thresh': lower_thresh,
                    'upper_thresh': upper_thresh,
                    'atr_window': atr_window,
                    'risk_reward_ratio': rr_ratio,
                    'total_cum_pl': total_cum_pl
                })



results_df = pd.DataFrame(results)
top_results = results_df.sort_values(by='total_cum_pl', ascending=False).head(5)
print("Top 5 Parameter Combinations Based on Cumulative P&L:")
print(top_results)



best_lower_thresh = top_results.iloc[0]['lower_thresh']
best_upper_thresh = top_results.iloc[0]['upper_thresh']
best_atr_window = int(top_results.iloc[0]['atr_window'])
best_rr_ratio = top_results.iloc[0]['risk_reward_ratio']

cumulative_pl_series = modified_backtest_strategy(aapl_data.copy(), spy_data.copy(),
                                                  best_upper_thresh, best_lower_thresh,
                                                  best_atr_window, best_rr_ratio)['total_cum_pl']

plt.figure(figsize=(12, 6))
cumulative_pl_series.plot()
plt.title('Cumulative P&L Over Time with Optimal Parameters')
plt.xlabel('Time')
plt.ylabel('Cumulative P&L (%)')
plt.grid(True)
plt.show()

# Statistical Display of P&L Distribution
pnl_statistics = results_df['total_cum_pl'].describe()
print("\\nStatistics on P&L Distribution Among All Parameter Combinations:")
print(pnl_statistics)


def plot_pair_trading_backtest(pairs_data):
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    ax[0].plot(aapl_data['Close'].pct_change() * 100, color='blue')
    ax[0].plot(spy_data['Close'].pct_change() * 100, color='orange')
    ax[1].plot(pairs_data['spread'])
    ax[1].plot(pairs_data['ma'])
    ax[1].plot(pairs_data['std_up'], linestyle='--', color='green')
    ax[1].plot(pairs_data['std_low'], linestyle='--', color='green')
    plt.show()

def plot_cumulative_pl(cumulative_pl_series):
    plt.figure(figsize=(12, 6))
    cumulative_pl_series.plot()
    plt.title('Cumulative P&L Over Time with Optimal Parameters')
    plt.xlabel('Time')
    plt.ylabel('Cumulative P&L (%)')
    plt.grid(True)
    plt.show()
def main_cli():

    print(aapl_data.columns)

    pairs_data = modified_backtest_strategy(aapl_data_untrained.copy(), spy_data_untrained.copy(), best_upper_thresh, best_lower_thresh,
                                            best_atr_window, best_rr_ratio, return_pairs_data=True)
    plot_pair_trading_backtest(pairs_data)
    plot_cumulative_pl(cumulative_pl_series)

    plot_trading_signals(aapl_data, spy_data, pairs_data)


main_cli()


#bruhz
if __name__ == '__main__':
    pass






