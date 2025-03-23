import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import torch
from datetime import datetime
from collections import defaultdict
import json

# Import dei nostri moduli
from portfolio_agent import PortfolioAgent
from portfolio_env import PortfolioEnvironment
from portfolio_models import PortfolioActor, PortfolioCritic, EnhancedPortfolioActor

# Configurazione
TICKERS = ["ARKG", "IBB", "IHI", "IYH", "XBI", "VHT"]  # I ticker nel portafoglio
BASE_PATH = '/Users/Alessandro/Desktop/DRL 2/NAS Results/Multi_Ticker/Normalized_RL_INPUT/'
NORM_PARAMS_PATH_BASE = f'{BASE_PATH}json/'
CSV_PATH_BASE = f'{BASE_PATH}'
OUTPUT_DIR = 'results/portfolio'
EVALUATION_DIR = f'{OUTPUT_DIR}/evaluation'
BENCHMARK_TICKER = 'SPY'  # Usato per confrontare le performance

# Crea directory di output se non esiste
os.makedirs(EVALUATION_DIR, exist_ok=True)

# Feature da utilizzare (manteniamo tutte le feature originali)
norm_columns = [
    "open", "volume", "change", "day", "week", "adjCloseGold", "adjCloseSpy",
    "Credit_Spread", "Log_Close", "m_plus", "m_minus", "drawdown", "drawup",
    "s_plus", "s_minus", "upper_bound", "lower_bound", "avg_duration", "avg_depth",
    "cdar_95", "VIX_Close", "MACD", "MACD_Signal", "MACD_Histogram", "SMA5",
    "SMA10", "SMA15", "SMA20", "SMA25", "SMA30", "SMA36", "RSI5", "RSI14", "RSI20",
    "RSI25", "ADX5", "ADX10", "ADX15", "ADX20", "ADX25", "ADX30", "ADX35",
    "BollingerLower", "BollingerUpper", "WR5", "WR14", "WR20", "WR25",
    "SMA5_SMA20", "SMA5_SMA36", "SMA20_SMA36", "SMA5_Above_SMA20",
    "Golden_Cross", "Death_Cross", "BB_Position", "BB_Width",
    "BB_Upper_Distance", "BB_Lower_Distance", "Volume_SMA20", "Volume_Change_Pct",
    "Volume_1d_Change_Pct", "Volume_Spike", "Volume_Collapse", "GARCH_Vol",
    "pred_lstm", "pred_gru", "pred_blstm", "pred_lstm_direction",
    "pred_gru_direction", "pred_blstm_direction"
]

def load_test_data():
    """Carica i dati di test per tutti i ticker."""
    dfs_test = {}
    norm_params_paths = {}
    valid_tickers = []
    
    for ticker in TICKERS:
        test_file = f'{OUTPUT_DIR}/test/{ticker}_test_aligned.csv'
        norm_params_path = f'{NORM_PARAMS_PATH_BASE}{ticker}_norm_params.json'
        
        # Verifica esistenza dei file
        if not os.path.exists(test_file):
            if os.path.exists(f'{OUTPUT_DIR}/test/{ticker}_test.csv'):
                test_file = f'{OUTPUT_DIR}/test/{ticker}_test.csv'
            else:
                print(f"File di test mancante per {ticker}. Provo a usare i dati originali.")
                # Tenta di caricare e preparare i dati dai file originali
                csv_path = f'{CSV_PATH_BASE}{ticker}/{ticker}_normalized.csv'
                if not os.path.exists(csv_path):
                    print(f"Salto il ticker {ticker} a causa di file mancanti")
                    continue
                
                # Carica il dataset
                df = pd.read_csv(csv_path)
                
                # Verifica le colonne
                missing_cols = [col for col in norm_columns if col not in df.columns]
                if missing_cols:
                    print(f"Salto il ticker {ticker}. Colonne mancanti: {missing_cols}")
                    continue
                
                # Ordina e prendi la parte di test
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                
                train_size = int(len(df) * 0.8)
                df_test = df.iloc[train_size:]
                df_test.to_csv(f'{OUTPUT_DIR}/test/{ticker}_test.csv', index=False)
                print(f"Creato file di test per {ticker} dai dati originali")
                test_file = f'{OUTPUT_DIR}/test/{ticker}_test.csv'
        
        if not os.path.exists(norm_params_path):
            print(f"File parametri normalizzazione mancante per {ticker}")
            continue
        
        # Carica i dati
        df_test = pd.read_csv(test_file)
        if 'date' in df_test.columns:
            df_test['date'] = pd.to_datetime(df_test['date'])
        
        dfs_test[ticker] = df_test
        norm_params_paths[ticker] = norm_params_path
        valid_tickers.append(ticker)
        print(f"Dati di test caricati per {ticker}: {len(df_test)} righe")
    
    return dfs_test, norm_params_paths, valid_tickers

def align_dataframes(dfs):
    """Allinea i DataFrame in modo che abbiano lo stesso intervallo di date."""
    aligned_dfs = {}
    
    # Trova l'intervallo di date comune
    if all('date' in df.columns for df in dfs.values()):
        # Trova la data di inizio più recente
        start_date = max(df['date'].min() for df in dfs.values())
        # Trova la data di fine più vecchia
        end_date = min(df['date'].max() for df in dfs.values())
        
        print(f"Intervallo di date comune: {start_date} - {end_date}")
        
        # Filtra e allinea ogni DataFrame
        for ticker, df in dfs.items():
            aligned_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
            # Assicurati che le date siano ordinate
            aligned_df = aligned_df.sort_values('date')
            aligned_dfs[ticker] = aligned_df
    else:
        # Se non ci sono colonne 'date', usa il numero minimo di righe
        min_rows = min(len(df) for df in dfs.values())
        for ticker, df in dfs.items():
            aligned_dfs[ticker] = df.iloc[:min_rows].copy()
    
    return aligned_dfs

def load_benchmark_data(start_date, end_date):
    """Carica dati per benchmark (SPY) per lo stesso periodo di tempo."""
    benchmark_path = f'{CSV_PATH_BASE}{BENCHMARK_TICKER}/{BENCHMARK_TICKER}_normalized.csv'
    if not os.path.exists(benchmark_path):
        print(f"Dati benchmark {BENCHMARK_TICKER} non trovati. Usa 'adjCloseSpy' dai dati esistenti.")
        return None
    
    try:
        benchmark_df = pd.read_csv(benchmark_path)
        if 'date' in benchmark_df.columns:
            benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
            benchmark_df = benchmark_df[(benchmark_df['date'] >= start_date) & 
                                       (benchmark_df['date'] <= end_date)]
            benchmark_df = benchmark_df.sort_values('date')
            print(f"Dati benchmark caricati: {len(benchmark_df)} righe")
            return benchmark_df
    except Exception as e:
        print(f"Errore caricamento benchmark: {e}")
    
    return None

def create_test_environment(dfs_test, norm_params_paths, valid_tickers):
    """Crea l'ambiente di test con i dati allineati."""
    max_steps = min(len(df) for df in dfs_test.values())
    
    env = PortfolioEnvironment(
        tickers=valid_tickers,
        sigma=0.1,
        theta=0.1,
        T=max_steps,
        lambd=0.05,
        psi=0.2,
        cost="trade_l1",
        max_pos_per_asset=2.0,
        max_portfolio_pos=6.0,
        squared_risk=False,
        penalty="tanh",
        alpha=3,
        beta=3,
        clip=True,
        scale_reward=5,
        dfs=dfs_test,
        max_step=max_steps,
        norm_params_paths=norm_params_paths,
        norm_columns=norm_columns,
        free_trades_per_month=10,
        commission_rate=0.0025,
        min_commission=1.0,
        trading_frequency_penalty_factor=0.1,
        position_stability_bonus_factor=0.2,
        correlation_penalty_factor=0.15,
        diversification_bonus_factor=0.15,
        initial_capital=100000,
        risk_free_rate=0.02,
        use_sortino=True,
        target_return=0.05
    )
    
    return env

def initialize_agent(num_assets, model_file=None, use_enhanced_actor=True):
    """Inizializza l'agente e carica il modello se specificato."""
    agent = PortfolioAgent(
        num_assets=num_assets,
        memory_type="prioritized",
        batch_size=256,
        theta=0.1,
        sigma=0.2,
        use_enhanced_actor=use_enhanced_actor,
        use_batch_norm=True
    )
    
    if model_file and os.path.exists(model_file):
        print(f"Caricamento del modello: {model_file}")
        critic_file = model_file.replace('actor', 'critic')
        
        # Inizializza le reti prima di caricare i pesi
        if use_enhanced_actor:
            features_per_asset = len(norm_columns)
            agent.actor_local = EnhancedPortfolioActor(
                state_size=len(norm_columns) * num_assets + num_assets + 5,
                action_size=num_assets,
                features_per_asset=features_per_asset,
                encoding_size=32
            )
        else:
            agent.actor_local = PortfolioActor(
                state_size=len(norm_columns) * num_assets + num_assets + 5,
                action_size=num_assets
            )
        
        # Carica i pesi se esistono
        agent.load_models(
            actor_path=model_file,
            critic_path=critic_file if os.path.exists(critic_file) else None
        )
    
    return agent

def run_backtest(env, agent, record_interval=5):
    """
    Esegue backtest sul modello usando i dati di test.
    
    Parametri:
        env: L'ambiente di portafoglio
        agent: L'agente con il modello caricato
        record_interval: Ogni quanti step registrare i dati
    
    Ritorna:
        dict con i risultati
    """
    env.reset()
    state = env.get_state()
    done = env.done
    
    # Tracciamento dei dati
    dates = []
    portfolio_values = []
    positions = {ticker: [] for ticker in env.tickers}
    actions = {ticker: [] for ticker in env.tickers}
    prices = {ticker: [] for ticker in env.tickers}
    returns = []
    rewards = []
    cash_history = []
    
    # Date from dataframe (assume tutti i df hanno le stesse date)
    if 'date' in next(iter(env.dfs.values())).columns:
        all_dates = next(iter(env.dfs.values()))['date'].values
    else:
        all_dates = [None] * len(next(iter(env.dfs.values())))
    
    step = 0
    date_counter = 0
    
    # Esegui il backtest
    while not done:
        # Esegui l'azione dall'agente (senza rumore per test)
        with torch.no_grad():
            actions_vector = agent.act(state, noise=False)
        
        # Registra i dati ogni N step
        if step % record_interval == 0:
            if date_counter < len(all_dates):
                dates.append(all_dates[date_counter])
            
            portfolio_values.append(env.get_portfolio_value())
            
            # Registra posizioni, prezzi e azioni per ogni asset
            for i, ticker in enumerate(env.tickers):
                positions[ticker].append(env.positions[i])
                prices[ticker].append(env.prices[i])
                actions[ticker].append(actions_vector[i])
            
            cash_history.append(env.cash)
            
            # Calcola rendimento
            if len(portfolio_values) > 1:
                ret = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                returns.append(ret)
        
        # Esegui azione
        reward = env.step(actions_vector)
        rewards.append(reward)
        
        # Aggiorna lo stato
        state = env.get_state()
        done = env.done
        
        step += 1
        date_counter += 1
    
    # Calcola metriche e raccogli risultati
    metrics = env.get_real_portfolio_metrics()
    
    # Converti i prezzi da scaled a real
    denormalized_prices = {}
    if hasattr(env, 'denormalize_price'):
        for ticker in env.tickers:
            denormalized_prices[ticker] = []
            for p in prices[ticker]:
                real_p = env.denormalize_price(ticker, p, "Log_Close")
                denormalized_prices[ticker].append(real_p)
    
    # Calcola altre metriche avanzate
    if len(returns) > 0:
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        # Metriche di drawdown
        cum_returns = np.cumprod(1 + np.array(returns))
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns / peak) - 1
        max_drawdown = np.min(drawdown)
        
        # Calcola ulcer index (radice della media dei quadrati dei drawdown)
        ulcer_index = np.sqrt(np.mean(np.square(drawdown))) if len(drawdown) > 0 else 0
        
        # Metrica di rischio vs rendimento
        calmar_ratio = (np.mean(returns) * 252) / (abs(max_drawdown) + 1e-8)
    else:
        sharpe = 0
        max_drawdown = 0
        ulcer_index = 0
        calmar_ratio = 0
    
    # Numero di operazioni e turnover
    total_trades = 0
    turnover = 0
    for ticker in env.tickers:
        ticker_actions = np.array(actions[ticker])
        ticker_trades = np.sum(np.abs(ticker_actions) > 1e-6)
        total_trades += ticker_trades
        
        # Calcola turnover (somma dei cambi di posizione in valore assoluto)
        ticker_positions = np.array(positions[ticker])
        if len(ticker_positions) > 1:
            ticker_turnover = np.sum(np.abs(ticker_positions[1:] - ticker_positions[:-1]))
            turnover += ticker_turnover
    
    results = {
        'dates': dates,
        'portfolio_values': portfolio_values,
        'positions': positions,
        'actions': actions,
        'prices': prices,
        'returns': returns,
        'rewards': rewards,
        'cash_history': cash_history,
        'metrics': metrics,
        'denormalized_prices': denormalized_prices,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'ulcer_index': ulcer_index,
        'calmar_ratio': calmar_ratio,
        'total_trades': total_trades,
        'turnover': turnover
    }
    
    return results

def calculate_benchmark_performance(results, dfs_test, benchmark_df=None):
    """
    Calcola le performance dei benchmark:
    1. Buy & Hold ugualmente pesato per tutti gli asset
    2. SPY o indice di mercato se disponibile
    """
    benchmark_results = {}
    
    # Ottieni le date dal backtest
    dates = results['dates']
    if not dates or len(dates) < 2:
        print("Dati insufficienti per calcolare performance benchmark")
        return benchmark_results
    
    # 1. Strategia Buy & Hold equamente pesata
    if len(results['prices']) > 0:
        initial_capital = 100000  # Stesso valore usato nel backtest
        
        # Inizializza
        equal_weight = 1.0 / len(results['prices'])
        bh_portfolio_values = []
        bh_returns = []
        
        # Calcola valore portafoglio per ogni data
        for i in range(len(dates)):
            # Se i == 0, compra e tieni la stessa quantità di ogni asset
            if i == 0:
                # Calcola quante unità comprare di ogni asset
                units = {}
                first_date_prices = {}
                
                for ticker in results['prices'].keys():
                    if i < len(results['prices'][ticker]):
                        price = results['prices'][ticker][i]
                        first_date_prices[ticker] = price
                        
                        # Allocazione uguale per ogni asset
                        allocation = initial_capital * equal_weight
                        units[ticker] = allocation / price if price > 0 else 0
                
                # Calcola valore iniziale del portafoglio
                portfolio_value = sum(units[t] * first_date_prices[t] for t in units)
                bh_portfolio_values.append(portfolio_value)
            else:
                # Calcola valore del portafoglio con le stesse unità ma prezzi aggiornati
                portfolio_value = 0
                for ticker in units:
                    if i < len(results['prices'][ticker]):
                        price = results['prices'][ticker][i]
                        ticker_value = units[ticker] * price
                        portfolio_value += ticker_value
                
                bh_portfolio_values.append(portfolio_value)
                
                # Calcola rendimento
                if i > 0 and bh_portfolio_values[i-1] > 0:
                    ret = (bh_portfolio_values[i] - bh_portfolio_values[i-1]) / bh_portfolio_values[i-1]
                    bh_returns.append(ret)
        
        # Calcola metriche
        if len(bh_returns) > 0:
            bh_cumulative_return = (bh_portfolio_values[-1] / bh_portfolio_values[0]) - 1
            bh_sharpe = np.mean(bh_returns) / (np.std(bh_returns) + 1e-8) * np.sqrt(252)
            
            # Metriche di drawdown
            bh_cum_returns = np.cumprod(1 + np.array(bh_returns))
            bh_peak = np.maximum.accumulate(bh_cum_returns)
            bh_drawdown = (bh_cum_returns / bh_peak) - 1
            bh_max_drawdown = np.min(bh_drawdown)
            
            benchmark_results['buy_and_hold'] = {
                'portfolio_values': bh_portfolio_values,
                'returns': bh_returns,
                'cumulative_return': bh_cumulative_return * 100,  # in percentuale
                'sharpe': bh_sharpe,
                'max_drawdown': bh_max_drawdown * 100  # in percentuale
            }
    
    # 2. Benchmark di mercato (SPY)
    if benchmark_df is not None and 'date' in benchmark_df.columns and 'Log_Close' in benchmark_df.columns:
        # Allinea date
        market_values = []
        market_returns = []
        
        # Trova il prezzo iniziale
        first_date = dates[0]
        first_idx = benchmark_df[benchmark_df['date'] == first_date].index
        
        if len(first_idx) > 0:
            initial_price = benchmark_df.loc[first_idx[0], 'Log_Close']
            initial_units = initial_capital / initial_price
            
            for d in dates:
                idx = benchmark_df[benchmark_df['date'] == d].index
                if len(idx) > 0:
                    price = benchmark_df.loc[idx[0], 'Log_Close']
                    portfolio_value = initial_units * price
                    market_values.append(portfolio_value)
                else:
                    if market_values:
                        market_values.append(market_values[-1])
                    else:
                        market_values.append(initial_capital)
            
            # Calcola rendimenti
            for i in range(1, len(market_values)):
                ret = (market_values[i] - market_values[i-1]) / market_values[i-1]
                market_returns.append(ret)
            
            # Calcola metriche
            if len(market_returns) > 0:
                market_cumulative_return = (market_values[-1] / market_values[0]) - 1
                market_sharpe = np.mean(market_returns) / (np.std(market_returns) + 1e-8) * np.sqrt(252)
                
                # Metriche di drawdown
                market_cum_returns = np.cumprod(1 + np.array(market_returns))
                market_peak = np.maximum.accumulate(market_cum_returns)
                market_drawdown = (market_cum_returns / market_peak) - 1
                market_max_drawdown = np.min(market_drawdown)
                
                benchmark_results['market'] = {
                    'portfolio_values': market_values,
                    'returns': market_returns,
                    'cumulative_return': market_cumulative_return * 100,  # in percentuale
                    'sharpe': market_sharpe,
                    'max_drawdown': market_max_drawdown * 100  # in percentuale
                }
    
    return benchmark_results

def visualize_results(results, benchmark_results, output_dir):
    """Genera visualizzazioni dettagliate dei risultati del backtest."""
    # Crea figura principale
    fig = plt.figure(figsize=(16, 20))
    
    # 1. Performance del portafoglio vs benchmark
    ax1 = plt.subplot(4, 1, 1)
    
    # Plot performance del modello
    portfolio_values = np.array(results['portfolio_values'])
    normalized_values = portfolio_values / portfolio_values[0] if len(portfolio_values) > 0 else []
    ax1.plot(results['dates'], normalized_values, label='Portfolio RL', linewidth=2, color='blue')
    
    # Plot benchmark se disponibile
    if 'buy_and_hold' in benchmark_results:
        bh_values = np.array(benchmark_results['buy_and_hold']['portfolio_values'])
        bh_normalized = bh_values / bh_values[0] if len(bh_values) > 0 else []
        ax1.plot(results['dates'], bh_normalized, label='Equal Weight Buy & Hold', linestyle='--', color='green')
    
    if 'market' in benchmark_results:
        market_values = np.array(benchmark_results['market']['portfolio_values'])
        market_normalized = market_values / market_values[0] if len(market_values) > 0 else []
        ax1.plot(results['dates'], market_normalized, label=f'{BENCHMARK_TICKER} Index', linestyle='-.', color='red')
    
    ax1.set_title('Portfolio Performance (Normalized)', fontsize=14)
    ax1.set_ylabel('Normalized Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Formatta asse x con date leggibili
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Allocazione dinamica nel tempo
    ax2 = plt.subplot(4, 1, 2)
    
    # Converti posizioni in percentuali di allocazione
    position_data = []
    total_abs_positions = []
    tickers = results['positions'].keys()
    
    # Calcola la posizione totale assoluta per ogni timestep
    for t in range(len(next(iter(results['positions'].values())))):
        abs_sum = sum(abs(results['positions'][ticker][t]) for ticker in tickers if t < len(results['positions'][ticker]))
        total_abs_positions.append(abs_sum if abs_sum > 0 else 1)
    
    # Calcola le percentuali di allocazione
    for ticker in tickers:
        ticker_positions = []
        for t in range(len(results['positions'][ticker])):
            if t < len(total_abs_positions):
                ticker_pct = 100 * results['positions'][ticker][t] / total_abs_positions[t]
                ticker_positions.append(ticker_pct)
        position_data.append(ticker_positions)
    
    # Converti in array numpy per lo stacking
    positions_array = np.array(position_data)
    
    # Split in long e short positions per visualizzazione
    long_positions = np.maximum(positions_array, 0)
    short_positions = np.minimum(positions_array, 0)
    
    # Plot long positions (sopra lo zero)
    ax2.stackplot(results['dates'], long_positions, labels=tickers, alpha=0.7)
    
    # Plot short positions (sotto lo zero)
    ax2.stackplot(results['dates'], short_positions, labels=[], alpha=0.7)
    
    ax2.set_title('Portfolio Allocation Over Time', fontsize=14)
    ax2.set_ylabel('Allocation (%)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Formatta asse x con date leggibili
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Evoluzione del cash e valore totale
    ax3 = plt.subplot(4, 1, 3)
    
    # Plot cash e valore totale
    ax3.plot(results['dates'], results['cash_history'], label='Cash', color='orange')
    ax3.plot(results['dates'], results['portfolio_values'], label='Portfolio Value', color='purple')
    
    ax3.set_title('Cash vs Portfolio Value', fontsize=14)
    ax3.set_ylabel('Value ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Formatta asse x con date leggibili
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Trading activity
    ax4 = plt.subplot(4, 1, 4)
    
    # Plot volume of trades per asset
    for ticker in tickers:
        abs_actions = np.abs(results['actions'][ticker])
        ax4.plot(results['dates'], abs_actions, label=f'{ticker} Trading Activity')
    
    ax4.set_title('Trading Activity Over Time', fontsize=14)
    ax4.set_ylabel('Absolute Position Change')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Formatta asse x con date leggibili
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/portfolio_performance.png')
    plt.close()
    
    # ----- Figura supplementare: Heatmap delle correlazioni -----
    plt.figure(figsize=(12, 10))
    
    # Crea dataframe con i rendimenti
    returns_dict = {ticker: [] for ticker in tickers}
    
    # Calcola i rendimenti per asset
    for ticker in tickers:
        ticker_prices = results['prices'][ticker]
        if len(ticker_prices) > 1:
            ticker_returns = np.diff(ticker_prices) / ticker_prices[:-1]
            returns_dict[ticker] = ticker_returns
    
    # Trova la lunghezza minima per allineare i rendimenti
    min_length = min(len(returns) for returns in returns_dict.values())
    for ticker in returns_dict:
        returns_dict[ticker] = returns_dict[ticker][:min_length]
    
    # Crea DataFrame per la heatmap
    returns_df = pd.DataFrame(returns_dict)
    
    # Calcola matrice di correlazione
    corr_matrix = returns_df.corr()
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Matrix of Asset Returns', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png')
    plt.close()
    
    # ----- Figura supplementare: Distribuzione dei rendimenti -----
    plt.figure(figsize=(14, 8))
    
    # Violin plot dei rendimenti per asset
    if min_length > 0:
        ax = sns.violinplot(data=returns_df)
        plt.title('Distribution of Returns by Asset', fontsize=16)
        plt.ylabel('Daily Return')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/returns_distribution.png')
        plt.close()
    
    # ----- Figura supplementare: Drawdown analysis -----
    plt.figure(figsize=(14, 8))
    
    # Calcola drawdown cumulativo
    if len(results['returns']) > 1:
        cumulative_returns = np.cumprod(1 + np.array(results['returns']))
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns / peak - 1) * 100  # percentuale
        
        plt.plot(results['dates'][1:], drawdown, color='red', label='Drawdown (%)')
        plt.fill_between(results['dates'][1:], drawdown, 0, color='red', alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.title('Portfolio Drawdown Analysis', fontsize=16)
        plt.ylabel('Drawdown (%)')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Formatta asse x con date leggibili
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/drawdown_analysis.png')
        plt.close()
    
    # ----- Figura supplementare: Metriche comparative -----
    plt.figure(figsize=(14, 7))
    
    # Prepara dati per confronto
    strategies = ['Portfolio RL']
    returns = [results['metrics']['total_return']]
    sharpes = [results['sharpe']]
    drawdowns = [abs(results['metrics']['max_drawdown'])]
    turnover_values = [results['turnover']]
    
    if 'buy_and_hold' in benchmark_results:
        strategies.append('Equal Weight')
        returns.append(benchmark_results['buy_and_hold']['cumulative_return'])
        sharpes.append(benchmark_results['buy_and_hold']['sharpe'])
        drawdowns.append(abs(benchmark_results['buy_and_hold']['max_drawdown']))
        turnover_values.append(0)  # Buy & Hold ha turnover zero
    
    if 'market' in benchmark_results:
        strategies.append(f'{BENCHMARK_TICKER}')
        returns.append(benchmark_results['market']['cumulative_return'])
        sharpes.append(benchmark_results['market']['sharpe'])
        drawdowns.append(abs(benchmark_results['market']['max_drawdown']))
        turnover_values.append(0)  # Indice ha turnover zero
    
    # Grafico delle metriche comparative
    plt.subplot(1, 4, 1)
    plt.bar(strategies, returns, color=['blue', 'green', 'red'][:len(strategies)])
    plt.title('Total Return (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 4, 2)
    plt.bar(strategies, sharpes, color=['blue', 'green', 'red'][:len(strategies)])
    plt.title('Sharpe Ratio')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 4, 3)
    plt.bar(strategies, drawdowns, color=['blue', 'green', 'red'][:len(strategies)])
    plt.title('Max Drawdown (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 4, 4)
    plt.bar(strategies, turnover_values, color=['blue', 'green', 'red'][:len(strategies)])
    plt.title('Portfolio Turnover')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparative_metrics.png')
    plt.close()
    
    return corr_matrix, returns_df

def create_summary_report(results, benchmark_results, corr_matrix, returns_df, output_dir):
    """Crea un report di riepilogo in formato HTML."""
    from datetime import datetime
    
    # Prepara i dati di sommario
    summary = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'initial_capital': 100000,
        'test_period': f"{results['dates'][0].strftime('%Y-%m-%d')} - {results['dates'][-1].strftime('%Y-%m-%d')}",
        'n_trading_days': len(results['dates']),
        'portfolio_model': {
            'final_value': results['portfolio_values'][-1],
            'total_return': results['metrics']['total_return'],
            'annualized_return': results['metrics']['total_return'] / len(results['dates']) * 252,
            'sharpe': results['sharpe'],
            'max_drawdown': results['metrics']['max_drawdown'],
            'volatility': np.std(results['returns']) * np.sqrt(252) if len(results['returns']) > 0 else 0,
            'total_trades': results['total_trades'],
            'turnover': results['turnover'],
            'calmar_ratio': results['calmar_ratio'],
            'ulcer_index': results['ulcer_index']
        }
    }
    
    # Aggiungi benchmark se disponibili
    if 'buy_and_hold' in benchmark_results:
        summary['equal_weight'] = {
            'final_value': benchmark_results['buy_and_hold']['portfolio_values'][-1],
            'total_return': benchmark_results['buy_and_hold']['cumulative_return'],
            'annualized_return': benchmark_results['buy_and_hold']['cumulative_return'] / len(results['dates']) * 252,
            'sharpe': benchmark_results['buy_and_hold']['sharpe'],
            'max_drawdown': benchmark_results['buy_and_hold']['max_drawdown'],
            'volatility': np.std(benchmark_results['buy_and_hold']['returns']) * np.sqrt(252) if benchmark_results['buy_and_hold']['returns'] else 0
        }
    
    if 'market' in benchmark_results:
        summary['market'] = {
            'final_value': benchmark_results['market']['portfolio_values'][-1],
            'total_return': benchmark_results['market']['cumulative_return'],
            'annualized_return': benchmark_results['market']['cumulative_return'] / len(results['dates']) * 252,
            'sharpe': benchmark_results['market']['sharpe'],
            'max_drawdown': benchmark_results['market']['max_drawdown'],
            'volatility': np.std(benchmark_results['market']['returns']) * np.sqrt(252) if benchmark_results['market']['returns'] else 0
        }
    
    # Crea report HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Portfolio Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
            th {{ background-color: #f2f2f2; color: #333; text-align: center; }}
            .section {{ margin-bottom: 30px; }}
            .images {{ display: flex; flex-wrap: wrap; justify-content: center; }}
            .images img {{ max-width: 100%; height: auto; margin: 10px; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>Portfolio Backtesting Report</h1>
        <div class="section">
            <h2>Summary</h2>
            <p><strong>Date:</strong> {summary['date']}</p>
            <p><strong>Test Period:</strong> {summary['test_period']} ({summary['n_trading_days']} trading days)</p>
            <p><strong>Initial Capital:</strong> ${summary['initial_capital']:,.2f}</p>
        </div>
        
        <div class="section">
            <h2>Performance Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Portfolio RL</th>
    """
    
    if 'equal_weight' in summary:
        html_content += f"<th>Equal Weight</th>"
    if 'market' in summary:
        html_content += f"<th>{BENCHMARK_TICKER}</th>"
    
    html_content += f"""
                </tr>
                <tr>
                    <td>Final Value</td>
                    <td>${summary['portfolio_model']['final_value']:,.2f}</td>
    """
    
    if 'equal_weight' in summary:
        html_content += f"<td>${summary['equal_weight']['final_value']:,.2f}</td>"
    if 'market' in summary:
        html_content += f"<td>${summary['market']['final_value']:,.2f}</td>"
    
    html_content += f"""
                </tr>
                <tr>
                    <td>Total Return</td>
                    <td>{summary['portfolio_model']['total_return']:.2f}%</td>
    """
    
    if 'equal_weight' in summary:
        html_content += f"<td>{summary['equal_weight']['total_return']:.2f}%</td>"
    if 'market' in summary:
        html_content += f"<td>{summary['market']['total_return']:.2f}%</td>"
    
    html_content += f"""
                </tr>
                <tr>
                    <td>Annualized Return</td>
                    <td>{summary['portfolio_model']['annualized_return']:.2f}%</td>
    """
    
    if 'equal_weight' in summary:
        html_content += f"<td>{summary['equal_weight']['annualized_return']:.2f}%</td>"
    if 'market' in summary:
        html_content += f"<td>{summary['market']['annualized_return']:.2f}%</td>"
    
    html_content += f"""
                </tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td>{summary['portfolio_model']['sharpe']:.2f}</td>
    """
    
    if 'equal_weight' in summary:
        html_content += f"<td>{summary['equal_weight']['sharpe']:.2f}</td>"
    if 'market' in summary:
        html_content += f"<td>{summary['market']['sharpe']:.2f}</td>"
    
    html_content += f"""
                </tr>
                <tr>
                    <td>Max Drawdown</td>
                    <td>{summary['portfolio_model']['max_drawdown']:.2f}%</td>
    """
    
    if 'equal_weight' in summary:
        html_content += f"<td>{summary['equal_weight']['max_drawdown']:.2f}%</td>"
    if 'market' in summary:
        html_content += f"<td>{summary['market']['max_drawdown']:.2f}%</td>"
    
    html_content += f"""
                </tr>
                <tr>
                    <td>Volatility (Ann.)</td>
                    <td>{summary['portfolio_model']['volatility']:.2f}%</td>
    """
    
    if 'equal_weight' in summary:
        html_content += f"<td>{summary['equal_weight']['volatility']:.2f}%</td>"
    if 'market' in summary:
        html_content += f"<td>{summary['market']['volatility']:.2f}%</td>"
    
    html_content += f"""
                </tr>
                <tr>
                    <td>Calmar Ratio</td>
                    <td>{summary['portfolio_model']['calmar_ratio']:.2f}</td>
                    <td colspan="{1 if 'equal_weight' in summary else 0 + 1 if 'market' in summary else 0}">-</td>
                </tr>
                <tr>
                    <td>Total Trades</td>
                    <td>{summary['portfolio_model']['total_trades']}</td>
                    <td colspan="{1 if 'equal_weight' in summary else 0 + 1 if 'market' in summary else 0}">-</td>
                </tr>
                <tr>
                    <td>Turnover</td>
                    <td>{summary['portfolio_model']['turnover']:.2f}</td>
                    <td colspan="{1 if 'equal_weight' in summary else 0 + 1 if 'market' in summary else 0}">-</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Portfolio Allocation</h2>
            <table>
                <tr>
                    <th>Asset</th>
                    <th>Avg Position</th>
                    <th>Max Position</th>
                    <th>Min Position</th>
                    <th>Avg Abs Position</th>
                    <th>Trade Count</th>
                </tr>
    """
    
    for ticker in results['positions'].keys():
        positions = results['positions'][ticker]
        actions = results['actions'][ticker]
        
        avg_pos = np.mean(positions) if positions else 0
        max_pos = np.max(positions) if positions else 0
        min_pos = np.min(positions) if positions else 0
        avg_abs_pos = np.mean(np.abs(positions)) if positions else 0
        trade_count = np.sum(np.abs(actions) > 1e-6) if actions else 0
        
        html_content += f"""
                <tr>
                    <td>{ticker}</td>
                    <td>{avg_pos:.2f}</td>
                    <td>{max_pos:.2f}</td>
                    <td>{min_pos:.2f}</td>
                    <td>{avg_abs_pos:.2f}</td>
                    <td>{trade_count}</td>
                </tr>
        """
    
    html_content += f"""
            </table>
        </div>
        
        <div class="section">
            <h2>Visualizations</h2>
            <div class="images">
                <img src="portfolio_performance.png" alt="Portfolio Performance">
                <img src="correlation_heatmap.png" alt="Correlation Heatmap">
                <img src="returns_distribution.png" alt="Returns Distribution">
                <img src="drawdown_analysis.png" alt="Drawdown Analysis">
                <img src="comparative_metrics.png" alt="Comparative Metrics">
            </div>
        </div>
    </body>
    </html>
    """
    
    # Salva report HTML
    with open(f'{output_dir}/evaluation_report.html', 'w') as f:
        f.write(html_content)
    
    print(f"Report di valutazione salvato in: {output_dir}/evaluation_report.html")
    
    # Salva anche le metriche in formato JSON per usi futuri
    with open(f'{output_dir}/evaluation_metrics.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Metriche di valutazione salvate in: {output_dir}/evaluation_metrics.json")

def main():
    """Funzione principale per la valutazione del portafoglio."""
    print("Avvio valutazione del portfolio trading system...")
    
    # 1. Carica i dati di test
    dfs_test, norm_params_paths, valid_tickers = load_test_data()
    
    if not valid_tickers:
        print("Nessun ticker valido trovato. Uscita.")
        return
    
    print(f"Ticker validi per la valutazione: {valid_tickers}")
    
    # 2. Allinea i DataFrame di test
    print("Allineamento dei DataFrame di test...")
    aligned_dfs_test = align_dataframes(dfs_test)
    
    # Ottieni le date per benchmark
    start_date = None
    end_date = None
    if all('date' in df.columns for df in aligned_dfs_test.values()):
        first_df = next(iter(aligned_dfs_test.values()))
        start_date = first_df['date'].iloc[0]
        end_date = first_df['date'].iloc[-1]
    
    # 3. Carica benchmark se possibile
    benchmark_df = None
    if start_date is not None and end_date is not None:
        benchmark_df = load_benchmark_data(start_date, end_date)
    
    # 4. Crea l'ambiente di test
    print("Creazione dell'ambiente di test...")
    test_env = create_test_environment(aligned_dfs_test, norm_params_paths, valid_tickers)
    
    # 5. Cerca e carica il modello migliore
    best_model_file = None
    model_files = [f for f in os.listdir(f'{OUTPUT_DIR}/weights/') if f.startswith('portfolio_actor_') and f.endswith('.pth')]
    
    if not model_files:
        print("Nessun modello trovato. Uscita.")
        return
    
    # Ordina per episodio e prendi l'ultimo (o quello specificato)
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    best_model_file = model_files[-1]  # Default: usa l'ultimo modello
    
    print(f"Utilizzo del modello: {best_model_file}")
    best_model_path = f'{OUTPUT_DIR}/weights/{best_model_file}'
    
    # 6. Inizializza l'agente con il modello
    print("Inizializzazione dell'agente...")
    agent = initialize_agent(len(valid_tickers), best_model_path, use_enhanced_actor=True)
    
    # 7. Esegui il backtest
    print("Esecuzione del backtest...")
    results = run_backtest(test_env, agent)
    
    # 8. Calcola performance benchmark
    print("Calcolo performance benchmark...")
    benchmark_results = calculate_benchmark_performance(results, aligned_dfs_test, benchmark_df)
    
    # 9. Visualizza risultati
    print("Generazione visualizzazioni...")
    corr_matrix, returns_df = visualize_results(results, benchmark_results, EVALUATION_DIR)
    
    # 10. Crea report di riepilogo
    print("Creazione report di valutazione...")
    create_summary_report(results, benchmark_results, corr_matrix, returns_df, EVALUATION_DIR)
    
    print(f"Valutazione completata. Risultati salvati in: {EVALUATION_DIR}")

if __name__ == "__main__":
    main()