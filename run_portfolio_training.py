import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from collections import deque

# Import dei nostri moduli
from portfolio_agent import PortfolioAgent
from portfolio_env import PortfolioEnvironment
from portfolio_models import PortfolioActor, PortfolioCritic, EnhancedPortfolioActor


#py run_portfolio_training.py --resume="results\portfolio\weights\checkpoint_ep0.pt"

# Lista dei ticker da utilizzare nel portafoglio
TICKERS = ["ARKG", "IBB", "IHI", "IYH", "XBI", "VHT"]

# Configurazione di base
BASE_PATH = 'C:\\Users\\Administrator\\Desktop\\DRL PORTFOLIO\\NAS Results\\Multi_Ticker\\Normalized_RL_INPUT\\'
NORM_PARAMS_PATH_BASE = f'{BASE_PATH}json\\'
CSV_PATH_BASE = f'{BASE_PATH}'
OUTPUT_DIR = 'results\\portfolio'

# Crea directory di output se non esiste
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}\\weights', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}\\test', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}\\analysis', exist_ok=True)

# Feature da utilizzare (sottogruppo delle feature originali, per rendere il modello più gestibile)
# Scegliamo feature essenziali e rilevanti per analisi multi-asset
# Sostituire la lista norm_columns attuale (circa alla riga 36-51) con questa:
norm_columns = [
    "open", "volume", "change", "day", "week", "adjCloseGold", "adjCloseSpy",
    "Credit_Spread", #"Log_Close",
    "m_plus", "m_minus", "drawdown", "drawup",
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

def check_file_exists(file_path):
    """Verifica se un file esiste e stampa un messaggio appropriato."""
    if not os.path.exists(file_path):
        print(f"ATTENZIONE: File non trovato: {file_path}")
        return False
    return True

def load_data_for_tickers(tickers, train_fraction=0.8):
    """
    Carica e prepara i dati per tutti i ticker.
    
    Parametri:
    - tickers: lista di ticker da caricare
    - train_fraction: frazione dei dati da usare per il training (0.8 = 80%)
    
    Ritorna:
    - dfs_train: dict di DataFrame per training
    - dfs_test: dict di DataFrame per test
    - norm_params_paths: dict di percorsi ai parametri di normalizzazione
    """
    dfs_train = {}
    dfs_test = {}
    norm_params_paths = {}
    valid_tickers = []
    
    for ticker in tickers:
        norm_params_path = f'{NORM_PARAMS_PATH_BASE}{ticker}_norm_params.json'
        csv_path = f'{CSV_PATH_BASE}{ticker}\\{ticker}_normalized.csv'
        
        # Verifica esistenza dei file
        if not (check_file_exists(norm_params_path) and check_file_exists(csv_path)):
            print(f"Salto il ticker {ticker} a causa di file mancanti")
            continue
        
        # Carica il dataset
        print(f"Caricamento dati per {ticker}...")
        df = pd.read_csv(csv_path)
        
        # Verifica la presenza di tutte le colonne necessarie
        missing_cols = [col for col in norm_columns if col not in df.columns]
        if missing_cols:
            print(f"Salto il ticker {ticker}. Colonne mancanti: {missing_cols}")
            continue
        
        # Ordina il dataset per data (se presente)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        # Separazione in training e test
        train_size = int(len(df) * train_fraction)
        dfs_train[ticker] = df.iloc[:train_size]
        dfs_test[ticker] = df.iloc[train_size:]
        norm_params_paths[ticker] = norm_params_path
        
        valid_tickers.append(ticker)
        print(f"Dataset per {ticker} caricato: {len(df)} righe")
    
    # Aggiorna la lista dei ticker con quelli validi
    return dfs_train, dfs_test, norm_params_paths, valid_tickers

# Modifica la funzione save_results in run_portfolio_training.py
def save_results(results, output_dir, tickers):
    """Salva i risultati dell'addestramento."""
    try:
        # Estrai in modo sicuro le metriche finali
        if isinstance(results['final_rewards'], deque):
            final_reward = np.mean(list(results['final_rewards'])[-3:]) if len(results['final_rewards']) >= 3 else np.mean(list(results['final_rewards']))
        else:
            final_reward = results['final_rewards']
            
        if isinstance(results['final_portfolio_values'], deque):
            final_portfolio_value = np.mean(list(results['final_portfolio_values'])[-3:]) if len(results['final_portfolio_values']) >= 3 else np.mean(list(results['final_portfolio_values']))
        else:
            final_portfolio_value = results['final_portfolio_values']
            
        if isinstance(results['final_sharpe_ratios'], deque):
            final_sharpe_ratio = np.mean(list(results['final_sharpe_ratios'])[-3:]) if len(results['final_sharpe_ratios']) >= 3 else np.mean(list(results['final_sharpe_ratios']))
        else:
            final_sharpe_ratio = results['final_sharpe_ratios']
        
        # Crea DataFrame
        results_df = pd.DataFrame({
            'ticker': [', '.join(tickers)],  # Unisci i ticker in una stringa
            'final_reward': [final_reward],
            'final_portfolio_value': [final_portfolio_value],
            'final_sharpe_ratio': [final_sharpe_ratio],
            'training_timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        })
        
        # Salva il DataFrame
        results_file = f'{output_dir}\\training_results.csv'
        # Aggiungi i risultati a un file esistente o creane uno nuovo
        if os.path.exists(results_file):
            existing_results = pd.read_csv(results_file)
            updated_results = pd.concat([existing_results, results_df], ignore_index=True)
            updated_results.to_csv(results_file, index=False)
        else:
            results_df.to_csv(results_file, index=False)
        
        print(f"Risultati salvati in: {results_file}")
    except Exception as e:
        print(f"Errore durante il salvataggio dei risultati: {e}")
        # Salva i dati grezzi in formato pickle per analisi successiva
        import pickle
        with open(f'{output_dir}\\raw_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(f"Risultati grezzi salvati in: {output_dir}\\raw_results.pkl")

def plot_training_performance(results, output_dir, tickers):
    """Crea grafici per visualizzare le performance di addestramento."""
    plt.figure(figsize=(15, 10))
    
    # Plot cumulative rewards
    plt.subplot(2, 2, 1)
    plt.plot(results['cum_rewards'])
    plt.title('Ricompensa cumulativa')
    plt.xlabel('Episodi (x5)')
    plt.ylabel('Ricompensa media')
    plt.grid(True, alpha=0.3)
    
    # Plot portfolio values
    plt.subplot(2, 2, 2)
    plt.plot(results['final_portfolio_values'])
    plt.title('Valore del portafoglio finale')
    plt.xlabel('Episodi')
    plt.ylabel('Valore ($)')
    plt.grid(True, alpha=0.3)
    
    # Plot Sharpe ratios
    plt.subplot(2, 2, 3)
    plt.plot(results['final_sharpe_ratios'])
    plt.title('Sharpe Ratio')
    plt.xlabel('Episodi')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot ticker weights/allocation
    plt.subplot(2, 2, 4)
    plt.pie([1/len(tickers)]*len(tickers), labels=tickers, autopct='%1.1f%%', startangle=90)
    plt.title('Allocazione iniziale del portafoglio (equamente distribuita)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}\\training_performance.png')
    print(f"Grafico delle performance salvato in: {output_dir}\\training_performance.png")

def align_dataframes(dfs):
        """
        Allinea i DataFrame in modo che abbiano lo stesso intervallo di date 
        e lo stesso numero di righe.
        """
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
            
            # Verifica che tutti i DataFrame allineati abbiano lo stesso numero di righe
            lengths = [len(df) for df in aligned_dfs.values()]
            if len(set(lengths)) > 1:
                print(f"ATTENZIONE: I DataFrame allineati hanno lunghezze diverse: {lengths}")
                # Trova la lunghezza minima
                min_length = min(lengths)
                print(f"Troncamento a {min_length} righe...")
                # Tronca tutti i DataFrame alla stessa lunghezza
                for ticker in aligned_dfs:
                    aligned_dfs[ticker] = aligned_dfs[ticker].iloc[:min_length].copy()
        else:
            # Se non ci sono colonne 'date', usa il numero minimo di righe
            min_rows = min(len(df) for df in dfs.values())
            for ticker, df in dfs.items():
                aligned_dfs[ticker] = df.iloc[:min_rows].copy()
        
        # Verifica finale delle lunghezze
        lengths = [len(df) for df in aligned_dfs.values()]
        print(f"Lunghezze dei DataFrame allineati: {lengths}")
        
        return aligned_dfs
def diagnose_missing_columns(df, ticker, required_columns):
    """
    Diagnostica dettagliata delle colonne mancanti in un DataFrame.
    
    Parametri:
    - df: DataFrame da analizzare
    - ticker: nome del ticker per riferimento
    - required_columns: lista di colonne richieste
    
    Ritorna:
    - True se tutte le colonne sono presenti, False altrimenti
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        print(f"Ticker {ticker} manca di {len(missing_cols)}/{len(required_columns)} colonne richieste:")
        print(f"Colonne mancanti: {missing_cols}")
        
        # Verifica se ci sono colonne simili che potrebbero essere rinominate
        for missing in missing_cols:
            similar_cols = [col for col in df.columns if missing.lower() in col.lower()]
            if similar_cols:
                print(f"  Per '{missing}' ci sono colonne simili: {similar_cols}")
        
        return False
    
    return True

def main(resume_from=None):
    """Funzione principale per l'addestramento del portafoglio."""
    # 1. Carica e prepara i dati
    print("Caricamento dei dati per tutti i ticker...")
    dfs_train, dfs_test, norm_params_paths, valid_tickers = load_data_for_tickers(TICKERS)
    
    if not valid_tickers:
        print("Nessun ticker valido trovato. Uscita.")
        return
    
    print(f"Ticker validi: {valid_tickers}")
    
    # Allinea i DataFrame di training e test
    print("Allineamento dei DataFrame...")
    aligned_dfs_train = align_dataframes(dfs_train)
    aligned_dfs_test = align_dataframes(dfs_test)
    
    # Salva i DataFrame allineati per riferimento futuro
    for ticker, df in aligned_dfs_test.items():
        df.to_csv(f'{OUTPUT_DIR}\\test\\{ticker}_test_aligned.csv', index=False)
        # Nel ciclo che carica i dati per ogni ticker
        diagnose_missing_columns(df, ticker, norm_columns)
    
    # 2. Crea l'ambiente di portafoglio
    print("Inizializzazione dell'ambiente di portafoglio...")
    max_steps = min(1000, min(len(df) for df in aligned_dfs_train.values()) - 10)
    
    env = PortfolioEnvironment(
        tickers=valid_tickers,
        sigma=0.1,               # Parametro di volatilità per OU
        theta=0.1,               # Parametro di mean-reversion per OU
        T=max_steps,             # Numero massimo di timestep
        lambd=0.05,              # Penalità dimensione posizione
        psi=0.2,                 # Fattore costi di trading
        cost="trade_l1",         # Tipo di costo
        max_pos_per_asset=2.0,   # Posizione massima per asset
        max_portfolio_pos=6.0,   # Esposizione massima totale
        squared_risk=False,      # Usa penalità quadratica
        penalty="tanh",          # Tipo di penalità
        alpha=3,                 # Parametro penalty alpha
        beta=3,                  # Parametro penalty beta
        clip=True,               # Limita le posizioni
        scale_reward=5,          # Fattore di scala ricompensa
        dfs=aligned_dfs_train,   # DataFrame di training
        max_step=max_steps,      # Numero massimo di step
        norm_params_paths=norm_params_paths,  # Parametri di normalizzazione
        norm_columns=norm_columns,  # Colonne da usare
        free_trades_per_month=10,  # Operazioni gratuite al mese
        commission_rate=0.0025,    # Commissione percentuale
        min_commission=1.0,        # Commissione minima
        trading_frequency_penalty_factor=0.1,  # Penalità trading frequente
        position_stability_bonus_factor=0.2,   # Bonus stabilità posizioni
        correlation_penalty_factor=0.15,       # Penalità per correlazione
        diversification_bonus_factor=0.15,     # Bonus diversificazione
        initial_capital=100000,                # Capitale iniziale
        risk_free_rate=0.02,                   # Tasso risk-free
        use_sortino=True,                      # Usare Sortino ratio
        target_return=0.05                     # Rendimento target
    )
    
    # 3. Inizializza l'agente
    print("Inizializzazione dell'agente di portafoglio...")
    num_assets = len(valid_tickers)
    agent = PortfolioAgent(
        num_assets=num_assets,
        memory_type="prioritized",
        batch_size=256,          # Dimensione batch aumentata per multi-asset
        max_step=max_steps,
        theta=0.1,               # Parametro OU
        sigma=0.2,               # Parametro OU
        use_enhanced_actor=True,  # Usa l'actor con attenzione
        use_batch_norm=True      # Usa batch normalization
    )
    
    # 4. Avvia il training
    print(f"Avvio del training per il portafoglio con {num_assets} asset...")
    
    # Calcola la dimensione per feature per asset (per l'EnhancedPortfolioActor)
    features_per_asset = len(norm_columns)
    
    results = agent.train(
        env=env,
        total_episodes=300,         # Episodi aumentati per complessità maggiore
        tau_actor=0.01,             # Tasso update actor
        tau_critic=0.03,            # Tasso update critic
        lr_actor=5e-6,              # Learning rate actor (ridotto)
        lr_critic=1e-4,             # Learning rate critic (ridotto)
        weight_decay_actor=1e-6,    # Regolarizzazione L2 actor
        weight_decay_critic=1e-5,   # Regolarizzazione L2 critic
        total_steps=3000,           # Passi di pretraining aumentati
        weights=f'{OUTPUT_DIR}\\weights\\',
        freq=10,                     # Frequenza salvataggio
        fc1_units_actor=512,         # Layer più grande per gestire più asset
        fc2_units_actor=256,
        fc3_units_actor=128,
        fc1_units_critic=1024,       # Layer più grande per critic
        fc2_units_critic=512,
        fc3_units_critic=256,
        decay_rate=5e-7,             # Decay esplorazione più lento
        explore_stop=0.1,
        tensordir=f'{OUTPUT_DIR}\\runs\\',
        checkpoint_freq=10,          # Salva checkpoint ogni 10 episodi
        checkpoint_path=f'{OUTPUT_DIR}\\weights\\',
        resume_from=resume_from,            # Imposta a un percorso specifico se vuoi riprendere
        learn_freq=5,                # Aggiornamento più frequente
        plots=False,
        progress="tqdm",
        features_per_asset=features_per_asset,  # Per EnhancedPortfolioActor
        encoding_size=32,             # Dimensione encoding per asset
        clip_grad_norm=1.0            # Limite per gradient clipping
    )
    
    # 5. Salva risultati e crea visualizzazioni
    save_results(results, OUTPUT_DIR, valid_tickers)
    plot_training_performance(results, OUTPUT_DIR, valid_tickers)
    
    print(f"Training completato per il portafoglio!")
    print(f"I modelli addestrati sono stati salvati in: {OUTPUT_DIR}\\weights\\")
    print(f"I log per TensorBoard sono stati salvati in: {OUTPUT_DIR}\\runs\\")
    
    # 6. Crea una configurazione per l'ambiente di test
    print("\nPreparazione dell'ambiente di test...")
    test_env = PortfolioEnvironment(
        tickers=valid_tickers,
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
        dfs=aligned_dfs_test,           # Usa dati di test
        max_step=len(next(iter(aligned_dfs_test.values()))),  # Usa tutto il dataset di test
        norm_params_paths=norm_params_paths,
        norm_columns=norm_columns,
        free_trades_per_month=10,       # Parametri realistici per test
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
    
    # Carica il modello migliore
    model_files = [f for f in os.listdir(f'{OUTPUT_DIR}\\weights\\') if f.startswith('portfolio_actor_') and f.endswith('.pth')]
    if model_files:
        # Filtra solo i file numerici, escludendo 'initial'
        numeric_models = [f for f in model_files if f.split('_')[-1].split('.')[0].isdigit()]
        if numeric_models:
            last_model = sorted(numeric_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        else:
            # Se non ci sono modelli numerici, usa initial o il primo disponibile
            last_model = next((f for f in model_files if 'initial' in f), model_files[0] if model_files else None)
        
        last_critic = last_model.replace('actor', 'critic')
        
        print(f"Caricamento del modello migliore: {last_model}")
        agent.load_models(
            actor_path=f'{OUTPUT_DIR}\\weights\\{last_model}',
            critic_path=f'{OUTPUT_DIR}\\weights\\{last_critic}' if os.path.exists(f'{OUTPUT_DIR}\\weights\\{last_critic}') else None
        )
        
        # Esegui una valutazione iniziale
        print("Esecuzione di una valutazione sul dataset di test...")
        test_env.reset()
        state = test_env.get_state()
        done = test_env.done
        
        while not done:
            with torch.no_grad():
                actions = agent.act(state, noise=False)
            
            reward = test_env.step(actions)
            state = test_env.get_state()
            done = test_env.done
        
        # Stampa risultati finali
        metrics = test_env.get_real_portfolio_metrics()
        print("\nRisultati sul dataset di test:")
        print(f"Rendimento totale: {metrics['total_return']:.2f}%")
        print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Valore finale portafoglio: ${metrics['final_portfolio_value']:.2f}")
    else:
        print("Nessun modello trovato per la valutazione.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Addestra il portafoglio RL')
    parser.add_argument('--resume', type=str, help='Percorso del checkpoint da cui riprendere')
    args = parser.parse_args()
    
    main(resume_from=args.resume)