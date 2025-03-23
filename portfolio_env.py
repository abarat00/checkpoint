import numpy as np
import json
import pandas as pd
from utils import build_ou_process

class PortfolioEnvironment:
    """
    Ambiente per l'ottimizzazione di un portafoglio multi-asset.
    Lo stato è costituito dalle feature normalizzate per ciascun asset, più il vettore
    delle posizioni correnti. Le azioni rappresentano le variazioni nelle posizioni
    per ciascun asset nel portafoglio.
    """
    def __init__(
        self,
        tickers,                        # Lista dei ticker da gestire nel portafoglio
        sigma=0.5,                      # Parametro di volatilità per processo OU (simulazione)
        theta=1.0,                      # Parametro di mean-reversion per processo OU
        T=1000,                         # Numero massimo di timestep per episodio
        random_state=None,              # Seed per riproducibilità
        lambd=0.5,                      # Fattore di penalità per dimensione posizione
        psi=0.5,                        # Fattore di penalità per costi di trading
        cost="trade_l1",                # Tipo di costo di trading
        max_pos_per_asset=2.0,          # Posizione massima per singolo asset
        max_portfolio_pos=10.0,         # Esposizione massima totale del portafoglio
        squared_risk=True,              # Se usare penalità quadratica sul rischio
        penalty="tanh",                 # Tipo di penalità per violazione vincoli
        alpha=10,                       # Parametro alpha per la funzione di penalità
        beta=10,                        # Parametro beta per la funzione di penalità
        clip=True,                      # Se limitare le posizioni
        noise=False,                    # Se aggiungere rumore ai rendimenti
        noise_std=10,                   # Deviazione standard del rumore
        noise_seed=None,                # Seed per il rumore
        scale_reward=10,                # Fattore di scala per la ricompensa
        dfs=None,                       # Dict di DataFrame per ciascun ticker
        max_step=100,                   # Numero massimo di step per episodio
        norm_params_paths=None,         # Dict di percorsi ai parametri di normalizzazione
        norm_columns=None,              # Lista delle colonne da utilizzare
        free_trades_per_month=10,       # Operazioni gratuite al mese
        commission_rate=0.0025,         # Commissione percentuale
        min_commission=1.0,             # Commissione minima
        trading_frequency_penalty_factor=0.0,  # Penalità per trading frequente
        position_stability_bonus_factor=0.0,   # Bonus per stabilità posizione
        correlation_penalty_factor=0.1,        # Nuovo: penalità per correlazione elevata
        diversification_bonus_factor=0.1,      # Nuovo: bonus per diversificazione
        initial_capital=100000,                # Capitale iniziale per il portafoglio
        risk_free_rate=0.02,                   # Tasso risk-free annualizzato
        use_sortino=True,                      # Usare Sortino invece di Sharpe
        target_return=0.05,                    # Rendimento target per Sortino
        rebalance_penalty=0.01                 # Penalità per ribilanciamento frequente
    ):
        # Salva i ticker da gestire
        self.tickers = tickers
        self.num_assets = len(tickers)
        
        # Parametri di base
        self.sigma = sigma
        self.theta = theta
        self.T = T
        self.lambd = lambd
        self.psi = psi
        self.cost = cost
        self.max_pos_per_asset = max_pos_per_asset
        self.max_portfolio_pos = max_portfolio_pos
        self.squared_risk = squared_risk
        self.random_state = random_state
        self.penalty = penalty
        self.alpha = alpha
        self.beta = beta
        self.clip = clip
        self.scale_reward = scale_reward
        self.noise = noise
        self.noise_std = noise_std
        self.noise_seed = noise_seed
        self.max_step = max_step
        self.correlation_penalty_factor = correlation_penalty_factor
        self.diversification_bonus_factor = diversification_bonus_factor
        self.risk_free_rate = risk_free_rate
        self.use_sortino = use_sortino
        self.target_return = target_return
        self.rebalance_penalty = rebalance_penalty
        
        # Commissioni
        self.free_trades_per_month = free_trades_per_month
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.trading_frequency_penalty_factor = trading_frequency_penalty_factor
        self.position_stability_bonus_factor = position_stability_bonus_factor
        
        # Portfolio tracking
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.current_index = 0
        self.current_month = None
        self.trade_count = 0
        
        # Inizializzazione di strutture dati per ogni asset
        self.signals = {}            # Segnali di prezzo per asset (se simulati)
        self.positions = np.zeros(self.num_assets)  # Posizioni correnti per ogni asset
        self.prices = np.zeros(self.num_assets)     # Prezzi correnti per ogni asset
        self.raw_states = {}         # States grezzi per ogni asset
        
        # Storia e tracking
        self.position_history = []   # Storia delle posizioni
        self.action_history = []     # Storia delle azioni
        self.price_history = {ticker: [] for ticker in tickers}  # Storia dei prezzi per ticker
        self.returns_history = {ticker: [] for ticker in tickers}  # Storia dei rendimenti
        self.portfolio_values_history = []  # Storia dei valori del portafoglio
        self.cash_history = []       # Storia del cash disponibile
        
        # Gestione dei dati
        self.dfs = dfs if dfs is not None else {}
        
        # Segnali simulati se non ci sono dati reali
        if not self.dfs:
            for ticker in self.tickers:
                self.signals[ticker] = build_ou_process(T, sigma, theta, random_state)
        else:
            # Verifica che tutti i DataFrame abbiano la stessa lunghezza temporale
            lengths = [len(df) for df in self.dfs.values()]
            if len(set(lengths)) > 1:
                raise ValueError("Tutti i DataFrame devono avere la stessa lunghezza")
            
            # Imposta la lunghezza massima in base ai dati disponibili
            self.T = min(lengths) - 1
        
        # Gestione del rumore
        if noise:
            self.noise_arrays = {}
            for ticker in self.tickers:
                if noise_seed is None:
                    self.noise_arrays[ticker] = np.random.normal(0, noise_std, T)
                else:
                    rng = np.random.RandomState(noise_seed)
                    self.noise_arrays[ticker] = rng.normal(0, noise_std, T)
                    
        # Parametri di normalizzazione
        self.norm_params_paths = norm_params_paths if norm_params_paths is not None else {}
        self.norm_params = {}
        
        for ticker, path in self.norm_params_paths.items():
            if path:
                with open(path, 'r') as f:
                    self.norm_params[ticker] = json.load(f)
        
        # Definizione delle colonne da utilizzare
        if norm_columns is not None:
            self.norm_columns = norm_columns
        else:
            # Esempio con una lista predefinita di feature
            self.norm_columns = [
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
        # Dimensione dello stato per asset e per il portfolio complessivo
        self.state_size_per_asset = len(self.norm_columns)
        self.action_size = self.num_assets  # Un'azione per ogni asset
        self.state_size = (self.state_size_per_asset * self.num_assets) + self.num_assets + 5
        # ^ feature di tutti gli asset + posizioni attuali + 5 metriche di portfolio
        
        # Inizializza raw_state per ogni ticker
        for ticker in self.tickers:
            self.raw_states[ticker] = {col: 0.0 for col in self.norm_columns}
        
        # Flag episodio terminato
        self.done = False
                
    def update_raw_states(self, current_index):
        """
        Aggiorna gli stati grezzi per tutti gli asset utilizzando i DataFrame.
        """
        if not self.dfs:
            return
        
        for ticker in self.tickers:
            if ticker in self.dfs:
                df = self.dfs[ticker]
                if current_index < len(df):
                    # Estrai la riga corrente per questo ticker
                    row = df.iloc[current_index].to_dict()
                    
                    # Verifica che tutte le colonne necessarie siano presenti
                    missing = [col for col in self.norm_columns if col not in row]
                    if missing:
                        raise ValueError(f"Mancano le seguenti colonne nel DataFrame di {ticker}: {missing}")
                    
                    # Aggiorna lo stato grezzo
                    for col in self.norm_columns:
                        self.raw_states[ticker][col] = row[col]
                    
                    # Aggiorna il prezzo corrente per questo asset
                    if "adjClose" in row:
                        self.prices[self.tickers.index(ticker)] = row["adjClose"]
                        self.price_history[ticker].append(row["adjClose"])
                    elif "close" in row:
                        self.prices[self.tickers.index(ticker)] = row["close"]
                        self.price_history[ticker].append(row["close"])
    
    def reset(self, random_state=None, noise_seed=None, start_index=None):
        """
        Resetta l'ambiente per iniziare un nuovo episodio.
        """
        # Reset delle posizioni
        self.positions = np.zeros(self.num_assets)
        
        # Reset del capitale
        self.cash = self.initial_capital
        
        # Reset delle liste di tracciamento
        self.position_history = []
        self.action_history = []
        self.portfolio_values_history = []
        self.cash_history = []
        
        for ticker in self.tickers:
            self.price_history[ticker] = []
            self.returns_history[ticker] = []
        
        # Gestione dei dati simulati o reali
        if not self.dfs:
            # Reset dei segnali simulati
            for ticker in self.tickers:
                self.signals[ticker] = build_ou_process(self.T, self.sigma, self.theta, random_state)
                # Imposta il prezzo iniziale
                ticker_idx = self.tickers.index(ticker)
                self.prices[ticker_idx] = self.signals[ticker][0]
        else:
            # Gestione dell'indice nei dati reali
            if start_index is not None:
                # Se specificato un indice di partenza, usalo (con limite di sicurezza)
                self.current_index = min(start_index, min([len(df) for df in self.dfs.values()]) - self.max_step - 1)
            else:
                # Altrimenti possiamo partire da un punto casuale o dall'inizio
                if random_state is not None:
                    rng = np.random.RandomState(random_state)
                    max_start = max(1, min([len(df) for df in self.dfs.values()]) - self.max_step - 1)
                    self.current_index = rng.randint(0, max_start)
                else:
                    self.current_index = 0
            
            # Aggiorna gli stati grezzi con i dati reali
            self.update_raw_states(self.current_index)
        
        # Reset del noise per i rendimenti simulati
        if self.noise:
            for ticker in self.tickers:
                if noise_seed is None:
                    self.noise_arrays[ticker] = np.random.normal(0, self.noise_std, self.T)
                else:
                    rng = np.random.RandomState(noise_seed)
                    self.noise_arrays[ticker] = rng.normal(0, self.noise_std, self.T)
        
        # Reset variabile episodio terminato
        self.done = False
        
        # Reset contatore operazioni
        self.trade_count = 0
        self.current_month = None
        
        # Calcola il valore iniziale del portafoglio
        portfolio_value = self.get_portfolio_value()
        self.portfolio_values_history.append(portfolio_value)
        self.cash_history.append(self.cash)
        
        # Ritorna lo stato iniziale
        return self.get_state()
    
    def get_portfolio_value(self):
        """
        Calcola il valore totale del portafoglio: cash + valore di tutte le posizioni.
        """
        positions_value = np.sum(self.positions * self.prices)
        return self.cash + positions_value
    
    def calculate_portfolio_metrics(self):
        """
        Calcola metriche di portafoglio come Sharpe ratio, correlazione, diversificazione.
        Restituisce un vettore di metriche aggregate.
        """
        if len(self.portfolio_values_history) < 2:
            # Non abbiamo abbastanza storia per calcolare metriche
            return np.zeros(5)  # 5 metriche portfolio
        
        # 1. Calcola i rendimenti del portafoglio
        portfolio_returns = np.diff(self.portfolio_values_history) / self.portfolio_values_history[:-1]
        
        # 2. Calcola volatilità del portafoglio (annualizzata)
        if len(portfolio_returns) > 1:
            portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)
            
            # 3. Calcola Sharpe o Sortino ratio
            mean_return = np.mean(portfolio_returns)
            excess_return = mean_return - (self.risk_free_rate / 252)  # Daily risk-free rate
            
            if self.use_sortino:
                # Sortino ratio usa solo rendimenti negativi per volatilità
                negative_returns = portfolio_returns[portfolio_returns < self.target_return / 252]
                if len(negative_returns) > 0:
                    downside_deviation = np.std(negative_returns) * np.sqrt(252)
                    sortino_ratio = excess_return * 252 / (downside_deviation + 1e-8)
                else:
                    sortino_ratio = 10.0  # Valore alto per tutti rendimenti positivi
                risk_adjusted_return = sortino_ratio
            else:
                # Sharpe ratio tradizionale
                sharpe_ratio = excess_return * 252 / (portfolio_volatility + 1e-8)
                risk_adjusted_return = sharpe_ratio
        else:
            portfolio_volatility = 0.0
            risk_adjusted_return = 0.0
        
        # 4. Calcola correlazione media tra asset
        correlation = 0.0
        n_pairs = 0
        
        # Calcola le correlazioni solo se abbiamo dati sufficienti
        if all(len(hist) > 5 for hist in self.returns_history.values()):
            returns_data = []
            for ticker in self.tickers:
                if len(self.returns_history[ticker]) > 0:
                    returns_data.append(self.returns_history[ticker])
            
            if len(returns_data) > 1:
                # Taglia tutti gli array alla stessa lunghezza
                min_length = min(len(data) for data in returns_data)
                returns_data = [data[:min_length] for data in returns_data]
                
                # Calcola la correlazione media
                for i in range(len(returns_data)):
                    for j in range(i + 1, len(returns_data)):
                        if len(returns_data[i]) > 1 and len(returns_data[j]) > 1:
                            corr = np.corrcoef(returns_data[i], returns_data[j])[0, 1]
                            if not np.isnan(corr):
                                correlation += abs(corr)  # Usiamo il valore assoluto
                                n_pairs += 1
                
                if n_pairs > 0:
                    correlation /= n_pairs  # Media delle correlazioni
        
        # 5. Calcola la diversificazione (usando l'indice di Herfindahl)
        # Normalizza le posizioni per usare i valori assoluti (ci interessa l'esposizione)
        abs_positions = np.abs(self.positions * self.prices)
        total_exposure = np.sum(abs_positions) + 1e-8  # Evita divisione per zero
        
        # Calcola l'indice di concentrazione (più basso = più diversificato)
        weights = abs_positions / total_exposure
        herfindahl_index = np.sum(weights ** 2)
        
        # Inverti in modo che valori più alti = più diversificato
        diversification = 1.0 - herfindahl_index
        
        # Metriche di portfolio aggregate
        return np.array([
            risk_adjusted_return,  # Sharpe o Sortino ratio
            portfolio_volatility,  # Volatilità annualizzata
            correlation,           # Correlazione media
            diversification,       # Indice di diversificazione
            total_exposure / self.initial_capital  # Esposizione totale / capitale
        ])
    
    def get_state(self):
        """
        Costruisce e restituisce lo stato completo del portafoglio.
        Lo stato è composto da:
        1. Feature normalizzate per ogni asset
        2. Posizioni correnti per ogni asset
        3. Metriche di portafoglio aggregate
        """
        state_components = []
        
        # 1. Estrai le feature normalizzate per ogni asset
        for ticker in self.tickers:
            ticker_features = [self.raw_states[ticker][col] for col in self.norm_columns]
            state_components.extend(ticker_features)
        
        # 2. Aggiungi le posizioni correnti
        state_components.extend(self.positions)
        
        # 3. Aggiungi metriche di portafoglio aggregate
        portfolio_metrics = self.calculate_portfolio_metrics()
        state_components.extend(portfolio_metrics)
        
        return np.array(state_components)
    
    def calculate_correlation_penalty(self):
        """
        Calcola una penalità basata sulla correlazione tra le posizioni e i rendimenti.
        Penalizza posizioni nella stessa direzione per asset altamente correlati.
        """
        if self.correlation_penalty_factor == 0:
            return 0
        
        if all(len(hist) > 5 for hist in self.returns_history.values()):
            penalty = 0
            n_pairs = 0
            
            returns_data = []
            for ticker in self.tickers:
                if len(self.returns_history[ticker]) > 0:
                    returns_data.append((ticker, self.returns_history[ticker]))
            
            if len(returns_data) > 1:
                # Calcola penalità per asset correlati nella stessa direzione
                for i in range(len(returns_data)):
                    for j in range(i + 1, len(returns_data)):
                        ticker_i, returns_i = returns_data[i]
                        ticker_j, returns_j = returns_data[j]
                        
                        if len(returns_i) > 1 and len(returns_j) > 1:
                            min_length = min(len(returns_i), len(returns_j))
                            corr = np.corrcoef(returns_i[:min_length], returns_j[:min_length])[0, 1]
                            
                            if not np.isnan(corr):
                                # Indici dei ticker
                                idx_i = self.tickers.index(ticker_i)
                                idx_j = self.tickers.index(ticker_j)
                                
                                # Posizioni attuali
                                pos_i = self.positions[idx_i]
                                pos_j = self.positions[idx_j]
                                
                                # Penalizza posizioni nella stessa direzione per asset correlati positivamente
                                # Penalizza posizioni in direzioni opposte per asset correlati negativamente
                                same_direction = (pos_i * pos_j > 0)
                                if (corr > 0 and same_direction) or (corr < 0 and not same_direction):
                                    penalty += abs(corr) * abs(pos_i) * abs(pos_j)
                                
                                n_pairs += 1
                
                if n_pairs > 0:
                    return penalty * self.correlation_penalty_factor / n_pairs
        
        return 0
    
    def calculate_diversification_bonus(self):
        """
        Calcola un bonus per un portafoglio ben diversificato.
        """
        if self.diversification_bonus_factor == 0:
            return 0
        
        # Normalizza le posizioni (usa valori assoluti per considerare esposizione)
        abs_positions = np.abs(self.positions * self.prices)
        total_exposure = np.sum(abs_positions) + 1e-8  # Evita divisione per zero
        
        # Calcola la diversificazione (1 - indice Herfindahl)
        if total_exposure > 0:
            weights = abs_positions / total_exposure
            herfindahl_index = np.sum(weights ** 2)
            diversification = 1.0 - herfindahl_index
            
            # Scala in base al fattore di diversificazione
            return diversification * self.diversification_bonus_factor
            
        return 0
    
    def calculate_risk(self):
        """
        Calcola una metrica di rischio basata sulla volatilità del portafoglio.
        """
        if len(self.portfolio_values_history) < 5:
            return 0
        
        # Calcola i rendimenti
        returns = np.diff(self.portfolio_values_history) / self.portfolio_values_history[:-1]
        
        if len(returns) < 2:
            return 0
            
        # Calcola la volatilità annualizzata
        volatility = np.std(returns) * np.sqrt(252)
        
        # Più alta la volatilità, maggiore il rischio
        return volatility * np.sum(np.abs(self.positions))
    
    def calculate_trading_costs(self, actions, current_prices):
        """
        Calcola i costi di trading per un insieme di azioni.
        """
        total_cost = 0
        
        for i, action in enumerate(actions):
            if abs(action) > 1e-6:  # Se c'è un'operazione effettiva
                # Calcola il valore dell'ordine
                order_value = abs(action) * current_prices[i]
                
                # Calcola commissione come percentuale dell'importo o minimo fisso
                if self.trade_count < self.free_trades_per_month:
                    # Operazione gratuita
                    trading_cost = 0
                    self.trade_count += 1
                else:
                    percentage_commission = order_value * self.commission_rate
                    trading_cost = max(percentage_commission, self.min_commission)
                
                total_cost += trading_cost
        
        return total_cost
    
    def step(self, actions):
        """
        Esegue un passo nell'ambiente applicando le azioni fornite e restituendo
        la ricompensa e il nuovo stato.
        
        Parametri:
        - actions: vettore numpy di azioni per ogni asset
        
        Ritorna:
        - float: ricompensa ottenuta
        """
        assert not self.done, "L'episodio è terminato. Resetta l'ambiente prima di procedere."
        assert len(actions) == self.num_assets, f"Numero di azioni non valido. Atteso: {self.num_assets}, ricevuto: {len(actions)}"
        
        # Salva posizioni e prezzi correnti
        positions_prev = self.positions.copy()
        current_prices = self.prices.copy()
        
        # Traccia lo storico
        self.position_history.append(positions_prev.copy())
        self.action_history.append(actions.copy())
        
        # Calcola nuove posizioni dopo le azioni
        positions_next_unclipped = positions_prev + actions
        
        # Limita le posizioni per ciascun asset se 'clip' è attivato
        if self.clip:
            positions_next = np.clip(
                positions_next_unclipped, 
                -self.max_pos_per_asset, 
                self.max_pos_per_asset
            )
            
            # Gestisce anche il limite di esposizione totale del portafoglio se necessario
            total_long = np.sum(np.maximum(positions_next, 0))
            total_short = np.sum(np.minimum(positions_next, 0))
            
            if total_long > self.max_portfolio_pos:
                # Scala proporzionalmente le posizioni long
                scale_factor = self.max_portfolio_pos / total_long
                long_positions = np.maximum(positions_next, 0)
                positions_next = np.where(positions_next > 0, 
                                          long_positions * scale_factor, 
                                          positions_next)
            
            if abs(total_short) > self.max_portfolio_pos:
                # Scala proporzionalmente le posizioni short
                scale_factor = self.max_portfolio_pos / abs(total_short)
                short_positions = np.minimum(positions_next, 0)
                positions_next = np.where(positions_next < 0, 
                                          short_positions * scale_factor, 
                                          positions_next)
        else:
            positions_next = positions_next_unclipped
        
        # Calcola penalità per posizioni che violano i limiti
        if self.penalty == "none":
            pen = 0
        elif self.penalty == "tanh":
            # Penalità per violazione dei limiti di asset singoli
            asset_penalties = self.beta * (np.tanh(self.alpha * (np.abs(positions_next_unclipped) - 5 * self.max_pos_per_asset / 4)) + 1)
            
            # Penalità per violazione del limite di portafoglio
            total_long = np.sum(np.maximum(positions_next_unclipped, 0))
            total_short = abs(np.sum(np.minimum(positions_next_unclipped, 0)))
            
            portfolio_long_penalty = self.beta * (np.tanh(self.alpha * (total_long - 5 * self.max_portfolio_pos / 4)) + 1)
            portfolio_short_penalty = self.beta * (np.tanh(self.alpha * (total_short - 5 * self.max_portfolio_pos / 4)) + 1)
            
            pen = np.sum(asset_penalties) + portfolio_long_penalty + portfolio_short_penalty
        elif self.penalty == "exp":
            # Versione con penalità esponenziale
            asset_penalties = self.beta * np.exp(self.alpha * (np.abs(positions_next_unclipped) - self.max_pos_per_asset))
            total_exposure = np.sum(np.abs(positions_next_unclipped))
            portfolio_penalty = self.beta * np.exp(self.alpha * (total_exposure - 2 * self.max_portfolio_pos))
            
            pen = np.sum(asset_penalties) + portfolio_penalty
        else:
            pen = 0
            
        # Calcola penalità per trading frequente
        trading_frequency_penalty = 0
        if self.trading_frequency_penalty_factor > 0:
            # Considera solo le ultime n azioni
            history_window = 5
            if len(self.action_history) >= history_window:
                recent_actions = self.action_history[-history_window:]
                # Calcola quante azioni significative ci sono state per ciascun asset
                significant_actions = np.zeros(self.num_assets)
                
                for past_actions in recent_actions:
                    significant_actions += (np.abs(past_actions) > 1e-6)
                
                # Media delle azioni significative per asset
                trading_frequency = np.mean(significant_actions) / history_window
                trading_frequency_penalty = self.trading_frequency_penalty_factor * trading_frequency
                
        # Calcola bonus per stabilità di posizione
        stability_bonus = 0
        if self.position_stability_bonus_factor > 0:
            # Bonus per azioni piccole (conservazione delle posizioni)
            small_actions_ratio = np.mean(np.abs(actions) < 0.05)
            stability_bonus = self.position_stability_bonus_factor * small_actions_ratio
            
        # Calcola bonus/penalità per diversificazione e correlazione
        correlation_penalty = self.calculate_correlation_penalty()
        diversification_bonus = self.calculate_diversification_bonus()
        
        # Aggiorna le posizioni
        self.positions = positions_next
        
        # Gestione del passo temporale e calcolo della ricompensa
        if self.dfs:  # Dati reali
            # Avanza al prossimo timestep nei DataFrame
            self.current_index += 1
            
            # Controlla se siamo arrivati alla fine del dataset
            if self.current_index >= min(len(df) for df in self.dfs.values()) - 1:
                self.done = True
                # Ricompensa finale (potrebbe essere personalizzata in base alla performance complessiva)
                return 0
            
            # Aggiorna gli stati grezzi con i nuovi dati
            self.update_raw_states(self.current_index)
            
            # Gestione del cambio mese e conteggio delle operazioni
            if all('date' in df.columns for df in self.dfs.values()):
                # Prendi la data dal primo DataFrame (assumiamo che siano allineati)
                current_date_str = list(self.dfs.values())[0]['date'].iloc[self.current_index]
                
                # Converti in datetime se necessario
                if isinstance(current_date_str, str):
                    from datetime import datetime
                    current_date = datetime.strptime(current_date_str, '%Y-%m-%d')
                else:
                    # Assume che sia già un datetime
                    current_date = current_date_str
                
                current_month = (current_date.year, current_date.month)
                
                # Reset contatore operazioni se cambia il mese
                if current_month != self.current_month:
                    self.current_month = current_month
                    self.trade_count = 0
            
            # Calcola i rendimenti per ogni asset
            pnl_per_asset = np.zeros(self.num_assets)
            price_returns = np.zeros(self.num_assets)
            
            for i, ticker in enumerate(self.tickers):
                prev_price = current_prices[i]
                current_price = self.prices[i]
                
                if prev_price != 0:
                    # Calcola il rendimento percentuale
                    price_return = (current_price / prev_price) - 1
                    price_returns[i] = price_return
                    
                    # Aggiungi alla storia dei rendimenti
                    self.returns_history[ticker].append(price_return)
                    
                    # Calcola P&L per questa posizione
                    pnl_per_asset[i] = positions_prev[i] * price_return * prev_price
                
            # Calcola il P&L totale del portafoglio
            total_pnl = np.sum(pnl_per_asset)
            
            # Calcola i costi di trading
            trading_costs = self.calculate_trading_costs(actions, current_prices)
            
            # Aggiorna il cash disponibile
            self.cash += total_pnl - trading_costs
            
            # Aggiorna il valore totale del portafoglio
            portfolio_value = self.get_portfolio_value()
            self.portfolio_values_history.append(portfolio_value)
            self.cash_history.append(self.cash)
            
            # Calcola penalità per dimensione della posizione (controllo del rischio)
            position_risk_penalty = self.lambd * np.sum(self.positions ** 2) * self.squared_risk
            
            # Calcola la penalità di rischio basata sulla volatilità
            risk_penalty = self.calculate_risk()
            
            # Calcola la ricompensa complessiva come combinazione di tutti i fattori
            reward = (total_pnl - trading_costs - position_risk_penalty - pen - 
                      risk_penalty - trading_frequency_penalty - correlation_penalty + 
                      stability_bonus + diversification_bonus) / self.scale_reward
                      
        else:  # Dati simulati con processo OU
            # Avanza al prossimo timestep per ogni segnale simulato
            for i, ticker in enumerate(self.tickers):
                # Salva prezzo corrente
                prev_price = self.prices[i]
                
                # Avanza il segnale
                if i == 0:  # Incrementa l'indice solo una volta
                    self.it += 1
                
                # Ottieni il nuovo prezzo dal segnale (con eventuale rumore)
                if self.noise:
                    self.prices[i] = self.signals[ticker][self.it + 1] + self.noise_arrays[ticker][self.it]
                else:
                    self.prices[i] = self.signals[ticker][self.it + 1]
                
                # Calcola il rendimento
                if prev_price != 0:
                    price_return = (self.prices[i] / prev_price) - 1
                    # Aggiungi alla storia dei rendimenti
                    self.returns_history[ticker].append(price_return)
                
                # Calcola P&L per questa posizione
                pnl_per_asset = positions_prev * (self.prices - current_prices)
            
            # Flag di fine episodio
            self.done = self.it >= (len(self.signals[self.tickers[0]]) - 2)
            
            # Calcola il P&L totale
            total_pnl = np.sum(pnl_per_asset)
            
            # Calcola i costi di trading in base al tipo di costo specificato
            if self.cost == "trade_0":
                trading_costs = 0
            elif self.cost == "trade_l1":
                trading_costs = self.psi * np.sum(np.abs(positions_next - positions_prev))
            elif self.cost == "trade_l2":
                trading_costs = self.psi * np.sum((positions_next - positions_prev) ** 2)
            
            # Calcola la penalità per dimensione della posizione
            position_risk_penalty = self.lambd * np.sum(positions_next ** 2) * self.squared_risk
            
            # Aggiorna il cash
            self.cash += total_pnl - trading_costs
            
            # Aggiorna il valore del portafoglio
            portfolio_value = self.get_portfolio_value()
            self.portfolio_values_history.append(portfolio_value)
            self.cash_history.append(self.cash)
            
            # Calcola la ricompensa
            reward = (total_pnl - trading_costs - position_risk_penalty - pen - 
                     correlation_penalty + diversification_bonus) / self.scale_reward
        
        return reward
        
    def test(self, agent, model, total_episodes=100, random_states=None, noise_seeds=None):
        """
        Testa un modello su un numero di episodi simulati e restituisce la ricompensa media.
        
        Parametri:
        - agent: oggetto Agent che carica il modello
        - model: oggetto Actor, la rete dell'attore
        - total_episodes: numero di episodi di test
        - random_states: None o lista di lunghezza total_episodes per generare episodi riproducibili
        - noise_seeds: None o lista di semi per il rumore
        
        Ritorna:
        - media delle ricompense cumulative, dict dei punteggi per episodio, ecc.
        """
        scores = {}
        scores_cumsum = {}
        pnls = {}
        positions = {}
        
        agent.actor_local = model
        
        if random_states is not None:
            assert total_episodes == len(random_states), "random_states deve essere una lista di lunghezza total_episodes!"
        
        cumulative_rewards = []
        cumulative_pnls = []
        portfolio_values = []
        
        for episode in range(total_episodes):
            episode_rewards = []
            episode_pnls = []
            episode_positions = [np.zeros(self.num_assets)]  # Inizia con posizioni zero
            
            random_state = None if random_states is None else random_states[episode]
            noise_seed = None if noise_seeds is None else noise_seeds[episode]
            
            self.reset(random_state, noise_seed)
            state = self.get_state()
            done = self.done
            
            while not done:
                # Ottieni l'azione dall'agente senza rumore di esplorazione
                action = agent.act(state, noise=False)
                
                # Applica l'azione e ottieni ricompensa e nuovo stato
                reward = self.step(action)
                state = self.get_state()
                done = self.done
                
                # Registra ricompensa, P&L e posizioni
                episode_rewards.append(reward)
                
                # Calcola il P&L semplificato (senza penalità)
                pnl = reward + (self.lambd * np.sum(self.positions ** 2)) * self.squared_risk
                episode_pnls.append(pnl)
                
                # Salva le posizioni correnti
                episode_positions.append(self.positions.copy())
                
                if done:
                    # Calcola metriche per l'episodio completo
                    total_reward = np.sum(episode_rewards)
                    total_pnl = np.sum(episode_pnls)
                    final_portfolio_value = self.get_portfolio_value()
                    
                    if random_states is not None:
                        scores[random_states[episode]] = total_reward
                        scores_cumsum[random_states[episode]] = np.cumsum(episode_rewards)
                        pnls[random_states[episode]] = total_pnl
                        positions[random_states[episode]] = episode_positions
                    
                    cumulative_rewards.append(total_reward)
                    cumulative_pnls.append(total_pnl)
                    portfolio_values.append(final_portfolio_value)
        
        # Calcola le medie
        avg_reward = np.mean(cumulative_rewards)
        avg_pnl = np.mean(cumulative_pnls)
        avg_portfolio_value = np.mean(portfolio_values)
        
        return (
            avg_reward,
            scores,
            scores_cumsum,
            avg_pnl,
            positions,
            avg_portfolio_value
        )
        
    def denormalize_price(self, ticker, normalized_price, price_feature="Log_Close"):
        """
        Denormalizza un prezzo per un ticker specifico.
        
        Parametri:
        - ticker: il ticker dell'asset
        - normalized_price: il prezzo normalizzato
        - price_feature: la feature del prezzo da denormalizzare
        
        Ritorna:
        - float: prezzo denormalizzato
        """
        if ticker not in self.norm_params:
            return normalized_price
        
        norm_params = self.norm_params[ticker]
        
        if price_feature not in norm_params['min'] or price_feature not in norm_params['max']:
            return normalized_price
        
        min_val = norm_params['min'][price_feature]
        max_val = norm_params['max'][price_feature]
        
        # Applica la formula inversa della normalizzazione min-max
        denorm_price = normalized_price * (max_val - min_val) + min_val
        
        return denorm_price
    
    def get_real_portfolio_metrics(self, price_feature="Log_Close"):
        """
        Calcola metriche di portafoglio con prezzi denormalizzati.
        
        Parametri:
        - price_feature: la feature del prezzo da usare
        
        Ritorna:
        - dict: dizionario con metriche di performance reali
        """
        if len(self.portfolio_values_history) < 2:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'volatility': 0,
                'final_portfolio_value': self.get_portfolio_value()
            }
        
        # Calcola rendimenti e metriche
        portfolio_returns = np.diff(self.portfolio_values_history) / self.portfolio_values_history[:-1]
        
        # Rendimento totale
        initial_value = self.portfolio_values_history[0]
        final_value = self.portfolio_values_history[-1]
        total_return = (final_value / initial_value) - 1
        
        # Volatilità annualizzata
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        # Sharpe ratio
        mean_daily_return = np.mean(portfolio_returns)
        sharpe_ratio = (mean_daily_return - (self.risk_free_rate / 252)) / (volatility / np.sqrt(252)) if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns / running_max) - 1
        max_drawdown = np.min(drawdowns)
        
        return {
            'total_return': total_return * 100,  # in percentuale
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,  # in percentuale
            'volatility': volatility * 100,  # in percentuale
            'final_portfolio_value': final_value
        }