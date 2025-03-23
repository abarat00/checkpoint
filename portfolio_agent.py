import os
from time import sleep
from collections import deque, namedtuple

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display

# Import dei moduli locali
from memory import Memory, PrioritizedMemory, Node
from portfolio_models import PortfolioActor, PortfolioCritic, EnhancedPortfolioActor, AssetEncoder

# Definizione di un namedtuple per le transizioni
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "dones"))

# Parametri globali
GAMMA = 0.99                 # discount factor
TAU_ACTOR = 1e-1             # soft update parameter for the actor target network
TAU_CRITIC = 1e-3            # soft update parameter for the critic target network
LR_ACTOR = 1e-4              # learning rate for the actor network
LR_CRITIC = 1e-3             # learning rate for the critic network
WEIGHT_DECAY_ACTOR = 0       # L2 weight decay for actor
WEIGHT_DECAY_CRITIC = 1e-2   # L2 weight decay for critic
BATCH_SIZE = 128             # minibatch size (aumentato per multi-asset)
BUFFER_SIZE = int(1e6)       # replay buffer size
PRETRAIN = 256               # number of pretraining steps (should be > BATCH_SIZE)
MAX_STEP = 100               # number of steps in an episode
WEIGHTS = "portfolio_weights/"  # path where to save model weights

# Dimensioni dei layer nelle reti
FC1_UNITS_ACTOR = 256        # actor: nodes in first hidden layer
FC2_UNITS_ACTOR = 128        # actor: nodes in second hidden layer
FC3_UNITS_ACTOR = 64         # actor: nodes in third hidden layer

FC1_UNITS_CRITIC = 512       # critic: nodes in first hidden layer
FC2_UNITS_CRITIC = 256       # critic: nodes in second hidden layer
FC3_UNITS_CRITIC = 128       # critic: nodes in third hidden layer

DECAY_RATE = 1e-6            # decay rate for exploration noise
EXPLORE_STOP = 0.1           # final exploration probability

class MultiAssetOUNoise:
    """
    Ornstein-Uhlenbeck process for generating temporally correlated exploration noise
    for multiple assets simultaneously.
    """
    def __init__(self, action_size, mu=0.0, theta=0.1, sigma=0.2):
        """
        Inizializza il processo OU per rumore multi-dimensionale.
        
        Parametri:
        - action_size: numero di asset (dimensione dello spazio azioni)
        - mu: valore medio del processo
        - theta: velocità di mean reversion
        - sigma: volatilità del processo
        """
        self.action_size = action_size
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """
        Resetta lo stato del processo a mu.
        """
        self.state = np.copy(self.mu)

    def sample(self, truncate=False, max_pos=2.0, positions=None, actions=None):
        """
        Genera un campione di rumore per ogni dimensione dell'azione.
        
        Parametri:
        - truncate: se True, tronca il rumore per mantenere le posizioni entro i limiti
        - max_pos: posizione massima consentita
        - positions: vettore delle posizioni attuali
        - actions: vettore delle azioni non perturbate
        
        Ritorna:
        - Vettore di rumore per ogni dimensione dell'azione
        """
        x = self.state
        
        if truncate:
            assert positions is not None, "positions required when truncate=True"
            assert actions is not None, "actions required when truncate=True"
            
            from scipy.stats import truncnorm
            noise = np.zeros(self.action_size)
            
            for i in range(self.action_size):
                # Calcola i limiti per il rumore in modo da rispettare i vincoli di posizione
                m = -max_pos - positions[i] - actions[i] - (1 - self.theta) * x[i]
                M = max_pos - positions[i] - actions[i] - (1 - self.theta) * x[i]
                
                # Normalizza i limiti
                x_a, x_b = m / self.sigma, M / self.sigma
                
                # Usa una normale troncata
                X = truncnorm(x_a, x_b, scale=self.sigma)
                
                # Campiona e aggiorna lo stato
                dx = self.theta * (self.mu[i] - x[i]) + X.rvs()
                self.state[i] = x[i] + dx
                noise[i] = self.state[i]
        else:
            # Versione standard del processo OU
            dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(size=self.action_size)
            self.state = x + dx
            noise = self.state
            
        return noise

class PortfolioAgent:
    def __init__(
        self,
        num_assets,               # Numero di asset nel portafoglio
        gamma=GAMMA,
        max_size=BUFFER_SIZE,
        max_step=MAX_STEP,
        memory_type="prioritized",
        alpha=0.6,
        beta0=0.4,
        epsilon=1e-8,
        sliding="oldest",
        batch_size=BATCH_SIZE,
        theta=0.1,
        sigma=0.2,
        use_enhanced_actor=False, # Se utilizzare l'actor avanzato con attenzione
        use_batch_norm=True       # Se utilizzare batch normalization
    ):
        """
        Costruttore dell'agente DDPG adattato per portafoglio multi-asset.

        Parametri:
          - num_assets: numero di asset nel portafoglio
          - gamma: fattore di sconto
          - max_size: dimensione massima del buffer di replay
          - max_step: numero di step per episodio
          - memory_type: 'uniform' o 'prioritized'
          - alpha, beta0, epsilon: parametri per replay prioritizzato
          - sliding: strategia di sostituzione in replay prioritizzato
          - batch_size: dimensione del batch per addestramento
          - theta, sigma: parametri per il rumore OU
          - use_enhanced_actor: se usare l'actor con meccanismi di attenzione
          - use_batch_norm: se usare batch normalization nelle reti
        """
        assert 0 <= gamma <= 1, "Gamma must be in [0,1]"
        assert memory_type in ["uniform", "prioritized"], "Invalid memory type"
        
        self.num_assets = num_assets
        self.gamma = gamma
        self.max_size = max_size
        self.memory_type = memory_type
        self.epsilon = epsilon
        self.use_enhanced_actor = use_enhanced_actor
        self.use_batch_norm = use_batch_norm

        # Inizializza la memoria appropriata
        if memory_type == "uniform":
            self.memory = Memory(max_size=max_size)
        elif memory_type == "prioritized":
            self.memory = PrioritizedMemory(max_size=max_size, sliding=sliding)

        self.max_step = max_step
        self.alpha = alpha
        self.beta0 = beta0
        self.batch_size = batch_size
        
        # Inizializza il processo OU multi-dimensionale per l'esplorazione
        self.noise = MultiAssetOUNoise(num_assets, theta=theta, sigma=sigma)

        # Inizializza le reti (saranno create durante l'addestramento)
        self.actor_local = None
        self.actor_target = None
        self.critic_local = None
        self.critic_target = None

    def reset(self):
        """
        Resetta il rumore di esplorazione.
        """
        self.noise.reset()
    # Aggiungi un metodo per salvare lo stato completo dell'agente
    # Aggiungi questa funzione alla classe PortfolioAgent
    def save_checkpoint(self, file_path, episode, iteration, metrics=None):
        """
        Salva un checkpoint con lo stato attuale dell'addestramento.
        """
        print(f"Preparazione checkpoint per episodio {episode}...")
        
        # Prepara le metriche
        checkpoint_metrics = {}
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, deque):
                    checkpoint_metrics[key] = list(value)
                else:
                    checkpoint_metrics[key] = value
        
        # Crea il checkpoint
        checkpoint = {
            'episode': episode,
            'iteration': iteration,
            'actor_state_dict': self.actor_local.state_dict(),
            'critic_state_dict': self.critic_local.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'metrics': checkpoint_metrics
        }
        
        # Salva il checkpoint
        torch.save(checkpoint, file_path)
        print(f"Checkpoint salvato: {file_path} (episodio {episode})")
# Aggiungi un metodo per caricare da checkpoint
    def load_checkpoint(self, path):
        """
        Carica un checkpoint completo dell'agente.
        
        Parametri:
        - path: percorso del checkpoint da caricare
        
        Ritorna:
        - episode: numero dell'episodio da cui riprendere
        - results: dizionario dei risultati fino a questo punto
        """
        checkpoint = torch.load(path, weights_only=False)
        
        # Carica stato dei modelli
        if self.actor_local:
            self.actor_local.load_state_dict(checkpoint['actor_state_dict'])
        if self.critic_local:
            self.critic_local.load_state_dict(checkpoint['critic_state_dict'])
        if self.actor_target and checkpoint['actor_target_state_dict']:
            self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        if self.critic_target and checkpoint['critic_target_state_dict']:
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        
        return checkpoint['episode'], checkpoint['results']
    def step(self, state, action, reward, next_state, done, pretrain=False):
        """
        Salva una transizione (state, action, reward, next_state, done) nel buffer di replay.
        
        Parametri:
        - state: stato corrente
        - action: azione eseguita (vettore di azioni per ogni asset)
        - reward: ricompensa ottenuta
        - next_state: stato successivo
        - done: flag di fine episodio
        - pretrain: se True, usa la ricompensa per calcolare la priorità iniziale
        """
        # Converti tutto in tensori
        state_mb = torch.tensor([state], dtype=torch.float)
        action_mb = torch.tensor([action], dtype=torch.float)  # Ora action è un vettore
        reward_mb = torch.tensor([[reward]], dtype=torch.float)
        next_state_mb = torch.tensor([next_state], dtype=torch.float)
        not_done_mb = torch.tensor([[not done]], dtype=torch.float)

        # Aggiungi alla memoria
        if self.memory_type == "uniform":
            self.memory.add((state_mb, action_mb, reward_mb, next_state_mb, not_done_mb))
        elif self.memory_type == "prioritized":
            # Durante il pretraining, usa l'ampiezza della ricompensa come priorità
            # Altrimenti usa la priorità massima corrente
            priority = (abs(reward) + self.epsilon) ** self.alpha if pretrain else self.memory.highest_priority()
            self.memory.add((state_mb, action_mb, reward_mb, next_state_mb, not_done_mb), priority)

    def act(self, state, noise=True, explore_probability=1.0, truncate=False, max_pos=2.0):
        """
        Restituisce un'azione per uno stato dato usando la rete actor, con rumore opzionale.
        
        Parametri:
        - state: stato corrente (array NumPy di dimensione state_size)
        - noise: se aggiungere rumore di esplorazione
        - explore_probability: fattore di scala per il rumore
        - truncate: se troncare il rumore in modo che le posizioni risultanti restino entro i limiti
        - max_pos: posizione massima consentita per asset
        
        Ritorna:
        - action: vettore di azioni, una per ogni asset
        """
        # Estrai le posizioni attuali (ultimi num_assets elementi dello stato)
        positions = state[-self.num_assets:]
        
        # Converti lo stato in tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        
        # Passa lo stato attraverso la rete Actor
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(state_tensor).data.numpy()[0]  # [0] per eliminare batch dim
        self.actor_local.train()
        
        # Aggiungi rumore per l'esplorazione se richiesto
        if noise:
            noise_sample = self.noise.sample(
                truncate=truncate,
                max_pos=max_pos,
                positions=positions,
                actions=actions
            )
            actions += explore_probability * noise_sample
        
        return actions

    def soft_update(self, local_model, target_model, tau):
        """
        Aggiorna soft i parametri del modello: target = tau*local + (1-tau)*target.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def pretrain(self, env, total_steps=PRETRAIN):
        """
        Pre-addestra l'agente riempiendo il buffer di replay.
        """
        env.reset()
        
        with torch.no_grad():
            for i in range(total_steps):
                state = env.get_state()
                
                # Genera azioni casuali ma ragionevoli durante il pretraining
                if self.actor_local is None:
                    # Se l'actor non è stato ancora inizializzato, usa azioni casuali
                    actions = np.random.uniform(-0.1, 0.1, self.num_assets)
                else:
                    # Altrimenti, usa l'actor con alto rumore di esplorazione
                    actions = self.act(
                        state, 
                        truncate=(not env.squared_risk), 
                        max_pos=env.max_pos_per_asset, 
                        explore_probability=2.0  # Aumenta l'esplorazione
                    )
                
                # Esegui l'azione e ottieni ricompensa e nuovo stato
                reward = env.step(actions)
                next_state = env.get_state()
                done = env.done
                
                # Salva l'esperienza
                self.step(state, actions, reward, next_state, done, pretrain=True)
                
                # Resetta l'ambiente se necessario
                if done:
                    env.reset()

    def train(
        self,
        env,
        total_episodes=100,
        tau_actor=TAU_ACTOR,
        tau_critic=TAU_CRITIC,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        weight_decay_actor=WEIGHT_DECAY_ACTOR,
        weight_decay_critic=WEIGHT_DECAY_CRITIC,
        total_steps=PRETRAIN,
        weights=WEIGHTS,
        freq=10,
        fc1_units_actor=FC1_UNITS_ACTOR,
        fc2_units_actor=FC2_UNITS_ACTOR,
        fc3_units_actor=FC3_UNITS_ACTOR,
        fc1_units_critic=FC1_UNITS_CRITIC,
        fc2_units_critic=FC2_UNITS_CRITIC,
        fc3_units_critic=FC3_UNITS_CRITIC,
        decay_rate=DECAY_RATE,
        explore_stop=EXPLORE_STOP,
        tensordir="runs/portfolio/",
        learn_freq=5,
        plots=False,
        progress="tqdm",
        features_per_asset=0,  # Per EnhancedPortfolioActor
        encoding_size=0,       # Per EnhancedPortfolioActor
        clip_grad_norm=1.0,     # Limite per gradient clipping
        checkpoint_path=None,
        checkpoint_freq=10,     # Frequenza di salvataggio checkpoint (ogni N episodi)
        resume_from=None       # Checkpoint da cui riprendere
    ):
        """
        Addestra l'agente per un certo numero di episodi.
        """
        # Variabile per tracciare l'ultimo checkpoint salvato
        self.last_checkpoint_episode = -1

        # Crea la directory per i pesi se non esiste
        if not os.path.isdir(weights):
            os.makedirs(weights, exist_ok=True)

        # Nel corpo della funzione, aggiungi:
        if checkpoint_path is None:
            checkpoint_path = weights

        # Inizializza TensorBoard
        writer = SummaryWriter(log_dir=tensordir)

        # ====== FASE 1: Analisi del checkpoint per recuperare le dimensioni ======
        # Invece di inizializzare subito i modelli, prima analizziamo il checkpoint
        checkpoint = None
        asset_encoder_dim = None
        
        if resume_from and os.path.exists(resume_from):
            print(f"Analisi del checkpoint {resume_from} per determinare le dimensioni corrette...")
            checkpoint = torch.load(resume_from, weights_only=False)
            
            # Estrai dimensioni dai nomi dei parametri per garantire compatibilità
            if 'actor_state_dict' in checkpoint:
                actor_dict = checkpoint['actor_state_dict']
                
                # Estrai dimensione dell'encoder
                if 'asset_encoder.fc2.weight' in actor_dict:
                    encoder_shape = actor_dict['asset_encoder.fc2.weight'].shape
                    if len(encoder_shape) >= 1:
                        asset_encoder_dim = encoder_shape[0]
                        print(f"Dimensione dell'encoder degli asset: {asset_encoder_dim}")
                
                # Estrai dimensione del layer di attenzione
                if 'attention.weight' in actor_dict:
                    attention_shape = actor_dict['attention.weight'].shape
                    if len(attention_shape) > 1:
                        encoding_size = attention_shape[1]
                        print(f"Adattamento encoding_size a {encoding_size} dal checkpoint")

        # Dopo aver analizzato il checkpoint
        attention_dim = None
        encoder_output_size = None

        if 'attention.weight' in actor_dict:
            attention_shape = actor_dict['attention.weight'].shape
            if len(attention_shape) > 1:
                attention_dim = attention_shape[1]
                print(f"Dimensione attention: {attention_dim}")

        if 'asset_encoder.fc2.weight' in actor_dict:
            encoder_shape = actor_dict['asset_encoder.fc2.weight'].shape
            if len(encoder_shape) >= 1:
                encoder_output_size = encoder_shape[0]
                print(f"Dimensione encoder output: {encoder_output_size}")

        # Usa entrambe le dimensioni
        self.actor_local = EnhancedPortfolioActor(
            env.state_size, 
            self.num_assets, 
            features_per_asset,
            fc1_units=fc1_units_actor,
            fc2_units=fc2_units_actor,
            encoding_size=encoding_size,
            use_attention=True,
            attention_size=attention_dim,
            encoder_output_size=encoder_output_size
        )

        # ====== FASE 2: Inizializzazione dei modelli con le dimensioni corrette ======
        if self.use_enhanced_actor and features_per_asset > 0:
            # Assicurati che encoding_size abbia un valore sensato
            if encoding_size == 0:
                encoding_size = 16  # Default
            
            # Se abbiamo ottenuto la dimensione dell'encoder dal checkpoint, usala
            if asset_encoder_dim is not None and asset_encoder_dim != encoding_size:
                print(f"ATTENZIONE: Dimensioni non coincidenti - asset_encoder: {asset_encoder_dim}, attention: {encoding_size}")
                print(f"Utilizzo di {encoding_size} per garantire compatibilità")
                # NON sovrascrivere encoding_size
            
            print(f"Inizializzazione EnhancedPortfolioActor con encoding_size={encoding_size}")
            
            # Inizializza l'asset encoder prima, così possiamo passarlo agli actor
            asset_encoder = AssetEncoder(
                features_per_asset=features_per_asset, 
                encoding_size=encoding_size,
                seed=0
            )
            
            # Versione avanzata dell'Actor con meccanismi di attenzione
            self.actor_local = EnhancedPortfolioActor(
                env.state_size, 
                self.num_assets, 
                features_per_asset,
                fc1_units=fc1_units_actor,
                fc2_units=fc2_units_actor,
                encoding_size=encoding_size,
                use_attention=True
            )
            self.actor_target = EnhancedPortfolioActor(
                env.state_size, 
                self.num_assets, 
                features_per_asset,
                fc1_units=fc1_units_actor,
                fc2_units=fc2_units_actor,
                encoding_size=encoding_size,
                use_attention=True
            )
        else:
            # Versione standard dell'Actor
            self.actor_local = PortfolioActor(
                env.state_size, 
                self.num_assets, 
                fc1_units=fc1_units_actor, 
                fc2_units=fc2_units_actor,
                fc3_units=fc3_units_actor,
                use_batch_norm=self.use_batch_norm
            )
            self.actor_target = PortfolioActor(
                env.state_size, 
                self.num_assets, 
                fc1_units=fc1_units_actor, 
                fc2_units=fc2_units_actor,
                fc3_units=fc3_units_actor,
                use_batch_norm=self.use_batch_norm
            )

        # Inizializza le reti Critic
        self.critic_local = PortfolioCritic(
            env.state_size, 
            self.num_assets, 
            fcs1_units=fc1_units_critic, 
            fc2_units=fc2_units_critic,
            fc3_units=fc3_units_critic,
            use_batch_norm=self.use_batch_norm
        )
        self.critic_target = PortfolioCritic(
            env.state_size, 
            self.num_assets, 
            fcs1_units=fc1_units_critic, 
            fc2_units=fc2_units_critic,
            fc3_units=fc3_units_critic,
            use_batch_norm=self.use_batch_norm
        )
        
        # Ottimizzatore per Actor
        actor_optimizer = optim.Adam(
            self.actor_local.parameters(), 
            lr=lr_actor, 
            weight_decay=weight_decay_actor
        )
        # Scheduler per learning rate
        actor_lr_scheduler = lr_scheduler.StepLR(
            actor_optimizer, 
            step_size=100, 
            gamma=0.5
        )
        
        # Ottimizzatore per Critic
        critic_optimizer = optim.Adam(
            self.critic_local.parameters(), 
            lr=lr_critic, 
            weight_decay=weight_decay_critic
        )
        # Scheduler per learning rate
        critic_lr_scheduler = lr_scheduler.StepLR(
            critic_optimizer, 
            step_size=100, 
            gamma=0.5
        )

        # Salva la rete Actor inizializzata (solo se non riprendiamo da checkpoint)
        if not resume_from:
            model_file = os.path.join(weights, "portfolio_actor_initial.pth")
            torch.save(self.actor_local.state_dict(), model_file)

        # Deque per memorizzare metriche di training
        mean_rewards = deque(maxlen=10)
        cum_rewards = []
        actor_losses = deque(maxlen=10)
        critic_losses = deque(maxlen=10)
        portfolio_values = deque(maxlen=10)
        sharpe_ratios = deque(maxlen=10)

        # Variabili per il training
        i = 0
        start_episode = 0
        N_train = total_episodes * env.T // learn_freq
        beta = self.beta0
        self.reset()
        n_train = 0

        # ====== FASE 3: Caricamento del checkpoint ======
        if checkpoint is not None:  # Se abbiamo già caricato il checkpoint
            print(f"Riprendendo l'addestramento da {resume_from}")
            
            try:
                # Carica stato dei modelli
                self.actor_local.load_state_dict(checkpoint['actor_state_dict'])
                self.critic_local.load_state_dict(checkpoint['critic_state_dict'])
                self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
                self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
                
                # Ripristina lo stato delle metriche
                if 'metrics' in checkpoint:
                    metrics = checkpoint['metrics']
                    if 'mean_rewards' in metrics and metrics['mean_rewards']:
                        mean_rewards = deque(metrics['mean_rewards'], maxlen=10)
                    if 'cum_rewards' in metrics and metrics['cum_rewards']:
                        cum_rewards = metrics['cum_rewards']
                    if 'portfolio_values' in metrics and metrics['portfolio_values']:
                        portfolio_values = deque(metrics['portfolio_values'], maxlen=10)
                    if 'sharpe_ratios' in metrics and metrics['sharpe_ratios']:
                        sharpe_ratios = deque(metrics['sharpe_ratios'], maxlen=10)
                    if 'actor_losses' in metrics and metrics['actor_losses']:
                        actor_losses = deque(metrics['actor_losses'], maxlen=10)
                    if 'critic_losses' in metrics and metrics['critic_losses']:
                        critic_losses = deque(metrics['critic_losses'], maxlen=10)
                
                # Ripristina episodio di partenza
                start_episode = checkpoint['episode']
                i = checkpoint.get('iteration', start_episode * env.T)
                n_train = checkpoint.get('n_train', start_episode * env.T // learn_freq)
                self.last_checkpoint_episode = start_episode - 1
                
                print(f"Addestramento ripreso dall'episodio {start_episode}")
            except Exception as e:
                print(f"Errore durante il caricamento del checkpoint: {e}")
                print("Iniziando nuovo addestramento...")
                start_episode = 0

        # Prepara l'agente con esperienze pre-training
        if start_episode == 0:  # Solo se non stiamo riprendendo da un checkpoint
            Node.reset_count()
            self.pretrain(env, total_steps=total_steps)

        # Configura la progress bar se richiesta
        if progress == "tqdm_notebook":
            from tqdm import tqdm_notebook
            range_total_episodes = tqdm_notebook(range(start_episode, total_episodes))
            progress_bar = range_total_episodes
        elif progress == "tqdm":
            from tqdm import tqdm
            range_total_episodes = tqdm(range(start_episode, total_episodes))
            progress_bar = range_total_episodes
        else:
            range_total_episodes = range(start_episode, total_episodes)
            progress_bar = None

        # ====== FASE 4: Loop di addestramento ======
        for episode in range_total_episodes:
            episode_rewards = []
            env.reset()
            state = env.get_state()
            done = env.done
            train_iter = 0
            episode_complete = False

            # Loop per un singolo episodio
            while not done:
                # Calcola il fattore di esplorazione (diminuisce nel tempo)
                explore_probability = explore_stop + (1 - explore_stop) * np.exp(-decay_rate * i)
                
                # Ottieni l'azione dall'agente
                actions = self.act(
                    state, 
                    truncate=(not env.squared_risk), 
                    max_pos=env.max_pos_per_asset, 
                    explore_probability=explore_probability
                )
                
                # Esegui l'azione nell'ambiente
                reward = env.step(actions)
                
                # Logging per TensorBoard
                for j, ticker in enumerate(env.tickers):
                    writer.add_scalar(f"Portfolio/Position/{ticker}", env.positions[j], i)
                    writer.add_scalar(f"Portfolio/Action/{ticker}", actions[j], i)
                
                writer.add_scalar("Portfolio/TotalValue", env.get_portfolio_value(), i)
                writer.add_scalar("Portfolio/Cash", env.cash, i)
                writer.add_scalar("Portfolio/Reward", reward, i)
                
                # Ottieni il nuovo stato
                next_state = env.get_state()
                done = env.done
                
                # Salva l'esperienza nel buffer
                self.step(state, actions, reward, next_state, done)
                
                # Aggiorna lo stato corrente
                state = next_state
                episode_rewards.append(reward)
                i += 1
                train_iter += 1
                
                # A fine episodio, resetta il rumore e calcola metriche
                if done:
                    episode_complete = True
                    self.reset()
                    total_reward = np.sum(episode_rewards)
                    mean_rewards.append(total_reward)
                    
                    # Calcola metriche di portafoglio
                    portfolio_metrics = env.get_real_portfolio_metrics()
                    portfolio_values.append(portfolio_metrics['final_portfolio_value'])
                    sharpe_ratios.append(portfolio_metrics['sharpe_ratio'])
                    
                    # Logging periodico
                    if (episode > 0) and (episode % 5 == 0):
                        mean_r = np.mean(mean_rewards)
                        cum_rewards.append(mean_r)
                        mean_portfolio_value = np.mean(portfolio_values)
                        mean_sharpe = np.mean(sharpe_ratios)
                        
                        writer.add_scalar("Portfolio/AvgReward", mean_r, episode)
                        writer.add_scalar("Portfolio/AvgPortfolioValue", mean_portfolio_value, episode)
                        writer.add_scalar("Portfolio/AvgSharpeRatio", mean_sharpe, episode)
                        writer.add_scalar("Loss/ActorLoss", np.mean(actor_losses), episode)
                        writer.add_scalar("Loss/CriticLoss", np.mean(critic_losses), episode)

                # Addestramento delle reti ogni learn_freq step
                if train_iter % learn_freq == 0:
                    n_train += 1
                    
                    # Campiona esperienze dal buffer
                    if self.memory_type == "uniform":
                        transitions = self.memory.sample(self.batch_size)
                        batch = Transition(*zip(*transitions))
                        states_mb = torch.cat(batch.state)
                        actions_mb = torch.cat(batch.action)
                        rewards_mb = torch.cat(batch.reward)
                        next_states_mb = torch.cat(batch.next_state)
                        dones_mb = torch.cat(batch.dones)
                    elif self.memory_type == "prioritized":
                        transitions, indices = self.memory.sample(self.batch_size)
                        batch = Transition(*zip(*transitions))
                        states_mb = torch.cat(batch.state)
                        actions_mb = torch.cat(batch.action)
                        rewards_mb = torch.cat(batch.reward)
                        next_states_mb = torch.cat(batch.next_state)
                        dones_mb = torch.cat(batch.dones)

                    # Compute target Q values
                    actions_next = self.actor_target(next_states_mb)
                    Q_targets_next = self.critic_target(next_states_mb, actions_next)
                    Q_targets = rewards_mb + (self.gamma * Q_targets_next * dones_mb)
                    
                    # Compute current Q values
                    Q_expected = self.critic_local(states_mb, actions_mb)
                    
                    # Compute TD error
                    td_errors = F.l1_loss(Q_expected, Q_targets, reduction="none")
                    
                    # Gestione di prioritized experience replay
                    if self.memory_type == "prioritized":
                        sum_priorities = self.memory.sum_priorities()
                        probabilities = (self.memory.retrieve_priorities(indices) / sum_priorities).reshape((-1, 1))
                        is_weights = torch.tensor(1 / ((self.max_size * probabilities) ** beta), dtype=torch.float)
                        is_weights /= is_weights.max()
                        
                        # Aggiorna beta annealing
                        beta = (1 - self.beta0) * (n_train / N_train) + self.beta0
                        
                        # Aggiorna priorità nel buffer
                        for i_enum, index in enumerate(indices):
                            self.memory.update(index, (abs(float(td_errors[i_enum].data)) + self.epsilon) ** self.alpha)
                        
                        # Calcola loss pesata
                        critic_loss = (is_weights * (td_errors ** 2)).mean() / 2
                    else:
                        # Loss standard
                        critic_loss = (td_errors ** 2).mean() / 2

                    # Aggiorna la rete Critic
                    critic_losses.append(critic_loss.data.item())
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    # Clip dei gradienti
                    torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), clip_grad_norm)
                    critic_optimizer.step()
                    critic_lr_scheduler.step()

                    # Aggiorna la rete Actor
                    actions_pred = self.actor_local(states_mb)
                    actor_loss = -self.critic_local(states_mb, actions_pred).mean()
                    actor_losses.append(actor_loss.data.item())
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    # Clip dei gradienti
                    torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), clip_grad_norm)
                    actor_optimizer.step()
                    actor_lr_scheduler.step()

                    # Soft update delle reti target
                    self.soft_update(self.critic_local, self.critic_target, tau_critic)
                    self.soft_update(self.actor_local, self.actor_target, tau_actor)

            # Salva il modello periodicamente (solo dopo episodi completi)
            if episode_complete and ((episode % freq) == 0 or episode == total_episodes - 1):
                actor_file = os.path.join(weights, f"portfolio_actor_{episode}.pth")
                critic_file = os.path.join(weights, f"portfolio_critic_{episode}.pth")
                torch.save(self.actor_local.state_dict(), actor_file)
                torch.save(self.critic_local.state_dict(), critic_file)

                # Salva checkpoint completo (solo se è un nuovo episodio rispetto all'ultimo checkpoint)
                if episode != self.last_checkpoint_episode and ((episode % checkpoint_freq == 0) or episode == total_episodes - 1):
                    self.last_checkpoint_episode = episode
                    checkpoint_file = os.path.join(checkpoint_path, f"checkpoint_ep{episode}.pt")
                    
                    # Prepara metriche per il salvataggio
                    metrics = {
                        'mean_rewards': list(mean_rewards),
                        'cum_rewards': cum_rewards,
                        'portfolio_values': list(portfolio_values),
                        'sharpe_ratios': list(sharpe_ratios),
                        'actor_losses': list(actor_losses),
                        'critic_losses': list(critic_losses)
                    }
                    
                    # Crea il checkpoint
                    checkpoint = {
                        'episode': episode + 1,  # +1 perché è il prossimo episodio da cui partire
                        'iteration': i,
                        'n_train': n_train,
                        'actor_state_dict': self.actor_local.state_dict(),
                        'critic_state_dict': self.critic_local.state_dict(),
                        'actor_target_state_dict': self.actor_target.state_dict(),
                        'critic_target_state_dict': self.critic_target.state_dict(),
                        'metrics': metrics
                    }
                    
                    # Salva il checkpoint
                    torch.save(checkpoint, checkpoint_file)
                    print(f"Checkpoint salvato: {checkpoint_file}")

        # Esporta i dati TensorBoard
        writer.export_scalars_to_json("./portfolio_scalars.json")
        writer.close()
        
        # Restituisci statistiche finali
        return {
            'final_rewards': mean_rewards,
            'cum_rewards': cum_rewards,
            'final_portfolio_values': portfolio_values,
            'final_sharpe_ratios': sharpe_ratios
        }
    
    def load_models(self, actor_path, critic_path=None):
        """
        Carica i pesi di modelli salvati.
        
        Parametri:
        - actor_path: percorso al file dei pesi dell'Actor
        - critic_path: percorso al file dei pesi del Critic (opzionale)
        """
        if self.actor_local is not None:
            self.actor_local.load_state_dict(torch.load(actor_path,weights_only=False ))
            if self.actor_target is not None:
                self.actor_target.load_state_dict(torch.load(actor_path, weights_only=False))
                
        if critic_path is not None and self.critic_local is not None:
            self.critic_local.load_state_dict(torch.load(critic_path, weights_only=False))
            if self.critic_target is not None:
                self.critic_target.load_state_dict(torch.load(critic_path, weights_only=False))