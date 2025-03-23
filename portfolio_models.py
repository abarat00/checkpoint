import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    """
    Inizializza i pesi del layer in base al numero di input (fan_in)
    usando un intervallo uniforme: (-1/sqrt(fan_in), 1/sqrt(fan_in)).
    """
    fan_in = layer.weight.data.size()[1]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)

class PortfolioActor(nn.Module):
    """
    Rete neurale per la policy (Actor) adattata per portafogli multi-asset.

    Input:
      - state: vettore di stato che include feature di ogni asset, posizioni attuali e metriche di portafoglio.

    Output:
      - Azioni: vettore di azioni, una per ogni asset nel portafoglio.
    """
    def __init__(self, state_size, action_size, seed=0, fc1_units=256, fc2_units=128, fc3_units=64, use_batch_norm=True):
        """
        Inizializza la rete dell'Actor.
        
        Parametri:
        - state_size: dimensione dello stato (feature di tutti gli asset + posizioni + metriche)
        - action_size: numero di asset nel portafoglio (un'azione per asset)
        - seed: seme per riproducibilità
        - fc1_units, fc2_units, fc3_units: dimensioni dei layer nascosti
        - use_batch_norm: se utilizzare la batch normalization
        """
        super(PortfolioActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_batch_norm = use_batch_norm
        
        # Layer lineari
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)
        
        # Batch normalization (opzionale)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(fc1_units)
            self.bn2 = nn.BatchNorm1d(fc2_units)
            self.bn3 = nn.BatchNorm1d(fc3_units)
        
        # Inizializza i parametri
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Inizializza i pesi dei layer con valori appropriati.
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.fill_(0)
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc3.bias.data.fill_(0)
        # Il layer finale viene inizializzato con valori più piccoli
        self.fc4.weight.data.uniform_(-3e-4, 3e-4)
        self.fc4.bias.data.fill_(0)
    
    def forward(self, state):
        """
        Forward pass dell'Actor avanzato.
        """
        batch_size = state.size(0)
        features_per_asset = (state.size(1) - (state.size(1) % self.action_size)) // self.action_size
        
        # Codifica gli asset
        encoded_state = self.asset_encoder(state, self.action_size)
        
        # Applica attenzione se abilitata
        if self.use_attention:
            encoded_state = self.apply_attention(encoded_state, batch_size)
        
        # Debug della dimensione
        #print(f"DEBUG - encoded_state shape dopo attenzione: {encoded_state.shape}, fc1 weight shape: {self.fc1.weight.shape}")
        
        # Adatta dinamicamente il layer FC1 se necessario
        if self.fc1.weight.shape[1] != encoded_state.size(1):
            print(f"ATTENZIONE: Ridimensionamento del layer FC1 da {self.fc1.weight.shape[1]} a {encoded_state.size(1)}")
            old_fc1_out_features = self.fc1.weight.shape[0]
            self.fc1 = torch.nn.Linear(encoded_state.size(1), old_fc1_out_features)
            # Reinizializza i pesi
            fan_in = self.fc1.weight.data.size()[1]
            lim = 1.0 / np.sqrt(fan_in)
            self.fc1.weight.data.uniform_(-lim, lim)
            self.fc1.bias.data.fill_(0)
        
        # Feed-forward con batch norm
        x = F.relu(self.bn1(self.fc1(encoded_state)))
        x = F.relu(self.bn2(self.fc2(x)))
        
        # Output layer
        return self.fc3(x)
    
class PortfolioCritic(nn.Module):
    """
    Rete neurale per il Critic adattata per portafogli multi-asset.
    
    Input:
      - state: vettore di stato che include feature di ogni asset, posizioni e metriche
      - action: vettore di azioni, una per ogni asset
      
    Output:
      - Valore Q: singolo valore che rappresenta il valore stimato dell'azione
    """
    def __init__(self, state_size, action_size, seed=0, fcs1_units=256, fc2_units=128, fc3_units=64, use_batch_norm=True):
        """
        Inizializza la rete del Critic.
        
        Parametri:
        - state_size: dimensione dello stato
        - action_size: numero di asset nel portafoglio
        - seed: seme per riproducibilità
        - fcs1_units, fc2_units, fc3_units: dimensioni dei layer nascosti
        - use_batch_norm: se utilizzare la batch normalization
        """
        super(PortfolioCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_batch_norm = use_batch_norm
        
        # Layer per elaborare lo stato e l'azione
        self.fcs1 = nn.Linear(state_size + action_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        
        # Batch normalization (opzionale)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(fcs1_units)
            self.bn2 = nn.BatchNorm1d(fc2_units)
            self.bn3 = nn.BatchNorm1d(fc3_units)
        
        # Inizializza i parametri
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Inizializza i pesi dei layer con valori appropriati.
        """
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fcs1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.fill_(0)
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc3.bias.data.fill_(0)
        # Il layer finale viene inizializzato con valori più piccoli
        self.fc4.weight.data.uniform_(-3e-4, 3e-4)
        self.fc4.bias.data.fill_(0)
    
    def forward(self, state, action):
        """
        Esegue il forward pass del Critic.
        
        Args:
            state (Tensor): vettore di stato
            action (Tensor): vettore di azioni
            
        Returns:
            Tensor: il valore Q della coppia (stato, azioni)
        """
        # Concatena stato e azione
        x = torch.cat((state, action), dim=1)
        
        # Applica i layer con attivazioni e batch norm (se abilitata)
        if self.use_batch_norm:
            x = F.relu(self.bn1(self.fcs1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.relu(self.bn3(self.fc3(x)))
        else:
            x = F.relu(self.fcs1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
        
        # Layer di output (singolo valore Q)
        return self.fc4(x)

class AssetEncoder(nn.Module):
    """
    Modulo opzionale per codificare le feature di ciascun asset in modo indipendente.
    Questo permette di processare ogni asset con gli stessi pesi prima di combinarli.
    """
    def __init__(self, features_per_asset, encoding_size=16, seed=0, output_size=None):
        """
        Inizializza il codificatore di asset.
        
        Parametri:
        - features_per_asset: numero di feature per singolo asset
        - encoding_size: dimensione dell'encoding per asset
        - seed: seme per riproducibilità
        - output_size: dimensione di output opzionale (diversa da encoding_size)
        """
        super(AssetEncoder, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.features_per_asset = features_per_asset
        self.encoding_size = encoding_size
        self.output_size = output_size or encoding_size  # Usa output_size se specificato
        
        # Layer di encoding
        self.fc1 = nn.Linear(features_per_asset, 32)
        self.fc2 = nn.Linear(32, self.output_size)
        
        # Inizializza i parametri
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Inizializza i pesi dei layer con valori appropriati.
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.fill_(0)
    
    def forward(self, state, num_assets):
        """
        Codifica le feature di ciascun asset indipendentemente.
        
        Args:
            state (Tensor): stato completo [batch_size, features_per_asset*num_assets + extra]
            num_assets: numero di asset nel portafoglio
            
        Returns:
            Tensor: feature codificate [batch_size, num_assets*encoding_size + extra]
        """
        batch_size = state.size(0)
        
        # Estrai le feature di ciascun asset
        asset_features = state[:, :num_assets*self.features_per_asset]
        
        # Riorganizza per processare ogni asset indipendentemente
        asset_features = asset_features.view(batch_size, num_assets, self.features_per_asset)
        
        # Codifica ogni asset
        x = F.relu(self.fc1(asset_features))
        asset_encodings = F.relu(self.fc2(x))
        
        # Appiattisci gli encoding
        asset_encodings = asset_encodings.view(batch_size, num_assets * self.encoding_size)
        
        # Se ci sono feature aggiuntive nello stato originale, concatenale agli encoding
        if state.size(1) > num_assets * self.features_per_asset:
            extra_features = state[:, num_assets*self.features_per_asset:]
            return torch.cat((asset_encodings, extra_features), dim=1)
        
        return asset_encodings

class EnhancedPortfolioActor(nn.Module):
    """
    Versione avanzata dell'Actor che utilizza un encoder per asset e meccanismi di attenzione.
    Adatta per portafogli con molti asset e relazioni complesse.
    """
    def __init__(self, state_size, action_size, features_per_asset, seed=0, 
             fc1_units=256, fc2_units=128, encoding_size=16, use_attention=True,
             attention_size=None, encoder_output_size=None):
        """
        Inizializza l'Actor avanzato.
        
        Parametri:
        - state_size: dimensione dello stato completo
        - action_size: numero di asset nel portafoglio
        - features_per_asset: feature per singolo asset
        - encoding_size: dimensione dell'encoding per asset
        - attention_size: dimensione opzionale per il layer di attenzione
        - encoder_output_size: dimensione di output opzionale per l'encoder
        """
        super(EnhancedPortfolioActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        
        # Dimensioni speciali per ripristino da checkpoint
        attention_dim = attention_size or encoding_size
        
        # Encoder per asset
        self.asset_encoder = AssetEncoder(
            features_per_asset, 
            encoding_size=encoding_size, 
            seed=seed,
            output_size=encoder_output_size
        )
        
        # Determina le dimensioni reali in uso
        real_encoding_size = encoder_output_size or encoding_size
        
        # Extra feature (posizioni attuali + metriche di portfolio)
        extra_features = state_size - (features_per_asset * action_size)
        
        # Se attenzione è abilitata, calcola dimensione considerando contesto
        if use_attention:
            # Quando l'output contiene contesto per ogni asset
            encoded_size = (action_size * attention_dim * 2) + extra_features
        else:
            encoded_size = (action_size * real_encoding_size) + extra_features
        
        # Layer principali con dimensioni calcolate dinamicamente
        self.fc1 = nn.Linear(encoded_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        # Layer di attenzione per modellare relazioni tra asset
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Linear(real_encoding_size, 1)
            self.value = nn.Linear(real_encoding_size, attention_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        
        # Inizializza i parametri
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Inizializza i pesi dei layer con valori appropriati.
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.fill_(0)
        self.fc3.weight.data.uniform_(-3e-4, 3e-4)
        self.fc3.bias.data.fill_(0)
        
        if self.use_attention:
            self.attention.weight.data.uniform_(*hidden_init(self.attention))
            self.attention.bias.data.fill_(0)
            self.value.weight.data.uniform_(*hidden_init(self.value))
            self.value.bias.data.fill_(0)
    
    def apply_attention(self, encoded_assets, batch_size):
        """
        Applica meccanismo di attenzione tra asset.
        """
        # Controlla le dimensioni effettive
        total_size = encoded_assets.size(1)
        
        # Debug informazioni
        #print(f"DEBUG - apply_attention: total_size={total_size}, action_size={self.action_size}")
        
        # Verifica che la divisione sia esatta
        if total_size % self.action_size != 0:
            #print(f"ATTENZIONE: Dimensione encoding non divisibile: {total_size} / {self.action_size}")
            encoding_size = total_size // self.action_size
            remainder = total_size % self.action_size
            
            # Aggiungi padding se necessario
            if remainder > 0:
                encoding_size += 1
                padding_needed = encoding_size * self.action_size - total_size
                padding = torch.zeros(batch_size, padding_needed, device=encoded_assets.device)
                encoded_assets = torch.cat([encoded_assets, padding], dim=1)
                #print(f"Aggiunto padding: nuovo encoding_size = {encoding_size}, nuova dimensione = {encoded_assets.size(1)}")
        else:
            # Calcola normalmente se divisibile
            encoding_size = total_size // self.action_size
        
        # Riorganizza in [batch, num_assets, features_per_asset]
        try:
            assets = encoded_assets.view(batch_size, self.action_size, encoding_size)
            #print(f"Reshape riuscito: assets.shape={assets.shape}")
        except RuntimeError as e:
            print(f"Errore nel reshape: encoded_assets.shape={encoded_assets.shape}, batch_size={batch_size}, action_size={self.action_size}, encoding_size={encoding_size}")
            print(f"Totale elementi: {encoded_assets.numel()}, target: {batch_size * self.action_size * encoding_size}")
            
            # Fallback: ricalcola una dimensione sicura
            safe_encoding_size = encoded_assets.numel() // (batch_size * self.action_size)
            print(f"Tentativo con encoding_size sicuro: {safe_encoding_size}")
            
            # Aggiungi padding se necessario
            if encoded_assets.numel() < batch_size * self.action_size * safe_encoding_size:
                padding_needed = batch_size * self.action_size * safe_encoding_size - encoded_assets.numel()
                padding = torch.zeros(batch_size, padding_needed, device=encoded_assets.device)
                encoded_assets = torch.cat([encoded_assets, padding], dim=1)
                print(f"Aggiunto padding di emergenza, nuova dimensione: {encoded_assets.size()}")
            
            assets = encoded_assets.view(batch_size, self.action_size, safe_encoding_size)
            encoding_size = safe_encoding_size
            print(f"Reshape di fallback riuscito: {assets.shape}")
        
        # Se necessario, adatta il layer di attenzione alle dimensioni corrette
        if self.attention.weight.shape[1] != encoding_size:
            print(f"ATTENZIONE: Ridimensionamento del layer di attenzione da {self.attention.weight.shape[1]} a {encoding_size}")
            old_attention = self.attention
            self.attention = torch.nn.Linear(encoding_size, 1)
            # Inizializza con valori sensati
            fan_in = encoding_size
            lim = 1.0 / np.sqrt(fan_in)
            self.attention.weight.data.uniform_(-lim, lim)
            self.attention.bias.data.fill_(0)
        
        # Calcola punteggi di attenzione
        attention_scores = self.attention(assets).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)
        
        # Trasforma gli asset - adatta il layer value se necessario
        if self.value.weight.shape[1] != encoding_size:
            print(f"ATTENZIONE: Ridimensionamento del layer value da {self.value.weight.shape[1]} a {encoding_size}")
            old_value = self.value
            self.value = torch.nn.Linear(encoding_size, encoding_size)
            # Inizializza con valori sensati
            fan_in = encoding_size
            lim = 1.0 / np.sqrt(fan_in)
            self.value.weight.data.uniform_(-lim, lim)
            self.value.bias.data.fill_(0)
        
        values = self.value(assets)
        
        # Applica attenzione
        context = (attention_weights * values).sum(dim=1)
        
        # Espandi il contesto e concatena con gli encoding originali
        context_expanded = context.unsqueeze(1).expand(-1, self.action_size, -1)
        enhanced_assets = torch.cat((assets, context_expanded), dim=2)
        
        # Appiattisci il risultato
        enhanced_size = enhanced_assets.size(2) * self.action_size
        flattened = enhanced_assets.view(batch_size, enhanced_size)
        
        #print(f"Enhanced output shape: {flattened.shape}")
        
        return flattened
    
    def forward(self, state):
        """
        Forward pass dell'Actor avanzato.
        """
        batch_size = state.size(0)
        
        # Estrai le feature extra (ultime 10 feature dello stato)
        # Le feature extra sono posizioni (num_assets) + metriche di portfolio (5)
        extra_features_size = self.action_size + 5  # posizioni + metriche
        
        if state.size(1) > self.features_per_asset * self.action_size + extra_features_size:
            print(f"AVVISO: Lo stato ha dimensione {state.size(1)}, più grande del previsto")
        
        # Assume che le feature extra siano in fondo
        extra_features = state[:, -extra_features_size:] if state.size(1) > extra_features_size else None
        
        # Codifica gli asset
        encoded_state = self.asset_encoder(state, self.action_size)
        
        # Applica attenzione se abilitata
        if self.use_attention:
            attention_output = self.apply_attention(encoded_state, batch_size)
            
            # Concatena con feature extra se presenti
            if extra_features is not None:
                encoded_state = torch.cat((attention_output, extra_features), dim=1)
                #print(f"Concatenato attention_output {attention_output.shape} con extra_features {extra_features.shape}")
            else:
                encoded_state = attention_output
                #print(f"Nessuna feature extra da concatenare, usando solo attention_output {attention_output.shape}")
        
        # Debug della dimensione finale prima di FC1
        #print(f"DEBUG - forward: encoded_state final shape: {encoded_state.shape}, fc1 weight shape: {self.fc1.weight.shape}")
        
        # Adatta dinamicamente il layer FC1 se necessario
        if self.fc1.weight.shape[1] != encoded_state.size(1):
            #print(f"ATTENZIONE: Ridimensionamento del layer FC1 da {self.fc1.weight.shape[1]} a {encoded_state.size(1)}")
            old_fc1_out_features = self.fc1.weight.shape[0]
            new_fc1 = torch.nn.Linear(encoded_state.size(1), old_fc1_out_features)
            
            # Inizializza i pesi
            fan_in = new_fc1.weight.data.size()[1]
            lim = 1.0 / np.sqrt(fan_in)
            new_fc1.weight.data.uniform_(-lim, lim)
            new_fc1.bias.data.fill_(0)
            
            self.fc1 = new_fc1
        
        # Feed-forward con batch norm
        try:
            x = F.relu(self.bn1(self.fc1(encoded_state)))
            x = F.relu(self.bn2(self.fc2(x)))
            
            # Output layer
            return self.fc3(x)
        except RuntimeError as e:
            print(f"ERRORE durante il forward pass: {e}")
            print(f"Dettagli: encoded_state={encoded_state.shape}, fc1.weight={self.fc1.weight.shape}")
            # Fallback sicuro
            return torch.zeros(batch_size, self.action_size)