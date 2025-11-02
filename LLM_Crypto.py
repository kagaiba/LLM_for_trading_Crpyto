#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.cluster import KMeans
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from itertools import combinations
import matplotlib.pyplot as plt

# ====================================================
# I. CONFIGURAÇÃO E PARÂMETROS
# ====================================================

# Lista EXPANDIDA de tickers (Crypto-USD)
TICKERS = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'ADA-USD', 
    'XRP-USD', 'DOGE-USD', 'DOT-USD', 'LINK-USD', 'LTC-USD', 
    'MATIC-USD', 'AVAX-USD', 'TRX-USD', 'ATOM-USD', 'ETC-USD',
    'BCH-USD', 'XLM-USD', 'UNI3-USD', 'DAI-USD', 'FIL-USD', 
    'NEAR-USD', 'ALGO-USD', 'MANA-USD', 'SAND-USD', 'VET-USD' 
]
START_DATE = '2022-01-01'
END_DATE = '2025-10-31'

# Parâmetros de Machine Learning e Estatística
NUM_CLUSTERS = 7           # Aumentei os clusters de 5 para 7, dada a maior quantidade de ativos.
SIGNIFICANCE_LEVEL = 0.05  # 5% para o teste de Cointegração (p-value < 0.05)

# Parâmetros de Trading e Backtest (Mantidos)
ENTRY_THRESHOLD = 2.0      # Limiar de entrada (ex: 2.0 Desvios-Padrão)
EXIT_THRESHOLD = 0.5       # Limiar de saída (ex: 0.5 Desvios-Padrão)
LOOKBACK_WINDOW = 90       # Janela de dias para regressão e cálculo de Z-Score

MAX_PAIRS_TO_TEST = 50     # Limite de pares para backtesting

# ====================================================
# II. FUNÇÕES CORE (MANTIDAS)
# ====================================================

def get_crypto_prices(tickers, start_date, end_date):
    """Baixa os preços históricos de fechamento dos tickers de criptomoedas."""
    print("1. Baixando dados históricos (yfinance)...")
    df = yf.download(tickers, start=start_date, end=end_date)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])
    return df.dropna(axis=1)

def cluster_assets(df_prices, k):
    """Aplica K-Means nos retornos logarítmicos normalizados para agrupar ativos."""
    print(f"2. Iniciando Clustering K-Means com K={k}...")
    log_returns = np.log(df_prices / df_prices.shift(1)).dropna()
    normalized_returns = (log_returns - log_returns.mean()) / log_returns.std()
    
    data_for_clustering = normalized_returns.T

    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(data_for_clustering)

    clusters = {}
    for ticker, label in zip(data_for_clustering.index, model.labels_):
        if label not in clusters: clusters[label] = []
        clusters[label].append(ticker)
        
    print(f"   -> Encontrados {k} clusters.")
    return clusters

def encontrar_pares_cointegrados(df_prices, clusters, sig_level):
    """Identifica pares cointegrados dentro de cada cluster."""
    pares_validos = []
    print("3. Validando pares por Cointegração...")
    
    for _, tickers_cluster in clusters.items():
        if len(tickers_cluster) < 2: continue
            
        pares_candidatos = list(combinations(tickers_cluster, 2))
        
        for ativo1, ativo2 in pares_candidatos:
            serie1 = df_prices[ativo1].dropna()
            serie2 = df_prices[ativo2].dropna()
            
            common_index = serie1.index.intersection(serie2.index)
            if len(common_index) < LOOKBACK_WINDOW * 2: continue
            
            serie1 = serie1.loc[common_index]
            serie2 = serie2.loc[common_index]

            c_test = coint(serie1, serie2)
            p_value = c_test[1]
            
            if p_value < sig_level:
                pares_validos.append((ativo1, ativo2))
                
    print(f"   -> {len(pares_validos)} pares cointegrados encontrados.")
    return pares_validos

def run_single_backtest(df_prices, ativo_Y, ativo_X, entry_sd, exit_sd, lookback):
    """Executa o backtest e retorna as métricas de performance e o DataFrame de resultados."""
    df_temp = df_prices[[ativo_Y, ativo_X]].copy()
    
    def calcular_beta_rolling(prices_y, prices_x):
        X = sm.add_constant(prices_x)
        try:
            model = sm.OLS(prices_y, X).fit()
            return model.params[ativo_X]
        except:
            return np.nan

    df_temp['Hedge_Ratio'] = df_temp[ativo_Y].rolling(window=lookback).apply(
        lambda y: calcular_beta_rolling(y, df_temp[ativo_X].loc[y.index]), raw=False
    )
    df_temp.dropna(subset=['Hedge_Ratio'], inplace=True)
    if df_temp.empty: return None

    df_temp['Spread'] = df_temp[ativo_Y] - (df_temp['Hedge_Ratio'] * df_temp[ativo_X])
    df_temp['Spread_Media_Movel'] = df_temp['Spread'].rolling(window=lookback).mean()
    df_temp['Spread_StdDev'] = df_temp['Spread'].rolling(window=lookback).std()
    df_temp['Z_Score'] = (df_temp['Spread'] - df_temp['Spread_Media_Movel']) / df_temp['Spread_StdDev']
    
    df_temp.dropna(inplace=True)
    if df_temp.empty: return None

    df_temp['Posicao'] = 0
    df_temp.loc[df_temp['Z_Score'] < -entry_sd, 'Posicao'] = 1
    df_temp.loc[df_temp['Z_Score'] > entry_sd, 'Posicao'] = -1
    df_temp.loc[abs(df_temp['Z_Score']) < exit_sd, 'Posicao'] = 0
    df_temp['Posicao_Ajustada'] = df_temp['Posicao'].replace(to_replace=0, method='ffill').fillna(0)
    
    df_temp['Spread_Diario_Change'] = df_temp['Spread'].diff()
    media_abs_spread = df_temp['Spread'].abs().rolling(window=lookback).mean().shift(1)

    df_temp['Retorno_Estrategia_Percentual'] = (
        df_temp['Posicao_Ajustada'].shift(1) * df_temp['Spread_Diario_Change']
    ) / media_abs_spread
    
    df_temp['Retorno_Acumulado'] = (1 + df_temp['Retorno_Estrategia_Percentual'].fillna(0)).cumprod()
    df_temp.dropna(subset=['Retorno_Acumulado'], inplace=True)
    
    retornos_validos = df_temp['Retorno_Estrategia_Percentual'].dropna()
    
    if len(retornos_validos) < 10: return None

    sharpe_ratio = retornos_validos.mean() / retornos_validos.std() * np.sqrt(365)
    retorno_total = df_temp['Retorno_Acumulado'].iloc[-1] - 1
    roll_max = df_temp['Retorno_Acumulado'].cummax()
    daily_drawdown = df_temp['Retorno_Acumulado'] / roll_max - 1.0
    max_drawdown = daily_drawdown.min()

    return {
        'Par': f'{ativo_Y}/{ativo_X}',
        'Sharpe Ratio': sharpe_ratio,
        'Retorno Total (%)': retorno_total * 100,
        'Max Drawdown (%)': max_drawdown * 100,
        'df_backtest': df_temp
    }

# ====================================================
# III. FLUXO PRINCIPAL DE EXECUÇÃO
# ====================================================

if __name__ == '__main__':
    
    df_prices = get_crypto_prices(TICKERS, START_DATE, END_DATE)
    crypto_clusters = cluster_assets(df_prices, NUM_CLUSTERS)
    pares_selecionados = encontrar_pares_cointegrados(df_prices, crypto_clusters, SIGNIFICANCE_LEVEL)

    pares_a_testar = pares_selecionados[:MAX_PAIRS_TO_TEST] 
    resultados_pares = []
    
    print(f"\n4. Iniciando backtest em {len(pares_a_testar)} dos pares encontrados...")

    # Variável para rastrear o par com o melhor Sharpe para visualização
    melhor_sharpe_global = -np.inf
    melhor_par_para_plot = None

    for ativo_Y, ativo_X in pares_a_testar:
        metrics = run_single_backtest(df_prices, ativo_Y, ativo_X, ENTRY_THRESHOLD, EXIT_THRESHOLD, LOOKBACK_WINDOW)
        
        if metrics is not None:
            resultados_pares.append(metrics)
            
            # Atualiza o melhor par para plotagem
            if metrics['Sharpe Ratio'] > melhor_sharpe_global:
                melhor_sharpe_global = metrics['Sharpe Ratio']
                melhor_par_para_plot = metrics
    
    if not resultados_pares:
        print("\nNenhum par pôde ser backtestado com sucesso.")
    else:
        df_ranking = pd.DataFrame(resultados_pares)
        df_ranking_ordenado = df_ranking.sort_values(by='Sharpe Ratio', ascending=False).reset_index(drop=True)

        print("\n" + "="*80)
        print("🏆 RANKING FINAL DOS MELHORES PARES POR SHARPE RATIO")
        print(f"Parâmetros: Entrada={ENTRY_THRESHOLD} SD | Saída={EXIT_THRESHOLD} SD | Lookback={LOOKBACK_WINDOW} dias")
        print("="*80)
        
        df_ranking_final = df_ranking_ordenado.copy()
        df_ranking_final['Retorno Total (%)'] = df_ranking_final['Retorno Total (%)'].apply(lambda x: f'{x:.2f}%')
        df_ranking_final['Max Drawdown (%)'] = df_ranking_final['Max Drawdown (%)'].apply(lambda x: f'{x:.2f}%')
        df_ranking_final['Sharpe Ratio'] = df_ranking_final['Sharpe Ratio'].apply(lambda x: f'{x:.2f}')

        print(df_ranking_final.drop(columns=['df_backtest']))
        print("="*80)
        print("O melhor par é aquele com o maior Sharpe Ratio (> 1.0 é um bom sinal).")

        # ====================================================
        # IV. VISUALIZAÇÃO DA CURVA DE CAPITAL DO MELHOR PAR
        # ====================================================
        
        # Usa os dados rastreados durante o loop
        if melhor_par_para_plot:
            df_melhor_par_backtest = melhor_par_para_plot['df_backtest']
            nome_melhor_par = melhor_par_para_plot['Par']
            melhor_sharpe = melhor_par_para_plot['Sharpe Ratio']

            plt.figure(figsize=(12, 7))
            plt.plot(df_melhor_par_backtest.index, df_melhor_par_backtest['Retorno_Acumulado'], 
                     label=f'Curva de Capital - {nome_melhor_par}', color='green')
            plt.title(f'Curva de Capital do Melhor Par: {nome_melhor_par} (Sharpe: {melhor_sharpe:.2f})')
            plt.xlabel('Data')
            plt.ylabel('Retorno Acumulado')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.axhline(1.0, color='red', linestyle='--', linewidth=1, label='Capital Inicial (1.0)')
            plt.legend()
            plt.tight_layout()


# In[ ]:




