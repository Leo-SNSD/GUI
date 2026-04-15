import os
import numpy as np
import random
import tensorflow as tf
from collections import deque
import pandas as pd
import streamlit as st
import time
import gc      
import io                              
import zipfile                         

# 解決 Matplotlib 在 Streamlit 背景執行的記憶體洩漏問題
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt        

# 隱藏 TensorFlow 煩人的除錯警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 設定 Matplotlib 支援顯示繁體中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 介面與環境初始化設定
# ==========================================
st.set_page_config(page_title="MEC 線上卸載最佳化系統", layout="wide")

st.title("基於深度強化學習之線上運算卸載系統")
st.markdown("本系統展示 **TOP-L 策略** 與 **DROO (OPQ)** 於行動邊緣運算(MEC)場景下之加權總延遲與訓練 Loss 收斂比較。")

# --- 側邊欄：展示模式與參數設定區 ---
st.sidebar.header("展示模式設定")
demo_mode = st.sidebar.radio(
    "請選擇系統運作模式：",
    [
        "🚀 現場即時運算模式 (單次推論展示)", 
        "⏳ 現場多種子運算模式 (現場算平均與陰影)", 
        "🔍 L參數敏感度分析模式 (Elbow Curve)", 
        "📊 預算數據解析模式 (載入完整大數據)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.header("系統與演算法參數設定")
N = st.sidebar.selectbox("終端設備數量 (N)", [20, 30, 40, 50], index=2)

if demo_mode != "🔍 L參數敏感度分析模式 (Elbow Curve)":
    L_PARAM = st.sidebar.slider("TOP-L 探索數量 (L)", min_value=1, max_value=N, value=N-2)
else:
    st.sidebar.markdown("---")
    l_input = st.sidebar.text_input("輸入欲測試的 L 值 (以逗號分隔)", f"1, 2, 5, 10, 15, {N//2}, {N}")

EPISODES = st.sidebar.slider("訓練時槽總數 (Episodes)", min_value=100, max_value=17500, value=3000, step=100)
LEARNING_RATE = st.sidebar.selectbox("學習率 (Learning Rate)", [0.1, 0.05, 0.01], index=0)

st.sidebar.markdown("---")
st.sidebar.header("圖表顯示與分析參數")
MA_WINDOW = st.sidebar.number_input(
    "移動平均視窗大小 (MA Window)", min_value=1, max_value=EPISODES, value=min(500, EPISODES), step=50
)

if demo_mode == "🔍 L參數敏感度分析模式 (Elbow Curve)":
    CALC_WINDOW = st.sidebar.number_input(
        "收斂判定基準 (末幾步平均)", min_value=1, max_value=EPISODES, value=min(500, EPISODES), step=50
    )

K = N + 1           
BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 1024
TRAIN_INTERVAL = 8  
f_s, c, s_val = 4.0, 1.0, 1.0
r_min, r_max = 0, 2

@st.cache_data
def load_env_params(N):
    if N == 20:
        f_i = np.array([0.56, 0.26, 0.52, 0.53, 0.61, 0.13, 0.22, 0.51, 0.19, 0.37, 0.32, 0.42, 0.24, 0.34, 0.64, 0.61, 0.42, 0.48, 0.10, 0.40])
        q = np.array([1, 1.5, 1.5, 1.5, 1.5, 1, 1, 1.5, 1, 1.5, 1.5, 1, 1.5, 1, 1, 1, 1.5, 1, 1, 1.5])
    elif N == 30:
        f_i = np.array([0.56, 0.26, 0.52, 0.53, 0.61, 0.13, 0.22, 0.51, 0.19, 0.37, 0.32, 0.42, 0.24, 0.34, 0.64, 0.61, 0.42, 0.48, 0.10, 0.40, 0.48, 0.25, 0.22, 0.4, 0.49, 0.33, 0.63, 0.47, 0.36, 0.31])
        q = np.array([1, 1.5, 1.5, 1.5, 1.5, 1, 1, 1.5, 1, 1.5, 1.5, 1, 1.5, 1, 1, 1, 1.5, 1, 1, 1.5, 1, 1.5, 1.5, 1.5, 1, 1.5, 1.5, 1, 1, 1])
    elif N == 40:
        f_i = np.array([0.56, 0.26, 0.52, 0.53, 0.61, 0.13, 0.22, 0.51, 0.19, 0.37, 0.32, 0.42, 0.24, 0.34, 0.64, 0.61, 0.42, 0.48, 0.10, 0.40, 0.48, 0.25, 0.22, 0.4, 0.49, 0.33, 0.63, 0.47, 0.36, 0.31, 0.29, 0.49, 0.34, 0.13, 0.31, 0.5, 0.2, 0.19, 0.39, 0.39])
        q = np.array([1, 1.5, 1.5, 1.5, 1.5, 1, 1, 1.5, 1, 1.5, 1.5, 1, 1.5, 1, 1, 1, 1.5, 1, 1, 1.5, 1, 1.5, 1.5, 1.5, 1, 1.5, 1.5, 1, 1, 1, 1.5, 1, 1.5, 1, 1.5, 1, 1.5, 1, 1.5, 1])
    else: 
        f_i = np.array([0.56, 0.26, 0.52, 0.53, 0.61, 0.13, 0.22, 0.51, 0.19, 0.37, 0.32, 0.42, 0.24, 0.34, 0.64, 0.61, 0.42, 0.48, 0.10, 0.40, 0.48, 0.25, 0.22, 0.4, 0.49, 0.33, 0.63, 0.47, 0.36, 0.31, 0.29, 0.49, 0.34, 0.13, 0.31, 0.5, 0.2, 0.19, 0.39, 0.39, 0.44, 0.56, 0.49, 0.43, 0.49, 0.27, 0.3, 0.22, 0.26, 0.44])
        q = np.array([1, 1.5, 1.5, 1.5, 1.5, 1, 1, 1.5, 1, 1.5, 1.5, 1, 1.5, 1, 1, 1, 1.5, 1, 1, 1.5, 1, 1.5, 1.5, 1.5, 1, 1.5, 1.5, 1, 1, 1, 1.5, 1, 1.5, 1, 1.5, 1, 1.5, 1, 1.5, 1, 1, 1.5, 1, 1.5, 1.5, 1, 1, 1.5, 1.5, 1])
    return f_i, q

f_i, q = load_env_params(N)
uniform_initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
loss_function = tf.keras.losses.BinaryCrossentropy()

# ==========================================
# 2. 核心模組與演算法
# ==========================================
class DRLNetwork(tf.keras.Model):
    def __init__(self, i_s, h_s, o_s):
        super(DRLNetwork, self).__init__()
        self.d1 = tf.keras.layers.Dense(h_s[0], activation='relu', kernel_initializer=uniform_initializer)
        self.d2 = tf.keras.layers.Dense(h_s[1], activation='relu', kernel_initializer=uniform_initializer)
        self.d3 = tf.keras.layers.Dense(o_s, activation='sigmoid', kernel_initializer=uniform_initializer)
    def call(self, x): return self.d3(self.d2(self.d1(x)))

def resource_allocation(q, x):
    active_indices = np.where(x == 1)[0]
    if len(active_indices) == 0: return np.zeros_like(q)
    k = np.zeros_like(q)
    k[active_indices] = np.sqrt(q[active_indices]) / np.sum(np.sqrt(q[active_indices]))
    return k

def compute_total_delay(x, k, q, r, f_s, f_i):
    d_L = c / f_i
    k = np.where(k == 0, 1e-9, k)
    d_ES = s_val / r + c / (k * f_s)
    return np.sum(q * (x * d_ES + (1 - x) * d_L))

def order_preserving_quantization(x, K_val):
    K_val = min(K_val, len(x) + 1)
    sorted_indices = np.argsort(np.abs(x - 0.5))
    actions = [np.round(x)]
    for i in range(1, K_val):
        action = np.zeros_like(x)
        threshold = x[sorted_indices[i - 1]]
        for j in range(len(x)):
            if x[j] > threshold or (x[j] == threshold and threshold <= 0.5):
                action[j] = 1
        actions.append(action)
    return actions

def top_l_candidate_generation(soft_action, L):
    p = np.asarray(soft_action, float)
    N_len = len(p)
    u = np.abs(p - 0.5)
    idx_hi = np.argpartition(u, L - 1)[:L]
    anchor = (p >= 0.5).astype(int)
    cand = [anchor, np.zeros(N_len, dtype=int), np.ones(N_len, dtype=int)]
    for j in idx_hi:
        x = anchor.copy()
        x[j] ^= 1
        cand.append(x)
    return np.unique(np.asarray(cand, dtype=int), axis=0)

def random_scheme(r):                 
    best_delay = float('inf')
    for _ in range(K):
        action = np.random.randint(0, 2, size=N)
        delay = compute_total_delay(action, resource_allocation(q, action), q, r, f_s, f_i)
        best_delay = min(best_delay, delay)
    return best_delay

def user_based_scheme(r):
    best_action = []
    for i in range(N):
        d_L = c / f_i[i]
        d_ES = s_val / r[i] + c / (np.sqrt(q[i]) / np.sum(np.sqrt(q)) * f_s)
        best_action.append(1 if d_ES < d_L else 0)
    best_action = np.array(best_action)
    return compute_total_delay(best_action, resource_allocation(q, best_action), q, r, f_s, f_i)


# ==========================================
# 3. 系統運作模式切換
# ==========================================

# 動態計算更新頻率，防止頻繁更新擠爆瀏覽器 RAM (最多更新約 50 次)
UPDATE_INTERVAL = max(50, EPISODES // 50)

# ------------------------------------------
# 模式 1: 現場即時運算模式 (Live Demo)
# ------------------------------------------
if demo_mode == "🚀 現場即時運算模式 (單次推論展示)":
    
    if st.sidebar.button("🚀 開始即時模擬訓練", type="primary"):
        st.markdown("### 📊 訓練收斂過程")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 保留即時數值看板
        col1, col2, col3, col4 = st.columns(4)
        metric_droo = col1.empty()
        metric_topl = col2.empty()
        metric_rand = col3.empty()
        metric_user = col4.empty()

        st.markdown("#### 📉 系統加權總延遲 (Moving Average)")
        chart_placeholder_delay = st.empty()
        st.markdown("#### 📉 神經網路訓練損失 Loss (Moving Average)")
        chart_placeholder_loss = st.empty()

        tf.random.set_seed(42)
        np.random.seed(42)
        random.seed(42)

        droo_model = DRLNetwork(N, [80, 64], N)
        top_l_model = DRLNetwork(N, [80, 64], N)
        _ = droo_model(tf.zeros((1, N)))
        _ = top_l_model(tf.zeros((1, N)))
        droo_model.set_weights(top_l_model.get_weights())

        optimizer_droo = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
        optimizer_top_l = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

        replay_memory_droo = deque(maxlen=REPLAY_MEMORY_SIZE)
        replay_memory_top_l = deque(maxlen=REPLAY_MEMORY_SIZE)

        history = {
            "Episode": [], 
            "DROO (OPQ)": [], "TOP-L": [], "Random": [], "UserBased": [],
            "DROO_Loss": [], "TOP-L_Loss": []
        }
        
        cur_loss_droo = 0.0
        cur_loss_top_l = 0.0

        for episode in range(1, EPISODES + 1):
            r = np.random.uniform(r_min, r_max, N)
            r_tensor = tf.convert_to_tensor([r], dtype=tf.float32)

            rand_delay = random_scheme(r) 
            user_delay = user_based_scheme(r)

            # DROO
            droo_soft_x = droo_model(r_tensor).numpy().flatten()
            droo_candidates = order_preserving_quantization(droo_soft_x, K)
            droo_best = min(droo_candidates, key=lambda x: compute_total_delay(x, resource_allocation(q, x), q, r, f_s, f_i))
            droo_delay = compute_total_delay(droo_best, resource_allocation(q, droo_best), q, r, f_s, f_i)
            replay_memory_droo.append((r, droo_best))

            # TOP-L
            top_l_soft_x = top_l_model(r_tensor).numpy().flatten()
            top_l_candidates = top_l_candidate_generation(top_l_soft_x, L_PARAM)
            top_l_best = min(top_l_candidates, key=lambda x: compute_total_delay(x, resource_allocation(q, x), q, r, f_s, f_i))
            top_l_delay = compute_total_delay(top_l_best, resource_allocation(q, top_l_best), q, r, f_s, f_i)
            replay_memory_top_l.append((r, top_l_best))

            # 訓練網路
            if len(replay_memory_droo) >= BATCH_SIZE and (episode % TRAIN_INTERVAL == 0):
                batch_droo = random.sample(replay_memory_droo, BATCH_SIZE)
                r_batch, x_batch = zip(*batch_droo)
                with tf.GradientTape() as tape:
                    preds = droo_model(tf.convert_to_tensor(np.array(r_batch), dtype=tf.float32), training=True)
                    loss_val = loss_function(tf.convert_to_tensor(np.array(x_batch), dtype=tf.float32), preds)
                optimizer_droo.apply_gradients(zip(tape.gradient(loss_val, droo_model.trainable_variables), droo_model.trainable_variables))
                cur_loss_droo = loss_val.numpy()

                batch_top_l = random.sample(replay_memory_top_l, BATCH_SIZE)
                r_batch, x_batch = zip(*batch_top_l)
                with tf.GradientTape() as tape:
                    preds = top_l_model(tf.convert_to_tensor(np.array(r_batch), dtype=tf.float32), training=True)
                    loss_val = loss_function(tf.convert_to_tensor(np.array(x_batch), dtype=tf.float32), preds)
                optimizer_top_l.apply_gradients(zip(tape.gradient(loss_val, top_l_model.trainable_variables), top_l_model.trainable_variables))
                cur_loss_top_l = loss_val.numpy()

            history["Episode"].append(episode)
            history["DROO (OPQ)"].append(droo_delay)
            history["TOP-L"].append(top_l_delay)
            history["Random"].append(rand_delay)
            history["UserBased"].append(user_delay)
            history["DROO_Loss"].append(cur_loss_droo)
            history["TOP-L_Loss"].append(cur_loss_top_l)

            # 更新 UI (利用降採樣機制防止記憶體溢位)
            if episode % UPDATE_INTERVAL == 0 or episode == EPISODES:
                progress_bar.progress(episode / EPISODES)
                status_text.text(f"訓練進度: {episode} / {EPISODES} 步")
                gc.collect() # 不在迴圈內 clear_session()
                
                df = pd.DataFrame(history).set_index("Episode")
                df_ma = df.rolling(window=MA_WINDOW, min_periods=1).mean()
                
                # 降採樣：確保最多送出約 300 個點，大幅降低瀏覽器渲染負擔
                plot_step = max(1, len(df_ma) // 300)
                df_ma_plot = df_ma.iloc[::plot_step]
                
                chart_placeholder_delay.line_chart(df_ma_plot[["DROO (OPQ)", "TOP-L", "Random", "UserBased"]])
                chart_placeholder_loss.line_chart(df_ma_plot[["DROO_Loss", "TOP-L_Loss"]])

                metric_droo.metric("DROO 延遲", f"{df_ma['DROO (OPQ)'].iloc[-1]:.2f}")
                metric_topl.metric("TOP-L 延遲 (提出方法)", f"{df_ma['TOP-L'].iloc[-1]:.2f}")
                metric_rand.metric("Random 延遲", f"{df_ma['Random'].iloc[-1]:.2f}")
                metric_user.metric("UserBased 延遲", f"{df_ma['UserBased'].iloc[-1]:.2f}")

        # 單次訓練完全結束後，才徹底清除 TF session
        tf.keras.backend.clear_session()
        gc.collect()

        # 下載單次紀錄
        st.markdown("---")
        csv_df = pd.DataFrame(history)
        csv_data = csv_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下載單次實驗數據 (包含 Loss CSV)", data=csv_data, file_name=f'MEC_Single_N{N}_L{L_PARAM}.csv', mime='text/csv')

# ------------------------------------------
# 模式 2: 現場多種子運算模式 (現場算平均與陰影)
# ------------------------------------------
elif demo_mode == "⏳ 現場多種子運算模式 (現場算平均與陰影)":
    
    num_runs = st.sidebar.slider("執行次數 (Seeds)", min_value=2, max_value=5, value=3)
    
    if st.sidebar.button("🚀 啟動動態多種子運算", type="primary"):
        st.markdown("### 📡 現場即時訓練監控站")
        progress_bar = st.progress(0)
        seed_status_text = st.empty()       
        episode_status_text = st.empty()    
        
        # 數值看板
        col1, col2, col3, col4 = st.columns(4)
        metric_droo = col1.empty()
        metric_topl = col2.empty()
        metric_rand = col3.empty()
        metric_user = col4.empty()
        
        st.markdown("#### 📉 當前 Seed 延遲收斂")
        realtime_chart_placeholder_delay = st.empty() 
        st.markdown("#### 📉 當前 Seed 訓練 Loss")
        realtime_chart_placeholder_loss = st.empty() 
        
        completed_runs_container = st.container()
        
        all_runs_droo_delay = np.zeros((num_runs, EPISODES))
        all_runs_top_l_delay = np.zeros((num_runs, EPISODES))
        all_runs_droo_loss = np.zeros((num_runs, EPISODES))
        all_runs_top_l_loss = np.zeros((num_runs, EPISODES))
        
        seeds = [42, 123, 456, 789, 1024][:num_runs]
        
        for run_idx, current_seed in enumerate(seeds):
            seed_status_text.markdown(f"**🔄 正在執行第 {run_idx + 1} / {num_runs} 次獨立訓練 (亂數種子: {current_seed})...**")
            current_history = {
                "Episode": [], 
                "DROO (OPQ)": [], "TOP-L": [], "Random": [], "UserBased": [],
                "DROO_Loss": [], "TOP-L_Loss": []
            }
            
            tf.random.set_seed(current_seed)
            np.random.seed(current_seed)
            random.seed(current_seed)

            droo_model = DRLNetwork(N, [80, 64], N)
            top_l_model = DRLNetwork(N, [80, 64], N)
            _ = droo_model(tf.zeros((1, N)))
            _ = top_l_model(tf.zeros((1, N)))
            droo_model.set_weights(top_l_model.get_weights())

            optimizer_droo = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
            optimizer_top_l = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

            replay_memory_droo = deque(maxlen=REPLAY_MEMORY_SIZE)
            replay_memory_top_l = deque(maxlen=REPLAY_MEMORY_SIZE)

            cur_loss_droo = 0.0
            cur_loss_top_l = 0.0

            for episode in range(1, EPISODES + 1):
                r = np.random.uniform(r_min, r_max, N)
                r_tensor = tf.convert_to_tensor([r], dtype=tf.float32)

                rand_delay = random_scheme(r) 
                user_delay = user_based_scheme(r)

                # DROO
                droo_soft_x = droo_model(r_tensor).numpy().flatten()
                droo_candidates = order_preserving_quantization(droo_soft_x, K)
                droo_best = min(droo_candidates, key=lambda x: compute_total_delay(x, resource_allocation(q, x), q, r, f_s, f_i))
                droo_delay = compute_total_delay(droo_best, resource_allocation(q, droo_best), q, r, f_s, f_i)
                all_runs_droo_delay[run_idx, episode-1] = droo_delay
                replay_memory_droo.append((r, droo_best))

                # TOP-L
                top_l_soft_x = top_l_model(r_tensor).numpy().flatten()
                top_l_candidates = top_l_candidate_generation(top_l_soft_x, L_PARAM)
                top_l_best = min(top_l_candidates, key=lambda x: compute_total_delay(x, resource_allocation(q, x), q, r, f_s, f_i))
                top_l_delay = compute_total_delay(top_l_best, resource_allocation(q, top_l_best), q, r, f_s, f_i)
                all_runs_top_l_delay[run_idx, episode-1] = top_l_delay
                replay_memory_top_l.append((r, top_l_best))

                if len(replay_memory_droo) >= BATCH_SIZE and (episode % TRAIN_INTERVAL == 0):
                    batch_droo = random.sample(replay_memory_droo, BATCH_SIZE)
                    r_batch, x_batch = zip(*batch_droo)
                    with tf.GradientTape() as tape:
                        preds = droo_model(tf.convert_to_tensor(np.array(r_batch), dtype=tf.float32), training=True)
                        loss_val = loss_function(tf.convert_to_tensor(np.array(x_batch), dtype=tf.float32), preds)
                    optimizer_droo.apply_gradients(zip(tape.gradient(loss_val, droo_model.trainable_variables), droo_model.trainable_variables))
                    cur_loss_droo = loss_val.numpy()

                    batch_top_l = random.sample(replay_memory_top_l, BATCH_SIZE)
                    r_batch, x_batch = zip(*batch_top_l)
                    with tf.GradientTape() as tape:
                        preds = top_l_model(tf.convert_to_tensor(np.array(r_batch), dtype=tf.float32), training=True)
                        loss_val = loss_function(tf.convert_to_tensor(np.array(x_batch), dtype=tf.float32), preds)
                    optimizer_top_l.apply_gradients(zip(tape.gradient(loss_val, top_l_model.trainable_variables), top_l_model.trainable_variables))
                    cur_loss_top_l = loss_val.numpy()
                
                all_runs_droo_loss[run_idx, episode-1] = cur_loss_droo
                all_runs_top_l_loss[run_idx, episode-1] = cur_loss_top_l

                current_history["Episode"].append(episode)
                current_history["DROO (OPQ)"].append(droo_delay)
                current_history["TOP-L"].append(top_l_delay)
                current_history["Random"].append(rand_delay)     
                current_history["UserBased"].append(user_delay)  
                current_history["DROO_Loss"].append(cur_loss_droo)
                current_history["TOP-L_Loss"].append(cur_loss_top_l)

                if episode % UPDATE_INTERVAL == 0 or episode == EPISODES:
                    current_progress = (run_idx * EPISODES + episode) / (num_runs * EPISODES)
                    progress_bar.progress(current_progress)
                    episode_status_text.text(f"當前進度: {episode} / {EPISODES} 步")
                    gc.collect()
                    
                    df_current = pd.DataFrame(current_history).set_index("Episode")
                    df_ma = df_current.rolling(window=MA_WINDOW, min_periods=1).mean()
                    
                    plot_step = max(1, len(df_ma) // 300)
                    df_ma_plot = df_ma.iloc[::plot_step]
                    
                    realtime_chart_placeholder_delay.line_chart(df_ma_plot[["DROO (OPQ)", "TOP-L", "Random", "UserBased"]])
                    realtime_chart_placeholder_loss.line_chart(df_ma_plot[["DROO_Loss", "TOP-L_Loss"]])
                    
                    metric_droo.metric("DROO 延遲", f"{df_ma['DROO (OPQ)'].iloc[-1]:.2f}")
                    metric_topl.metric("TOP-L 延遲 (提出方法)", f"{df_ma['TOP-L'].iloc[-1]:.2f}")
                    metric_rand.metric("Random 延遲", f"{df_ma['Random'].iloc[-1]:.2f}")
                    metric_user.metric("UserBased 延遲", f"{df_ma['UserBased'].iloc[-1]:.2f}")
            
            with completed_runs_container.expander(f"✅ 第 {run_idx + 1} 次訓練結果 (Seed: {current_seed}) - 點擊展開", expanded=False):
                st.write("**延遲曲線**")
                st.line_chart(df_ma[["DROO (OPQ)", "TOP-L", "Random", "UserBased"]].iloc[::plot_step])
                st.write("**訓練 Loss 曲線**")
                st.line_chart(df_ma[["DROO_Loss", "TOP-L_Loss"]].iloc[::plot_step])
                
            realtime_chart_placeholder_delay.empty()
            realtime_chart_placeholder_loss.empty()
            
            # 單一 Seed 結束後清空模型
            tf.keras.backend.clear_session()
            gc.collect()
            
        seed_status_text.markdown("✅ **所有訓練完成！正在進行多維度數據融合與繪製變異數分析圖...**")
        
        st.markdown("---")
        st.markdown("### 🛡️ 演算法強健性最終分析 (平均值 $\pm 1\sigma$ 標準差)")
        
        # Delay 分析
        ma_droo_delay = pd.DataFrame(all_runs_droo_delay.T).rolling(window=MA_WINDOW, min_periods=1).mean().values.T
        ma_top_l_delay = pd.DataFrame(all_runs_top_l_delay.T).rolling(window=MA_WINDOW, min_periods=1).mean().values.T
        mean_droo_delay, std_droo_delay = np.mean(ma_droo_delay, axis=0), np.std(ma_droo_delay, axis=0)
        mean_top_l_delay, std_top_l_delay = np.mean(ma_top_l_delay, axis=0), np.std(ma_top_l_delay, axis=0)

        # Loss 分析
        ma_droo_loss = pd.DataFrame(all_runs_droo_loss.T).rolling(window=MA_WINDOW, min_periods=1).mean().values.T
        ma_top_l_loss = pd.DataFrame(all_runs_top_l_loss.T).rolling(window=MA_WINDOW, min_periods=1).mean().values.T
        mean_droo_loss, std_droo_loss = np.mean(ma_droo_loss, axis=0), np.std(ma_droo_loss, axis=0)
        mean_top_l_loss, std_top_l_loss = np.mean(ma_top_l_loss, axis=0), np.std(ma_top_l_loss, axis=0)

        episodes_x = np.arange(1, EPISODES + 1)
        
        col_fig1, col_fig2 = st.columns(2)
        
        with col_fig1:
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(episodes_x, mean_droo_delay, label="DROO (OPQ) 延遲", color='#FF4B4B')
            ax1.fill_between(episodes_x, mean_droo_delay - std_droo_delay, mean_droo_delay + std_droo_delay, color='#FF4B4B', alpha=0.15)
            ax1.plot(episodes_x, mean_top_l_delay, label=f"TOP-L 延遲 (L={L_PARAM})", color='#0068C9')
            ax1.fill_between(episodes_x, mean_top_l_delay - std_top_l_delay, mean_top_l_delay + std_top_l_delay, color='#0068C9', alpha=0.2)
            ax1.set_title("加權總延遲收斂", fontweight='bold')
            ax1.legend()
            ax1.grid(True, linestyle=':', alpha=0.7)
            st.pyplot(fig1)

        with col_fig2:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.plot(episodes_x, mean_droo_loss, label="DROO (OPQ) Loss", color='#FF4B4B')
            ax2.fill_between(episodes_x, mean_droo_loss - std_droo_loss, mean_droo_loss + std_droo_loss, color='#FF4B4B', alpha=0.15)
            ax2.plot(episodes_x, mean_top_l_loss, label=f"TOP-L Loss (L={L_PARAM})", color='#0068C9')
            ax2.fill_between(episodes_x, mean_top_l_loss - std_top_l_loss, mean_top_l_loss + std_top_l_loss, color='#0068C9', alpha=0.2)
            ax2.set_title("神經網路訓練 Loss 收斂", fontweight='bold')
            ax2.legend()
            ax2.grid(True, linestyle=':', alpha=0.7)
            st.pyplot(fig2)
        
        # 關閉 Matplotlib 資源，防止記憶體洩漏
        plt.close('all')

        st.markdown("---")
        st.subheader("📥 實驗數據下載區")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            summary_df = pd.DataFrame({
                "Episode": episodes_x,
                "DROO_Delay_Mean": mean_droo_delay, "DROO_Delay_Std": std_droo_delay,
                "TOP_L_Delay_Mean": mean_top_l_delay, "TOP_L_Delay_Std": std_top_l_delay,
                "DROO_Loss_Mean": mean_droo_loss, "DROO_Loss_Std": std_droo_loss,
                "TOP_L_Loss_Mean": mean_top_l_loss, "TOP_L_Loss_Std": std_top_l_loss
            })
            zip_file.writestr("Robustness_Summary.csv", summary_df.to_csv(index=False).encode('utf-8-sig'))
            
            img_buffer_delay = io.BytesIO()
            fig1.savefig(img_buffer_delay, format="png", dpi=300, bbox_inches="tight")
            zip_file.writestr("Robustness_Delay_Shaded.png", img_buffer_delay.getvalue())

            img_buffer_loss = io.BytesIO()
            fig2.savefig(img_buffer_loss, format="png", dpi=300, bbox_inches="tight")
            zip_file.writestr("Robustness_Loss_Shaded.png", img_buffer_loss.getvalue())
            
            for idx, seed in enumerate(seeds):
                run_df = pd.DataFrame({
                    "Episode": episodes_x,
                    "DROO_Delay": all_runs_droo_delay[idx], "TOP_L_Delay": all_runs_top_l_delay[idx],
                    "DROO_Loss": all_runs_droo_loss[idx], "TOP_L_Loss": all_runs_top_l_loss[idx]
                })
                zip_file.writestr(f"Raw_Data_Run_{idx+1}_Seed_{seed}.csv", run_df.to_csv(index=False).encode('utf-8-sig'))
                
        zip_buffer.seek(0)
        st.download_button(
            label="📦 一鍵打包下載所有實驗數據與分析圖 (.zip)",
            data=zip_buffer,
            file_name=f"MEC_Robustness_Experiment_N{N}.zip",
            mime="application/zip"
        )

# ------------------------------------------
# 模式 3: L參數敏感度分析模式 (Elbow Curve) [📦 新增模塊]
# ------------------------------------------
elif demo_mode == "🔍 L參數敏感度分析模式 (Elbow Curve)":
    
    try:
        L_list = [int(x.strip()) for x in l_input.split(",")]
        L_list.sort() 
    except ValueError:
        st.error("⚠️ L 值輸入格式錯誤。")
        st.stop()

    if st.sidebar.button("🚀 啟動參數敏感度分析", type="primary"):
        st.markdown("### 📡 現場即時訓練監控站")
        progress_bar = st.progress(0)
        l_status_text = st.empty()       
        
        # 數值看板
        col1, col2 = st.columns(2)
        metric_droo = col1.empty()
        metric_topl = col2.empty()
        
        st.markdown("#### 📉 當前 L 參數延遲收斂")
        realtime_chart_placeholder_delay = st.empty() 
        st.markdown("#### 📉 當前 L 參數訓練 Loss")
        realtime_chart_placeholder_loss = st.empty() 

        completed_runs_container = st.container()

        final_results_top_l = []
        final_results_droo = []
        all_dfs = {} 

        for idx, current_L in enumerate(L_list):
            l_status_text.markdown(f"**🔄 正在訓練參數 L = {current_L} ({idx + 1} / {len(L_list)})...**")
            
            tf.random.set_seed(42)
            np.random.seed(42)
            random.seed(42)

            droo_model = DRLNetwork(N, [80, 64], N)
            top_l_model = DRLNetwork(N, [80, 64], N)
            _ = droo_model(tf.zeros((1, N)))
            _ = top_l_model(tf.zeros((1, N)))
            droo_model.set_weights(top_l_model.get_weights())

            optimizer_droo = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
            optimizer_top_l = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

            replay_memory_droo = deque(maxlen=REPLAY_MEMORY_SIZE)
            replay_memory_top_l = deque(maxlen=REPLAY_MEMORY_SIZE)

            current_history = {
                "Episode": [], "DROO (OPQ)": [], f"TOP-L (L={current_L})": [],
                "DROO_Loss": [], f"TOP-L_Loss (L={current_L})": []
            }

            cur_loss_droo = 0.0
            cur_loss_top_l = 0.0

            for episode in range(1, EPISODES + 1):
                r = np.random.uniform(r_min, r_max, N)
                r_tensor = tf.convert_to_tensor([r], dtype=tf.float32)

                # DROO
                droo_soft_x = droo_model(r_tensor).numpy().flatten()
                droo_candidates = order_preserving_quantization(droo_soft_x, K)
                droo_best = min(droo_candidates, key=lambda x: compute_total_delay(x, resource_allocation(q, x), q, r, f_s, f_i))
                droo_delay = compute_total_delay(droo_best, resource_allocation(q, droo_best), q, r, f_s, f_i)
                replay_memory_droo.append((r, droo_best))

                # TOP-L
                top_l_soft_x = top_l_model(r_tensor).numpy().flatten()
                top_l_candidates = top_l_candidate_generation(top_l_soft_x, current_L)
                top_l_best = min(top_l_candidates, key=lambda x: compute_total_delay(x, resource_allocation(q, x), q, r, f_s, f_i))
                top_l_delay = compute_total_delay(top_l_best, resource_allocation(q, top_l_best), q, r, f_s, f_i)
                replay_memory_top_l.append((r, top_l_best))

                if len(replay_memory_droo) >= BATCH_SIZE and (episode % TRAIN_INTERVAL == 0):
                    batch_droo = random.sample(replay_memory_droo, BATCH_SIZE)
                    r_batch, x_batch = zip(*batch_droo)
                    with tf.GradientTape() as tape:
                        preds = droo_model(tf.convert_to_tensor(np.array(r_batch), dtype=tf.float32), training=True)
                        loss_val = loss_function(tf.convert_to_tensor(np.array(x_batch), dtype=tf.float32), preds)
                    optimizer_droo.apply_gradients(zip(tape.gradient(loss_val, droo_model.trainable_variables), droo_model.trainable_variables))
                    cur_loss_droo = loss_val.numpy()

                    batch_top_l = random.sample(replay_memory_top_l, BATCH_SIZE)
                    r_batch, x_batch = zip(*batch_top_l)
                    with tf.GradientTape() as tape:
                        preds = top_l_model(tf.convert_to_tensor(np.array(r_batch), dtype=tf.float32), training=True)
                        loss_val = loss_function(tf.convert_to_tensor(np.array(x_batch), dtype=tf.float32), preds)
                    optimizer_top_l.apply_gradients(zip(tape.gradient(loss_val, top_l_model.trainable_variables), top_l_model.trainable_variables))
                    cur_loss_top_l = loss_val.numpy()

                current_history["Episode"].append(episode)
                current_history["DROO (OPQ)"].append(droo_delay)
                current_history[f"TOP-L (L={current_L})"].append(top_l_delay)
                current_history["DROO_Loss"].append(cur_loss_droo)
                current_history[f"TOP-L_Loss (L={current_L})"].append(cur_loss_top_l)

                if episode % UPDATE_INTERVAL == 0 or episode == EPISODES:
                    current_progress = (idx * EPISODES + episode) / (len(L_list) * EPISODES)
                    progress_bar.progress(current_progress)
                    gc.collect()
                    
                    df_current = pd.DataFrame(current_history).set_index("Episode")
                    df_ma = df_current.rolling(window=MA_WINDOW, min_periods=1).mean()
                    
                    plot_step = max(1, len(df_ma) // 300)
                    df_ma_plot = df_ma.iloc[::plot_step]
                    
                    realtime_chart_placeholder_delay.line_chart(df_ma_plot[["DROO (OPQ)", f"TOP-L (L={current_L})"]])
                    realtime_chart_placeholder_loss.line_chart(df_ma_plot[["DROO_Loss", f"TOP-L_Loss (L={current_L})"]])
                    
                    metric_droo.metric("DROO 延遲", f"{df_ma['DROO (OPQ)'].iloc[-1]:.2f}")
                    metric_topl.metric(f"TOP-L (L={current_L}) 延遲", f"{df_ma[f'TOP-L (L={current_L})'].iloc[-1]:.2f}")

            calc_window = min(CALC_WINDOW, EPISODES)
            converged_top_l = df_current[f"TOP-L (L={current_L})"].iloc[-calc_window:].mean()
            converged_droo = df_current["DROO (OPQ)"].iloc[-calc_window:].mean()
            
            final_results_top_l.append(converged_top_l)
            final_results_droo.append(converged_droo)
            all_dfs[current_L] = df_current.reset_index()

            with completed_runs_container.expander(f"✅ L={current_L} 訓練完成 (末 {calc_window} 步平均延遲: {converged_top_l:.2f})", expanded=False):
                st.write("**延遲曲線**")
                st.line_chart(df_ma[["DROO (OPQ)", f"TOP-L (L={current_L})"]].iloc[::plot_step])
                st.write("**Loss 曲線**")
                st.line_chart(df_ma[["DROO_Loss", f"TOP-L_Loss (L={current_L})"]].iloc[::plot_step])

            realtime_chart_placeholder_delay.empty()
            realtime_chart_placeholder_loss.empty()
            
            tf.keras.backend.clear_session()
            gc.collect()

        st.markdown("---")
        st.markdown("### 📈 參數敏感度分析 (Elbow Curve)")
        
        baseline_droo = np.mean(final_results_droo)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(L_list, final_results_top_l, marker='o', markersize=8, color='#0068C9', linewidth=2, label="TOP-L 最終收斂延遲")
        ax.axhline(y=baseline_droo, color='#FF4B4B', linewidth=2, label=f"DROO 基準線 (平均延遲={baseline_droo:.2f})")
        ax.set_title(f"TOP-L 策略之探索數量 (L) 參數敏感度分析 (N={N})", fontsize=14, fontweight='bold')
        ax.set_xlabel("探索數量 L (位元翻轉個數)", fontsize=12)
        ax.set_ylabel(f"加權總延遲 (末 {calc_window} 步平均)", fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            summary_df = pd.DataFrame({"L_Value": L_list, "TOP_L_Converged_Delay": final_results_top_l, "DROO_Converged_Delay": final_results_droo})
            zip_file.writestr(f"Ablation_Study_Summary_N{N}.csv", summary_df.to_csv(index=False).encode('utf-8-sig'))
            
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
            zip_file.writestr(f"Ablation_Elbow_Curve_N{N}.png", img_buffer.getvalue())
            
            for L_val, df in all_dfs.items():
                zip_file.writestr(f"Convergence_Data_N{N}_L{L_val}.csv", df.to_csv(index=False).encode('utf-8-sig'))
                
        zip_buffer.seek(0)
        st.download_button(
            label="📦 一鍵打包下載分析數據與 Elbow Curve (.zip)",
            data=zip_buffer,
            file_name=f"MEC_Ablation_Study_N{N}.zip",
            mime="application/zip"
        )

# ------------------------------------------
# 模式 4: 預算數據解析模式 (Load CSV)
# ------------------------------------------
elif demo_mode == "📊 預算數據解析模式 (載入完整大數據)":
    
    tab1, tab2 = st.tabs(["🛡️ 演算法強健性與變異數分析 (多組亂數種子)", "📈 單次極限收斂分析 (17,500步)"])
    
    with tab1:
        st.markdown("### 演算法強健性分析 (平均值 $\pm 1\sigma$ 標準差)")
        uploaded_robust = st.file_uploader("📂 上傳多種子實驗數據 (需包含 Loss_Mean 欄位)", type=['csv'], key="multi_run")
        if uploaded_robust is not None:
            df_robust = pd.read_csv(uploaded_robust)
            episodes_x = df_robust["Episode"]
            
            # Delay
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            if "DROO_Delay_Mean" in df_robust.columns: # 新版
                ax1.plot(episodes_x, df_robust["DROO_Delay_Mean"], label="DROO (OPQ)", color='#FF4B4B')
                ax1.fill_between(episodes_x, df_robust["DROO_Delay_Mean"] - df_robust["DROO_Delay_Std"], df_robust["DROO_Delay_Mean"] + df_robust["DROO_Delay_Std"], color='#FF4B4B', alpha=0.15)
                ax1.plot(episodes_x, df_robust["TOP_L_Delay_Mean"], label="TOP-L", color='#0068C9')
                ax1.fill_between(episodes_x, df_robust["TOP_L_Delay_Mean"] - df_robust["TOP_L_Delay_Std"], df_robust["TOP_L_Delay_Mean"] + df_robust["TOP_L_Delay_Std"], color='#0068C9', alpha=0.2)
            else: # 舊版相容
                ax1.plot(episodes_x, df_robust["DROO_Mean"], label="DROO (OPQ)", color='#FF4B4B')
                ax1.fill_between(episodes_x, df_robust["DROO_Mean"] - df_robust["DROO_Std"], df_robust["DROO_Mean"] + df_robust["DROO_Std"], color='#FF4B4B', alpha=0.15)
                ax1.plot(episodes_x, df_robust["TOP_L_Mean"], label="TOP-L", color='#0068C9')
                ax1.fill_between(episodes_x, df_robust["TOP_L_Mean"] - df_robust["TOP_L_Std"], df_robust["TOP_L_Mean"] + df_robust["TOP_L_Std"], color='#0068C9', alpha=0.2)
            
            ax1.set_title("加權總延遲收斂之強健性分析", fontweight='bold')
            ax1.legend()
            st.pyplot(fig1)

            # Loss
            if "DROO_Loss_Mean" in df_robust.columns:
                st.markdown("### 神經網路訓練 Loss 強健性分析")
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.plot(episodes_x, df_robust["DROO_Loss_Mean"], label="DROO (OPQ) Loss", color='#FF4B4B')
                ax2.fill_between(episodes_x, df_robust["DROO_Loss_Mean"] - df_robust["DROO_Loss_Std"], df_robust["DROO_Loss_Mean"] + df_robust["DROO_Loss_Std"], color='#FF4B4B', alpha=0.15)
                ax2.plot(episodes_x, df_robust["TOP_L_Loss_Mean"], label="TOP-L Loss", color='#0068C9')
                ax2.fill_between(episodes_x, df_robust["TOP_L_Loss_Mean"] - df_robust["TOP_L_Loss_Std"], df_robust["TOP_L_Loss_Mean"] + df_robust["TOP_L_Loss_Std"], color='#0068C9', alpha=0.2)
                ax2.set_title("訓練 Loss 收斂之強健性分析", fontweight='bold')
                ax2.legend()
                st.pyplot(fig2)
            
            plt.close('all')

    with tab2:
        st.markdown("### 單次完整訓練收斂過程")
        uploaded_single = st.file_uploader("📂 上傳單次實驗數據 (需包含 Loss 欄位)", type=['csv'], key="single_run")
        if uploaded_single is not None:
            df_single = pd.read_csv(uploaded_single)
            df_single_ma = df_single.set_index("Episode").rolling(window=500, min_periods=1).mean()
            
            # 使用一樣的降採樣技術來顯示上傳的龐大資料集
            plot_step = max(1, len(df_single_ma) // 400)
            
            st.write("**延遲曲線**")
            cols_delay = [c for c in ["DROO (OPQ)", "TOP-L", "Random", "UserBased"] if c in df_single_ma.columns]
            st.line_chart(df_single_ma[cols_delay].iloc[::plot_step])

            st.write("**Loss 曲線**")
            cols_loss = [c for c in df_single_ma.columns if "Loss" in c]
            if cols_loss:
                st.line_chart(df_single_ma[cols_loss].iloc[::plot_step])