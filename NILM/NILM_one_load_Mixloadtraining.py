import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import Input
from keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Reshape, Multiply, GlobalAveragePooling1D, Dropout, BatchNormalization, MultiHeadAttention, Add, Bidirectional, LSTM, Concatenate, LayerNormalization
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard
import tensorflow.keras.backend as K
import gc
import time
from sklearn.metrics.pairwise import cosine_similarity

# 開始計時
start_time = time.time()
print('負載辨識開始執行=============================================')

# 設置參數 =============================================================================================
Model_batch_size = 16 
model_large_epochs = 100
ae_epochs = 100
latent_dim = 32
gan_epoch = 100
batch_size = 32
n_critic = 5
gp_weight = 1
# ======================================================================================================

# 路徑設定
class_1_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 1'  
new_appliance_input_folder = r"D:\graduate_info\Research\code\lab load\new_appliance_output_folder_one_load_combine"

# 讀取 new_appliance_input_folder 內 CSV 檔（示範用）
new_appliance_data = {}
if os.listdir(new_appliance_input_folder):
    for file_name in os.listdir(new_appliance_input_folder):
        appliance_name = file_name.replace(".csv", "")
        file_path = os.path.join(new_appliance_input_folder, file_name)
        device_data = pd.read_csv(file_path)
        new_appliance_data[appliance_name] = device_data
    print(f"已成功讀取 {len(new_appliance_data)} 個設備的 CSV 檔並存入字典。")
else:
    print(f"資料夾 {new_appliance_input_folder} 為空，無法處理設備數據。")

# 資料前處理相關函式
def z_score_normalize(data):
    return (data - np.mean(data)) / (np.std(data) + 1e-8)

def processing(features_HHT_list, labels_list, features_list, name_list, features_DWT_list, delt_features_list):
    HHT_features = np.concatenate(features_HHT_list, axis=0)
    DWT_features = np.concatenate(features_DWT_list, axis=0)
    features = np.concatenate(features_list, axis=0)
    print("HHT_features:", HHT_features.shape)
    print("DWT_features:", DWT_features.shape)
    print("features:", features.shape)
    delt_features = np.concatenate(delt_features_list, axis=0)
    name = np.concatenate(name_list, axis=0)
    large_labels = np.concatenate(labels_list, axis=0)
    HHT_features = np.expand_dims(HHT_features, axis=-1)
    DWT_features = np.expand_dims(DWT_features, axis=-1)
    features = np.expand_dims(features, axis=-1)
    delt_features = np.expand_dims(delt_features, axis=-1)
    name = np.expand_dims(name, axis=-1)

    features = z_score_normalize(features)
    DWT_features = z_score_normalize(DWT_features)
    HHT_features = z_score_normalize(HHT_features)

    juice_feature = features[:815]
    steam_feature = features[815:]
    
    juice_HHT = HHT_features[:815]
    steam_HHT = HHT_features[815:]

    juice_DWT = DWT_features[:815]
    steam_DWT = DWT_features[815:]

    target_size = features.shape[0]

    if len(juice_feature) < target_size:
        juice_feature = np.random.choice(juice_feature.flatten(), size=(target_size, juice_feature.shape[1], juice_feature.shape[2]), replace=True)
    
    if len(steam_feature) < target_size:
        steam_feature = np.random.choice(steam_feature.flatten(), size=(target_size, steam_feature.shape[1], steam_feature.shape[2]), replace=True)
    
    if len(juice_HHT) < target_size:
        juice_HHT = np.random.choice(juice_HHT.flatten(), size=(target_size, juice_HHT.shape[1], juice_HHT.shape[2]), replace=True)
    
    if len(steam_HHT) < target_size:
        steam_HHT = np.random.choice(steam_HHT.flatten(), size=(target_size, steam_HHT.shape[1], steam_HHT.shape[2]), replace=True)

    if len(juice_DWT) < target_size:
        juice_DWT = np.random.choice(juice_DWT.flatten(), size=(target_size, juice_DWT.shape[1], juice_DWT.shape[2]), replace=True)
    
    if len(steam_DWT) < target_size:
        steam_DWT = np.random.choice(steam_DWT.flatten(), size=(target_size, steam_DWT.shape[1], steam_DWT.shape[2]), replace=True)

    print("🚀 擴增後的 juice_feature shape:", juice_feature.shape)
    print("🚀 擴增後的 steam_feature shape:", steam_feature.shape)
    
    print("🚀 擴增後的 juice_HHT shape:", juice_HHT.shape)
    print("🚀 擴增後的 steam_HHT shape:", steam_HHT.shape)
    
    print("🚀 擴增後的 juice_DWT shape:", juice_DWT.shape)
    print("🚀 擴增後的 steam_DWT shape:", steam_DWT.shape)
    
    print("正規化完成")
    print("large_labels shape before processing:", large_labels.shape)
    print("🔍 Augmented `y_train`: (前 10 筆)")
    print(large_labels[:10])
    
    load_one_hot = to_categorical(name)

    all_indices = np.arange(len(HHT_features))
    train_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=1)
    
    juice_feature_train = juice_feature[train_idx]
    steam_feature_train = steam_feature[train_idx]
    juice_feature_test = juice_feature[test_idx]
    steam_feature_test = steam_feature[test_idx]

    juice_HHT_train = juice_HHT[train_idx]
    steam_HHT_train = steam_HHT[train_idx]
    juice_HHT_test = juice_HHT[test_idx]
    steam_HHT_test = steam_HHT[test_idx]

    juice_DWT_train = juice_DWT[train_idx]
    steam_DWT_train = steam_DWT[train_idx]
    juice_DWT_test = juice_DWT[test_idx]
    steam_DWT_test = steam_DWT[test_idx]

    x_HHT_train = HHT_features[train_idx]
    x_HHT_test  = HHT_features[test_idx]
    x_DWT_train = DWT_features[train_idx]
    x_DWT_test  = DWT_features[test_idx]
    x_features_train = features[train_idx]
    x_features_test  = features[test_idx]

    x_delt_features_train = delt_features[train_idx]
    x_delt_features_test  = delt_features[test_idx]

    if x_delt_features_train.shape[1] < x_features_train.shape[1]:
        pad_size = x_features_train.shape[1] - x_delt_features_train.shape[1]
        x_delt_features_train = np.pad(x_delt_features_train, ((0, 0), (0, pad_size), (0, 0)), mode='constant')
        x_delt_features_test = np.pad(x_delt_features_test, ((0, 0), (0, pad_size), (0, 0)), mode='constant')

    print("🚀 修正後的 x_delt_features_train shape:", x_delt_features_train.shape)
    print("🚀 修正後的 x_delt_features_test shape:", x_delt_features_test.shape)

    y_train = large_labels[train_idx]
    y_test  = large_labels[test_idx]
    y_train_load = load_one_hot[train_idx]
    y_test_load  = load_one_hot[test_idx]
    
    return (x_HHT_train, x_HHT_test, x_DWT_train, x_DWT_test, x_features_train, x_features_test, 
            y_train, y_test, y_train_load, y_test_load, x_delt_features_train, x_delt_features_test,
            juice_feature_train,steam_feature_train,juice_feature_test,steam_feature_test,juice_HHT_train,steam_HHT_train,juice_HHT_test,
            steam_HHT_test,juice_DWT_train,steam_DWT_train,juice_DWT_test,steam_DWT_test)

def process(folder_path, class_name, features_HHT_list, labels_list, features_list, delt_features_list, name_list, features_DWT_list):
    if not os.listdir(folder_path):
        print(f"資料夾 {folder_path} 為空，跳過處理。")
        return 
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            csv_data = pd.read_csv(file_path, header=None)
            class_name.append(csv_data)
    print(f"已成功處理資料夾 {folder_path}。")
    for file_name in sorted(os.listdir(folder_path)):
        if 'HHT_features' in file_name:
            HHT_large_features = np.load(os.path.join(folder_path, file_name))
            features_HHT_list.append(HHT_large_features)
        if 'labels' in file_name:
            labels_large = np.load(os.path.join(folder_path, file_name))
            labels_list.append(labels_large)
        if 'load_features' in file_name:
            class_appliance_features = np.load(os.path.join(folder_path, file_name))
            features_list.append(class_appliance_features)
        if 'load_name' in file_name:
            class_appliance_name = np.load(os.path.join(folder_path, file_name))
            name_list.append(class_appliance_name)
        if 'DWT_features' in file_name:
            DWT_class_feature = np.load(os.path.join(folder_path, file_name))
            features_DWT_list.append(DWT_class_feature)
        if 'load_delt_features' in file_name:
            class_appliance_delt_features = np.load(os.path.join(folder_path, file_name))
            delt_features_list.append(class_appliance_delt_features)
    return processing(features_HHT_list, labels_list, features_list, name_list, features_DWT_list, delt_features_list)

# 讀取 class1 資料（示範用，其他類別同理）
(Class1_HHT_train, Class1_HHT_test,  Class1_DWT_train, Class1_DWT_test, 
 Class1_features_train, Class1_features_test,  
 Class1_y_train, Class1_y_test, Class1_y_train_load, 
 Class1_y_test_load, Class1_delt_features_train, Class1_delt_features_test,
 juice_train,steam_train,juice_test,steam_test,juice_HHT_train,steam_HHT_train,juice_HHT_test,
 steam_HHT_test,juice_DWT_train,steam_DWT_train,juice_DWT_test,steam_DWT_test) = process(
    class_1_output_dir, [], [], [], [], [], [], []
)

# -------------------- 其他前處理 / 讀取資料等函式 --------------------
def z_score_normalize(data):
    return (data - np.mean(data)) / (np.std(data) + 1e-8)

def data_generator(HHT_data, DWT_data, fused_features_data, delt_features_data, labels, class_weights, batch_size):
    # ... 略，保持你原本的實作 ...
    num_samples = len(labels)
    if not isinstance(labels, (np.ndarray, list)):
        print(f"⚠️ labels 不是 numpy 陣列，而是 {type(labels)}，轉換中...")
        labels = np.array(labels)
    print(f"✅ labels shape: {labels.shape}, type: {type(labels)}")
    
    while True:
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            yield ([HHT_data[start_idx:end_idx],
                    DWT_data[start_idx:end_idx],
                    fused_features_data[start_idx:end_idx],
                    delt_features_data[start_idx:end_idx]], 
                   labels[start_idx:end_idx])

def compute_class_weight_multilabel(y_train):
    # ... 略，保持你原本的實作 ...
    num_samples, num_classes = y_train.shape
    label_counts = np.sum(y_train, axis=0)
    total_samples = len(y_train)
    class_weights = {i: total_samples / (num_classes * count) for i, count in enumerate(label_counts)}
    return class_weights

def find_best_threshold(y_true, y_probs):
    # ... 略，保持你原本的實作 ...
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_threshold

def fusion_block(input1, input2, name_prefix=""):
    # 注意力分數 (使用 input1 去 attend input2)
    attention = MultiHeadAttention(num_heads=2, key_dim=16, name=f"{name_prefix}_attn")(input1, input2)
    attention = LayerNormalization(name=f"{name_prefix}_norm")(attention)
    fusion = Add(name=f"{name_prefix}_add")([input1, attention])
    return fusion

def weighted_fusion(input1, input2, alpha, name_prefix):
    seq_len = input1.shape[1]
    feat_dim = input1.shape[2]

    def tile_alpha(a):
        a = tf.expand_dims(a, axis=1)  # (batch, 1, 1)
        return tf.tile(a, [1, seq_len, feat_dim])  # (batch, seq_len, feat_dim)

    alpha_tile = Lambda(tile_alpha, name=f"{name_prefix}_alpha_tile")(alpha)
    
    return Add(name=f"{name_prefix}_fused")([
        Multiply()([alpha_tile, input1]),
        Multiply()([1.0 - alpha_tile, input2])
    ])
# -------------------- 模型定義區 --------------------
from tensorflow.keras.layers import Input, Lambda,Conv1D, LayerNormalization, Add, Flatten, Dense, Reshape, LeakyReLU, Dropout, Concatenate, GaussianNoise

def build_generator(feat_shape, hht_shape, dwt_shape):
    from tensorflow.keras.layers import Input, Conv1D, GlobalAveragePooling1D, Dense, Dropout, LayerNormalization, Add, Multiply, Lambda, RepeatVector, Concatenate, Reshape
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import MultiHeadAttention

    def fusion_block(input1, input2, name_prefix=""):
        attn = MultiHeadAttention(num_heads=2, key_dim=16, name=f"{name_prefix}_attn")(input1, input2)
        norm = LayerNormalization(name=f"{name_prefix}_norm")(attn)
        return Add(name=f"{name_prefix}_add")([input1, norm])

    def reduce_dim_block(x, name_prefix):
        x = Conv1D(64, kernel_size=7, strides=7, padding='same', activation='relu', name=f'{name_prefix}_conv1')(x)
        x = Conv1D(32, kernel_size=5, strides=5, padding='same', activation='relu', name=f'{name_prefix}_conv2')(x)
        return x

    def cnn_branch(x, name):
        x = Conv1D(64, 3, padding='same', activation='relu', name=f'{name}_conv1')(x)
        x = Conv1D(128, 3, padding='same', activation='relu', name=f'{name}_conv2')(x)
        return GlobalAveragePooling1D(name=f'{name}_gap')(x)

    # === Input ===
    feat_j = Input(shape=feat_shape, name='feat_juice')
    feat_s = Input(shape=feat_shape, name='feat_steam')
    HHT_j  = Input(shape=hht_shape, name='HHT_juice')
    HHT_s  = Input(shape=hht_shape, name='HHT_steam')
    DWT_j  = Input(shape=dwt_shape, name='DWT_juice')
    DWT_s  = Input(shape=dwt_shape, name='DWT_steam')
    alpha_input = Input(shape=(1,), name='alpha')

    # === Interpolation (原始特徵) ===
    alpha_tile = Lambda(lambda a: tf.tile(a[..., None], [1, feat_shape[0], 1]))(alpha_input)
    feat_concat = Concatenate(axis=-1)([feat_j, feat_s, alpha_tile])
    mask = Conv1D(1, kernel_size=3, padding='same', activation='sigmoid', name='learned_alpha_mask')(feat_concat)
    feat_cat = mask * feat_j + (1 - mask) * feat_s

    # Mid-supervision Target
    mid_out = Conv1D(1, kernel_size=1, padding='same', name='mid_interp_out')(feat_cat)

    x = Conv1D(64, kernel_size=5, padding='same', activation='relu')(feat_cat)
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    feat_interp = Conv1D(64, kernel_size=1, padding='same')(x)

    HHT_j_r = reduce_dim_block(HHT_j, "HHT_j")
    HHT_s_r = reduce_dim_block(HHT_s, "HHT_s")
    DWT_j_r = reduce_dim_block(DWT_j, "DWT_j")
    DWT_s_r = reduce_dim_block(DWT_s, "DWT_s")

    fused_hht = weighted_fusion(HHT_j_r, HHT_s_r, alpha_input, "fused_hht")
    fused_dwt = weighted_fusion(DWT_j_r, DWT_s_r, alpha_input, "fused_dwt")

    x_hht = cnn_branch(fused_hht, "x_fused_hht")
    x_dwt = cnn_branch(fused_dwt, "x_fused_dwt")
    x_feat_interp = cnn_branch(feat_interp, "feat_interp")

    alpha_dense = Dense(32, activation='relu')(alpha_input)

    merged = Concatenate()([x_feat_interp, x_hht, x_dwt, alpha_dense])
    merged = Dense(256, activation='relu')(merged)
    merged = Dropout(0.3)(merged)
    merged = LayerNormalization()(merged)
    merged = Dense(128, activation='relu')(merged)
    merged = Dropout(0.3)(merged)
    merged = LayerNormalization()(merged)

    alpha_pred = Dense(1, activation='sigmoid', name='alpha_pred')(merged)

    seq_len = feat_shape[0]
    x_out = Dense(128, activation='relu')(merged)
    x_out = RepeatVector(seq_len)(x_out)
    x_out = Conv1D(1, kernel_size=1, name="generated_mixed")(x_out)

    x_feat_j = cnn_branch(feat_j, "feat_j")
    x_feat_s = cnn_branch(feat_s, "feat_s")

    recon_j = Dense(feat_shape[0], activation='relu')(x_feat_j)
    recon_j = RepeatVector(feat_shape[0])(recon_j)
    recon_j = Conv1D(1, kernel_size=1, name='recon_feat_juice')(recon_j)

    recon_s = Dense(feat_shape[0], activation='relu')(x_feat_s)
    recon_s = RepeatVector(feat_shape[0])(recon_s)
    recon_s = Conv1D(1, kernel_size=1, name='recon_feat_steam')(recon_s)

    hht_len = hht_shape[0]
    dwt_len = dwt_shape[0]

    recon_hht = Dense(hht_len, activation='relu', name='recon_hht')(merged)
    recon_hht = RepeatVector(1)(recon_hht)
    recon_hht = Reshape((hht_len, 1))(recon_hht)

    recon_dwt = Dense(dwt_len, activation='relu', name='recon_dwt')(merged)
    recon_dwt = RepeatVector(1)(recon_dwt)
    recon_dwt = Reshape((dwt_len, 1))(recon_dwt)

    return Model(
        inputs=[feat_j, feat_s, HHT_j, HHT_s, DWT_j, DWT_s, alpha_input],
        outputs=[x_out, mid_out, recon_j, recon_s, recon_hht, recon_dwt, alpha_pred],
        name="GeneratorFusionCycle"
    )



from tensorflow.keras.layers import Dropout

def build_critic(feature_shape):
    inp = Input(shape=(feature_shape[0], 1))
    # ... 你的 CNN 卷積
    x = Conv1D(64, kernel_size=3, padding="same")(inp)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1D(128, kernel_size=3, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = LayerNormalization()(x)
    x = Flatten()(x)

    # 用 Dense(1) 作為 WGAN real/fake 的 linear output
    wgan_out = Dense(1, activation=None, name='wgan_out')(x)

    # 用 Dense(num_classes) 作為 multi‐class
    class_out = Dense(1, activation='softmax', name='class_out')(x)

    return Model(inp, [wgan_out, class_out], name="MultiCritic")

# 3. VAE：先訓練以取得 encoder 作為 latent 映射器
def build_vae(input_shape, latent_dim=32, name_prefix="VAE"):
    inputs = Input(shape=input_shape, name=f"{name_prefix}_input")
    x = Conv1D(128, 5, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(64, 5, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    mu = Dense(latent_dim, name=f"{name_prefix}_mu")(x)
    encoder = Model(inputs, mu, name=f"{name_prefix}_encoder")
    
    decoder_input = Input(shape=(latent_dim,), name=f"{name_prefix}_decoder_input")
    x = Dense(input_shape[0] * 1, activation='relu')(decoder_input)
    x = Reshape((input_shape[0], 1))(x)
    x = Conv1D(32, 5, activation='relu', padding='same')(x)
    x = Conv1D(64, 5, activation='relu', padding='same')(x)
    decoded_output = Conv1D(1, 5, activation='sigmoid', padding='same')(x)
    decoder = Model(decoder_input, decoded_output, name=f"{name_prefix}_decoder")
    
    vae_outputs = decoder(encoder(inputs))
    vae = Model(inputs, vae_outputs, name=f"{name_prefix}_vae")
    vae.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
    return vae, encoder, decoder

# 清除 GPU 記憶體並設定動態增長
tf.keras.backend.clear_session()
gc.collect()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ 設定 GPU {gpu} 記憶體動態增長")
    except RuntimeError as e:
        print(e)

# 4. 數據生成器
def data_generator(HHT_data, DWT_data, fused_features_data, delt_features_data, labels, class_weights, batch_size):
    num_samples = len(labels)
    if not isinstance(labels, (np.ndarray, list)):
        print(f"⚠️ labels 不是 numpy 陣列，而是 {type(labels)}，轉換中...")
        labels = np.array(labels)
    print(f"✅ labels shape: {labels.shape}, type: {type(labels)}")
    
    while True:
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
         
            yield ([HHT_data[start_idx:end_idx], DWT_data[start_idx:end_idx], 
                    fused_features_data[start_idx:end_idx], delt_features_data[start_idx:end_idx],], 
                    labels[start_idx:end_idx])

def compute_class_weight_multilabel(y_train):
    num_samples, num_classes = y_train.shape
    label_counts = np.sum(y_train, axis=0)
    total_samples = len(y_train)
    class_weights = {i: total_samples / (num_classes * count) for i, count in enumerate(label_counts)}
    return class_weights

from sklearn.metrics import precision_recall_curve
def find_best_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_threshold

# -------------------- WGAN-GP 損失函式 --------------------
def wasserstein_loss_critic(real_score, fake_score):
    """
    Critic 的原始 WGAN Loss: D_loss = E[fake_score] - E[real_score]
    """
    return tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)

def wasserstein_loss_gen(fake_score):
    """
    Generator 的 WGAN Loss: G_loss = - E[fake_score]
    """
    return -tf.reduce_mean(fake_score)

def gradient_penalty(critic, real_samples, fake_samples):
    """
    real_samples, fake_samples shape: (batch_size, seq_len, 1)
    """
    alpha = tf.random.uniform([tf.shape(real_samples)[0], 1, 1], 0.0, 1.0)
    interpolates = alpha * real_samples + (1.0 - alpha) * fake_samples

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolates)
        # Critic 對 interpolates 的評分
        pred = critic(interpolates, training=True)
    grads = gp_tape.gradient(pred, [interpolates])[0]  # (batch_size, seq_len, 1)

    # L2 norm over axis=(1,2)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]) + 1e-12)
    gp = tf.reduce_mean((grad_norm - 1.0)**2)
    return gp

# -------------------- 分類模型 (保持原本 create_model) --------------------

# 5. 建立整合模型：利用多路 CNN、AE encoder、LSTM 與生成器產生的特徵進行分類
def create_model(num_classes, HHT_train, DWT_train, features_train,
                 delta_features_train, encoder_features):

    # ✅ 定義所有 Input 層
    HHT_input = Input(shape=(HHT_train.shape[1], HHT_train.shape[2]))
    DWT_input = Input(shape=(DWT_train.shape[1], DWT_train.shape[2]))
    features_input = Input(shape=(features_train.shape[1], features_train.shape[2]))
    delta_features_input = Input(shape=(delta_features_train.shape[1], delta_features_train.shape[2]))
    # ✅ 這裡使用 Input，確保計算圖連接

    # ✅ HHT 特徵處理
    HHT = Conv1D(128, 10, padding='same', activation='relu')(HHT_input)
    HHT = MaxPooling1D(2)(HHT)
    HHT = Conv1D(64, 8, padding='same', activation='relu')(HHT)
    HHT = Conv1D(32, 6, padding='same', activation='relu')(HHT)
    HHT = GlobalAveragePooling1D()(HHT)
    HHT = BatchNormalization()(HHT)
    HHT = Dropout(0.5)(HHT)

    # ✅ DWT 特徵處理
    DWT = Conv1D(128, 10, padding='same', activation='relu')(DWT_input)
    DWT = MaxPooling1D(2)(DWT)
    DWT = Conv1D(64, 8, padding='same', activation='relu')(DWT)
    DWT = Conv1D(32, 6, padding='same', activation='relu')(DWT)
    DWT = GlobalAveragePooling1D()(DWT)
    DWT = BatchNormalization()(DWT)
    DWT = Dropout(0.5)(DWT)

    # ✅ 原始特徵處理
    features = Conv1D(128, 10, padding='same', activation='relu')(features_input)
    features = MaxPooling1D(2)(features)
    features = Conv1D(64, 8, padding='same', activation='relu')(features)
    features = Conv1D(32, 6, padding='same', activation='relu')(features)
    features = GlobalAveragePooling1D()(features)
    features = BatchNormalization()(features)
    features = Dropout(0.5)(features)

    # ✅ 取得 VAE 編碼特徵
    latent_feature = encoder_features(features_input)
    latent_feature = Dense(32, activation='relu')(latent_feature)

    # ✅ LSTM 處理 Delta Features
    delta_LSTM = Bidirectional(LSTM(128, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, dropout=0.4))(delta_features_input)
    delta_LSTM = BatchNormalization()(delta_LSTM)
    delta_LSTM = Bidirectional(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, dropout=0.4))(delta_LSTM)
    delta_LSTM = BatchNormalization()(delta_LSTM)
    delta_LSTM = Bidirectional(LSTM(32, activation='tanh', recurrent_activation='sigmoid', return_sequences=False, dropout=0.4))(delta_LSTM)
    delta_LSTM = BatchNormalization()(delta_LSTM)

    # ✅ 合併所有特徵
    combined = Concatenate()([HHT, DWT, features, latent_feature, delta_LSTM])
    combined = Reshape((1, -1))(combined)

    # ✅ 加入 Multi-Head Attention
    attention_output = MultiHeadAttention(num_heads=2, key_dim=32)(combined, combined)
    attention_output = LayerNormalization()(attention_output)
    combined = Add()([combined, attention_output])
    combined = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(combined)
    combined = Dropout(0.5)(combined)
    combined = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(combined)
    combined = Dropout(0.5)(combined)

    # ✅ 最終分類層
    combined = Flatten()(combined)
    output = Dense(num_classes, activation='sigmoid')(combined)

    # ✅ 修正模型 Inputs，確保計算圖不會斷開
    model = Model(inputs=[HHT_input, DWT_input, features_input,
                          delta_features_input], outputs=output)
    
    return model

def apply_attention_with_residual(original, generated):

    attention_output = MultiHeadAttention(num_heads=2, key_dim=32)(generated, original)
    attention_output = LayerNormalization()(attention_output)
    enhanced_features = Add()([original, attention_output])
    return enhanced_features
# -------------------- 訓練流程 --------------------
def train_model(HHT_train, HHT_test, DWT_train, DWT_test, features_train, features_test, 
                delt_features_train, delt_features_test,juice_train,
                juice_test ,steam_train, steam_test,juice_HHT_train,steam_HHT_train,juice_HHT_test,
                steam_HHT_test,juice_DWT_train,steam_DWT_train,juice_DWT_test,steam_DWT_test,
                y_train, y_test, name):
    
    num_classes = y_train.shape[1] 
    input_shape = (features_train.shape[1], features_train.shape[2])
    
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    # 先訓練 VAE，獲得 encoder
    ae_features, encoder_features, decoder_features = build_vae(input_shape, latent_dim, name_prefix="features")
    ae_features.fit(features_train, features_train, epochs=ae_epochs, batch_size=32, validation_data=(features_test, features_test), callbacks=[early_stopping])
    encoder_features.save(r"D:\graduate_info\Research\code\lab load\Machine learning\vae_encoder2.h5")
    decoder_features.save(r"D:\graduate_info\Research\code\lab load\Machine learning\vae_decoder2.h5")

    feat_shape = juice_train.shape[1:]      # (17, 1)
    hht_shape = juice_HHT_train.shape[1:]   # (39990, 1)
    dwt_shape = juice_DWT_train.shape[1:]   # (2142, 1)

    # ✅ 1. 建立 Generator 與 Discriminator
    generator = build_generator(feat_shape, hht_shape, dwt_shape)
    critic = build_critic(feature_shape=(features_train.shape[1],))

    # Optimizer / 超參數
    g_optimizer = Adam(learning_rate=5e-5, beta_1=0.5, beta_2=0.9)
    c_optimizer = Adam(learning_rate=5e-5, beta_1=0.5, beta_2=0.9)

    print("🚀 `juice_train` shape:", juice_train.shape)
    print("🚀 `steam_train` shape:", steam_train.shape)
    print("🚀 `juice_HHT_train` shape:", juice_HHT_train.shape)
    print("🚀 `steam_HHT_train` shape:", steam_HHT_train.shape)
    print("🚀 `juice_DWT_train` shape:", juice_DWT_train.shape)
    print("🚀 `steam_DWT_train` shape:", steam_DWT_train.shape)

    # 確保兩組數據長度一致
    min_samples = min(len(juice_train), len(steam_train))  
    juice_train = juice_train[:min_samples]  
    steam_train = steam_train[:min_samples]  
    juice_HHT_train = juice_HHT_train[:min_samples]  
    steam_HHT_train = steam_HHT_train[:min_samples]  
    juice_DWT_train = juice_DWT_train[:min_samples]  
    steam_DWT_train = steam_DWT_train[:min_samples]  

    def train_critic_step(real_data, juice_data, steam_data, juice_HHT_data,
                          steam_HHT_data, juice_DWT_data, steam_DWT_data):
        juice_data = tf.cast(juice_data, tf.float32)
        steam_data = tf.cast(steam_data, tf.float32)
        juice_HHT_data = tf.cast(juice_HHT_data, tf.float32)
        steam_HHT_data = tf.cast(steam_HHT_data, tf.float32)
        juice_DWT_data = tf.cast(juice_DWT_data, tf.float32)
        steam_DWT_data = tf.cast(steam_DWT_data, tf.float32)
        batch_size = tf.shape(juice_data)[0]
        alpha_val = tf.random.uniform([batch_size, 1], minval=0.3, maxval=0.7)
        with tf.GradientTape() as tape:
            fake_data_full = generator([juice_data, steam_data, juice_HHT_data,
                                        steam_HHT_data, juice_DWT_data, steam_DWT_data,
                                        alpha_val], training=True)

            fake_data = fake_data_full[0]  # 取主輸出 x_out

            # Critic 評分
            real_score = critic(real_data, training=True)
            fake_score = critic(fake_data, training=True)
            # Wasserstein loss
            c_loss = wasserstein_loss_critic(real_score, fake_score)

            # Gradient Penalty
            gp = gradient_penalty(critic, real_data, fake_data)
            c_loss_total = c_loss + gp_weight * gp

        grads = tape.gradient(c_loss_total, critic.trainable_variables)
        c_optimizer.apply_gradients(zip(grads, critic.trainable_variables))

        return c_loss, gp
    
    CYCLE_LOSS_WEIGHT = 3.0        # 原本 10 可以先降到 5 或更小
    FREQ_RECON_WEIGHT = 0.5  

    @tf.function
    def train_generator_step(juice_data, steam_data, juice_HHT_data,
                         steam_HHT_data, juice_DWT_data, steam_DWT_data):

        juice_data = tf.cast(juice_data, tf.float32)
        steam_data = tf.cast(steam_data, tf.float32)
        juice_HHT_data = tf.cast(juice_HHT_data, tf.float32)
        steam_HHT_data = tf.cast(steam_HHT_data, tf.float32)
        juice_DWT_data = tf.cast(juice_DWT_data, tf.float32)
        steam_DWT_data = tf.cast(steam_DWT_data, tf.float32)

        batch_size = tf.shape(juice_data)[0]
        alpha_val = tf.random.uniform([batch_size, 1], minval=0.0, maxval=1.0)

        with tf.GradientTape() as tape:
            # 取出 generator 輸出
            generated_out, mid_out, recon_j, recon_s, recon_hht, recon_dwt, pred_alpha = generator(
                [juice_data, steam_data,
                juice_HHT_data, steam_HHT_data,
                juice_DWT_data, steam_DWT_data,
                alpha_val],
                training=True
            )

            # Generator WGAN loss
            # Generator WGAN loss
            fake_score = critic(generated_out, training=True)
            g_loss = wasserstein_loss_gen(fake_score)

            # alpha 預測損失
            alpha_loss = tf.reduce_mean(tf.square(pred_alpha - alpha_val))  # 或用 binary_crossentropy 也可以

            # Cycle reconstruction losses
            cycle_loss_j = tf.reduce_mean(tf.square(recon_j - juice_data))
            cycle_loss_s = tf.reduce_mean(tf.square(recon_s - steam_data))

            # 中間 supervision loss（alpha * juice + (1-alpha) * steam）
            # 中間 supervision loss（alpha * juice + (1-alpha) * steam）
            alpha_tile = tf.tile(alpha_val[:, tf.newaxis, :], [1, juice_data.shape[1], 1])  # shape: (batch, seq_len, 1)
            target_mid = alpha_tile * juice_data + (1.0 - alpha_tile) * steam_data

            # 計算基礎 mid interpolation loss
            mid_interp_loss_each = tf.reduce_mean(tf.square(mid_out - target_mid), axis=[1, 2])  # shape: (batch,)

            # 建立權重：只對中間段（alpha ∈ [0.3, 0.7]）給予額外加權
            alpha_flat = tf.squeeze(alpha_val, axis=1)  # shape: (batch,)
            mid_weight = tf.where(
                tf.logical_and(alpha_flat >= 0.3, alpha_flat <= 0.7),
                tf.ones_like(alpha_flat) * 2.0,  # 強化學習
                tf.ones_like(alpha_flat) * 0.5   # 其餘部分給較低權重
            )

            # 計算加權後的 mid loss
            mid_interp_loss = tf.reduce_mean(mid_interp_loss_each * mid_weight)

            # 頻域重建 loss
            recon_loss_hht = tf.reduce_mean(tf.square(recon_hht - juice_HHT_data))
            recon_loss_dwt = tf.reduce_mean(tf.square(recon_dwt - juice_DWT_data))

            # === ✅ 全部合併 loss，請注意順序 ===
            total_g_loss = (
                g_loss +
                0.5 * alpha_loss +
                0.3 * mid_interp_loss +  # 強化後的中間監督
                CYCLE_LOSS_WEIGHT * (cycle_loss_j + cycle_loss_s) +
                FREQ_RECON_WEIGHT  * (recon_loss_hht + recon_loss_dwt)
            )

        # Apply gradients
        grads = tape.gradient(total_g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        return {
            "g_loss": g_loss,
            "cycle_j": cycle_loss_j,
            "cycle_s": cycle_loss_s,
            "recon_hht": recon_loss_hht,
            "recon_dwt": recon_loss_dwt,
            "alpha_loss": alpha_loss,
            "mid_interp_loss": mid_interp_loss
        }
    # ------------------------------------------------
    # (D) 開始 WGAN-GP 訓練
    # ------------------------------------------------
    print("🚀 開始 WGAN-GP 訓練 ...")
    num_train = min_samples  # 先簡單用
    indices = np.arange(num_train)

    for epoch in range(gan_epoch):
        # 每個 epoch, 打亂索引
        np.random.shuffle(indices)
        c_losses = []
        gp_vals = []
        g_losses = []
        cycle_losses_j = []
        cycle_losses_s = []
        hht_losses = []
        dwt_losses = []
        alpha_losses = []
        mid_losses = []

        # === 這裡用一個簡單的 loop，batch-wise 訓練 ===
        for i in range(0, num_train, batch_size):
            batch_idx = indices[i:i+batch_size]
            if len(batch_idx) < batch_size:
                break
            real_data_batch = features_train[batch_idx]   # shape: (batch_size, seq_len, 1)
            juice_data_batch = juice_train[batch_idx]
            steam_data_batch = steam_train[batch_idx]
            juice_HHT_data_batch = juice_HHT_train[batch_idx]
            steam_HHT_data_batch = steam_HHT_train[batch_idx]
            juice_DWT_data_batch = juice_DWT_train[batch_idx]
            steam_DWT_data_batch = steam_DWT_train[batch_idx]
            # ---- 先訓練 Critic 幾次 ----
            for _ in range(n_critic):
                c_loss, gp_val = train_critic_step(real_data_batch, juice_data_batch, steam_data_batch,
                                                   juice_HHT_data_batch, steam_HHT_data_batch,
                                                   juice_DWT_data_batch, steam_DWT_data_batch)
                c_losses.append(c_loss)
                gp_vals.append(gp_val)

            # ---- 再訓練一次 Generator ----
# ---- 再訓練一次 Generator ----
            result = train_generator_step(juice_data_batch, steam_data_batch,
                                        juice_HHT_data_batch, steam_HHT_data_batch,
                                        juice_DWT_data_batch, steam_DWT_data_batch)

            g_losses.append(result['g_loss'])
            cycle_losses_j.append(result['cycle_j'])
            cycle_losses_s.append(result['cycle_s'])
            hht_losses.append(result['recon_hht'])
            dwt_losses.append(result['recon_dwt'])
            alpha_losses.append(result['alpha_loss'])        # ✅ 新增
            mid_losses.append(result['mid_interp_loss']) 

        print(f"Epoch [{epoch+1}/{gan_epoch}] - "
            f"D_loss: {np.mean(c_losses):.4f}, GP: {np.mean(gp_vals):.4f}, "
            f"G_loss: {np.mean(g_losses):.4f}, "
            f"AlphaLoss: {np.mean(alpha_losses):.4f}, "
            f"MidInterp: {np.mean(mid_losses):.4f}, "
            f"Cycle_J: {np.mean(cycle_losses_j):.4f}, "
            f"Cycle_S: {np.mean(cycle_losses_s):.4f}, "
            f"HHT_Loss: {np.mean(hht_losses):.4f}, "
            f"DWT_Loss: {np.mean(dwt_losses):.4f}")

    # 訓練完畢後存檔
    generator.save(r'D:\graduate_info\Research\code\lab load\Machine learning\generator7.h5')
    critic.save(r"D:\graduate_info\Research\code\lab load\Machine learning\critic7.h5")

    # ------------------------------------------------
    # (E) 驗證 Generator 效果
    # ------------------------------------------------
    print("\n🔍 驗證 Generator 產生的混合數據...\n")
    print("\n🔍 驗證 Generator 產生的混合數據...\n")

    alphas_train = np.random.uniform(0.0, 1.0, size=(len(juice_train), 1))
    alphas_test = np.random.uniform(0.0, 1.0, size=(len(juice_test), 1))
    generated_train = generator.predict([
        juice_train, steam_train, juice_HHT_train, steam_HHT_train,
        juice_DWT_train, steam_DWT_train, alphas_train
    ])[0]

    generated_test= generator.predict([
        juice_test, steam_test, juice_HHT_test, steam_HHT_test,
        juice_DWT_test, steam_DWT_test, alphas_test
    ])[0]

    print(f"🔹 Juice Mean: {juice_train.mean():.4f}, Std: {juice_train.std():.4f}")
    print(f"🔹 Steam Mean: {steam_train.mean():.4f}, Std: {steam_train.std():.4f}")
    
    print(f"🔹 Juice HHT Mean: {juice_HHT_train.mean():.4f}, Std: {juice_HHT_train.std():.4f}")
    print(f"🔹 Steam HHT Mean: {steam_HHT_train.mean():.4f}, Std: {steam_HHT_train.std():.4f}")
    
    print(f"🔹 Juice DWT Mean: {juice_DWT_train.mean():.4f}, Std: {juice_DWT_train.std():.4f}")
    print(f"🔹 Steam DWT Mean: {steam_DWT_train.mean():.4f}, Std: {steam_DWT_train.std():.4f}")
    print(f"🔹 Generated Mean: {generated_train.mean():.4f}, Std: {generated_train.std():.4f}")

    from sklearn.manifold import TSNE
    import seaborn as sns


    # t-SNE 視覺化
    X_tsne_all = np.concatenate([
        juice_train.reshape(juice_train.shape[0], -1),
        steam_train.reshape(steam_train.shape[0], -1),
        generated_train.reshape(generated_train.shape[0], -1)
    ], axis=0)

    labels_all = (["Juice"] * len(juice_train) +
                ["Steam"] * len(steam_train) +
                ["Generated"] * len(generated_train))

    X_embedded = TSNE(n_components=2, perplexity=50, learning_rate=100).fit_transform(X_tsne_all)
    tsne_df = pd.DataFrame(X_embedded, columns=["x", "y"])
    tsne_df["label"] = labels_all

    plt.figure(figsize=(8,6))
    sns.scatterplot(data=tsne_df, x="x", y="y", hue="label", alpha=0.6)
    plt.title("t-SNE Visualization (Juice, Steam, Generated)")
    plt.grid(True)
    plt.show()

    # K-S 檢定
    from scipy.stats import ks_2samp
    ks_stat_juice, p_juice = ks_2samp(juice_train.flatten(), generated_train.flatten())
    ks_stat_steam, p_steam = ks_2samp(steam_train.flatten(), generated_train.flatten())
    print(f"🧐 K-S Test (Generated vs Juice): Stat={ks_stat_juice:.4f}, p-value={p_juice:.4f}")
    print(f"🧐 K-S Test (Generated vs Steam): Stat={ks_stat_steam:.4f}, p-value={p_steam:.4f}")
    if p_juice < 0.05 and p_steam < 0.05:
        print("✅ Generator 產生的數據與 Juice / Steam 有顯著差異，可能學到了混合特徵！")
    else:
        print("⚠️ Generator 與 Juice / Steam 差異不大，可能學習失敗！")


    juice_vec = juice_train.reshape(juice_train.shape[0], -1)
    steam_vec = steam_train.reshape(steam_train.shape[0], -1)

    alphas = np.linspace(0, 1, 11)  # 從 0 到 1 間隔生成
    sim_juice_list = []
    sim_steam_list = []

    for a in alphas:
        alpha_array = np.full((len(juice_train), 1), a)
        gen = generator.predict([juice_train, steam_train,
                                juice_HHT_train, steam_HHT_train,
                                juice_DWT_train, steam_DWT_train, alpha_array])[0]
        gen_vec = gen.reshape(gen.shape[0], -1)
        sim_juice_list.append(cosine_similarity(gen_vec, juice_vec).mean())
        sim_steam_list.append(cosine_similarity(gen_vec, steam_vec).mean())

    plt.plot(alphas, sim_juice_list, label='sim with Juice')
    plt.plot(alphas, sim_steam_list, label='sim with Steam')
    plt.xlabel('Alpha (mixing weight)')
    plt.ylabel('Cosine Similarity')
    plt.title('Generated Feature Similarity vs. Alpha')
    plt.legend()
    plt.grid(True)
    plt.show()

    # flatten 為向量
    juice_vec = juice_train.reshape(juice_train.shape[0], -1)
    steam_vec = steam_train.reshape(steam_train.shape[0], -1)
    gen_vec   = generated_train.reshape(generated_train.shape[0], -1)

    # 計算 gen 和 juice/steam 的相似度
    sim_juice = cosine_similarity(gen_vec, juice_vec).mean()
    sim_steam = cosine_similarity(gen_vec, steam_vec).mean()

    print(f"平均與 Juice 相似度: {sim_juice:.4f}")
    print(f"平均與 Steam 相似度: {sim_steam:.4f}")


    attention_score_train = Dense(1, activation='sigmoid')(generated_train)
    attention_feature_train = Multiply()([features_train, attention_score_train])
    fused_features_train = Add()([features_train, attention_feature_train])

    attention_score_test = Dense(1, activation='sigmoid')(generated_test)
    attention_feature_test = Multiply()([features_test, attention_score_test])
    fused_features_test = Add()([features_test, attention_feature_test])

    gen_vec_train = tf.keras.layers.Flatten()(generated_train)
    gen_vec_test  = tf.keras.layers.Flatten()(generated_test)

    # 🔸 HHT train
    gen_to_hht_train = Dense(HHT_train.shape[1], activation='sigmoid')(gen_vec_train)
    gen_to_hht_train = tf.expand_dims(gen_to_hht_train, axis=-1)
    attention_feature_hht_train = Multiply()([HHT_train, gen_to_hht_train])
    HHT_train = Add()([HHT_train, attention_feature_hht_train])

    # 🔹 HHT test ✅
    gen_to_hht_test = Dense(HHT_test.shape[1], activation='sigmoid')(gen_vec_test)
    gen_to_hht_test = tf.expand_dims(gen_to_hht_test, axis=-1)
    attention_feature_hht_test = Multiply()([HHT_test, gen_to_hht_test])
    HHT_test = Add()([HHT_test, attention_feature_hht_test])

     # 🔹 DWT train
    gen_to_dwt_train = Dense(DWT_train.shape[1], activation='sigmoid')(gen_vec_train)
    gen_to_dwt_train = tf.expand_dims(gen_to_dwt_train, axis=-1)
    attention_feature_dwt_train = Multiply()([DWT_train, gen_to_dwt_train])
    DWT_train = Add()([DWT_train, attention_feature_dwt_train])

    # 🔹 DWT test ✅
    gen_to_dwt_test = Dense(DWT_test.shape[1], activation='sigmoid')(gen_vec_test)
    gen_to_dwt_test = tf.expand_dims(gen_to_dwt_test, axis=-1)
    attention_feature_dwt_test = Multiply()([DWT_test, gen_to_dwt_test])
    DWT_test = Add()([DWT_test, attention_feature_dwt_test])

    # 開始建構分類模型
    class_weights = compute_class_weight_multilabel(y_train)
    optimizer = Adam(learning_rate=1e-4)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

    model_large = create_model(num_classes, HHT_train, DWT_train, fused_features_train, delt_features_train, encoder_features)
    model_large.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print(model_large.summary())

    # data_generator
    bs_for_class = min(16, len(juice_train))
    train_gen = data_generator(HHT_train, DWT_train, fused_features_train, delt_features_train, y_train, class_weights, bs_for_class)
    val_gen = data_generator(HHT_test, DWT_test, fused_features_test, delt_features_test, y_test, class_weights, bs_for_class)
    steps_per_epoch = len(y_train) // bs_for_class
    validation_steps = len(y_test) // bs_for_class

    model_large.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=model_large_epochs,
        callbacks=[early_stopping, reduce_lr]
    )

    # 存檔 + 評估
    model_large.save(r'D:\graduate_info\Research\code\lab load\Machine learning\1DCNN_model_DWT&HHT2_training_My_data_多分類旋轉機7.h5')
    y_pred_prob = model_large.predict([HHT_test, DWT_test, features_test, delt_features_test])
    y_pred = (y_pred_prob > 0.5).astype(int)

    # ... 後面你原本的繪圖、打印 confusion matrix, classification_report 等 ...

    plt.figure(figsize=(8, 6))
    plt.hist(y_pred_prob.flatten(), bins=50, color='b', alpha=0.7)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Distribution of y_pred_prob Values")
    plt.grid(True)
    plt.savefig(r'D:\graduate_info\Research\code\lab load\Machine learning\y_pred_prob7.png')
    plt.close()

    # ...
    # (以下你原本的最佳 threshold 處理、混淆矩陣、report 等照舊)
    num_classes_ = y_pred_prob.shape[1]
    best_thresholds = []
    for i in range(num_classes_):
        best_threshold = find_best_threshold(y_test[:, i], y_pred_prob[:, i])
        best_thresholds.append(best_threshold)
    print(f"✅ 最佳信心閾值: {best_thresholds}")
    np.save(r"D:\graduate_info\Research\code\lab load\Machine learning\confidence_thresholds7.npy", best_thresholds)

    y_pred = np.zeros_like(y_pred_prob)
    for i in range(y_pred_prob.shape[1]):
        threshold = best_thresholds[i]
        y_pred[:, i] = (y_pred_prob[:, i] > threshold).astype(int)

    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    accuracy_text = f"Accuracy: {accuracy:.4f}"
    print(accuracy_text)
    
    load_names = [f"Class {i}" for i in range(y_test.shape[1])]  # 根據實際類別
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=load_names, yticklabels=load_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.text(0.5, -0.5, accuracy_text, fontsize=12, color="red", 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title("Confusion Matrix for 4 Appliances")
    plt.savefig(r'D:\graduate_info\Research\code\lab load\Classification7.png')
    plt.close()

    print("=== Classification Report (by label) ===")
    print(classification_report(y_test, y_pred, target_names=load_names, zero_division=0))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='micro')
    print(f"Micro-average => precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")


# -------------------- 資料結構設定與訓練呼叫 --------------------
class_HHT_train_lists = { 1: Class1_HHT_train }
class_DWT_train_lists = { 1: Class1_DWT_train }
class_HHT_test_lists = { 1: Class1_HHT_test }
class_DWT_test_lists = { 1: Class1_DWT_test }
class_features_train_lists = { 1: Class1_features_train }
class_features_test_lists = { 1: Class1_features_test }
class_delt_features_train_lists = { 1: Class1_delt_features_train }
class_delt_features_test_lists = { 1: Class1_delt_features_test }
class_y_train_lists = { 1: Class1_y_train }
class_y_test_lists = { 1: Class1_y_test }
juice_train_lists = { 1: juice_train }
steam_train_lists = { 1: steam_train }
juice_test_lists = { 1: juice_test }
steam_test_lists = { 1: steam_test }
juice_HHT_train_lists = { 1: juice_HHT_train }
steam_HHT_train_lists = { 1: steam_HHT_train }
juice_HHT_test_lists = { 1: juice_HHT_test }
steam_HHT_test_lists = { 1: steam_HHT_test }
juice_DWT_train_lists = { 1: juice_DWT_train }
steam_DWT_train_lists = { 1: steam_DWT_train }
juice_DWT_test_lists = { 1: juice_DWT_test }
steam_DWT_test_lists = { 1: steam_DWT_test }
class_name_lists = { 1: [] }  # 如有需要請填入設備名稱

classnumber = 1

for i in range(1, classnumber + 1):
    print("juice_train length:", len(juice_train))
    print("steam_train length:", len(steam_train))
    train_model(
        HHT_train = class_HHT_train_lists[i], 
        HHT_test = class_HHT_test_lists[i], 
        DWT_train = class_DWT_train_lists[i], 
        DWT_test = class_DWT_test_lists[i],
        features_train = class_features_train_lists[i], 
        features_test = class_features_test_lists[i], 
        delt_features_train = class_delt_features_train_lists[i],
        delt_features_test = class_delt_features_test_lists[i],
        y_train = class_y_train_lists[i], 
        y_test = class_y_test_lists[i], 
        name = class_name_lists[i],
        juice_train = juice_train_lists[i],
        juice_test = juice_test_lists[i],
        steam_train = steam_train_lists[i], 
        steam_test = steam_test_lists[i],
        juice_HHT_train = juice_HHT_train_lists[i],
        juice_HHT_test = juice_HHT_test_lists[i],
        steam_HHT_train = steam_HHT_train_lists[i], 
        steam_HHT_test = steam_HHT_test_lists[i],
        juice_DWT_train = juice_DWT_train_lists[i],
        juice_DWT_test = juice_DWT_test_lists[i],
        steam_DWT_train = steam_DWT_train_lists[i], 
        steam_DWT_test = steam_DWT_test_lists[i],
    )

# 結束計時
end_time = time.time()
execution_time = end_time - start_time
print(f"程式執行時間: {execution_time} 秒")
