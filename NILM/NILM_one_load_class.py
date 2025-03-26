from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
import keras
import time
import shutil

# 開始計時
start_time = time.time()
print('分類程式開始執行=============================================')
print('keras version: ', keras.__version__)

# 資料集
csv_folder_path = r"D:\graduate_info\Research\code\lab load\2025 lab mix"
meta_json_path = r"D:\graduate_info\Research\code\lab load\2025 lab mix\2025_lab_submetered.json"
input_folder = r"D:\graduate_info\Research\code\lab load\Devices_lab_oneload_Feature"

new_appliance_input_folder = r"D:\graduate_info\Research\code\lab load\new_appliance_output_folder_one_load_combine"



# 設置參數 =============================================================================================

Number_of_features = 10
window_size = 2000  # 視窗大小
step_size = 1000  # 視窗滑動步長
max_imfs = 10
level = 5

# ======================================================================================================


# 定義清空資料夾的函數
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 刪除文件或符號連結
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 遞歸刪除資料夾
        print(f"資料夾 {folder_path} 已清空")
    else:
        os.makedirs(folder_path)  # 如果資料夾不存在，創建它
        print(f"資料夾 {folder_path} 已創建")

# 讀取所有設備的 CSV 文件
Device = []
valid_indices = []

for file_name in os.listdir(input_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(input_folder, file_name)
        device_data = pd.read_csv(file_path)
        Device.append(device_data)

print(f"已成功讀取 {len(Device)} 個設備的 CSV 文件。")

target_appliances = [
    "Juice",
    "Steam",
]  # 根據需要修改


if len(Device) != len(target_appliances):
    raise ValueError("❌ `Device` 長度與 `target_appliances` 不匹配，請檢查資料！")

# 建立每個電器的 WindowIndex 集合
valid_indices_per_appliance = {}

for i, appliance in enumerate(target_appliances):
    # 獲取該設備的 `WindowIndex`
    valid_indices_per_appliance[appliance] = set(Device[i]["WindowIndex"].unique())

    print(f"📌 {appliance} 的有效 WindowIndex 數量: {len(valid_indices_per_appliance[appliance])}")

# 測試輸出
print("✅ 所有電器的有效 WindowIndex 已存入 `valid_indices_per_appliance`")

all_data = pd.concat(Device, ignore_index=True)
# print('all_data:',all_data)
new_appliance_data = {}
# 定義您感興趣的特定負載名稱

for file_name in os.listdir(new_appliance_input_folder):
    appliance_name = file_name.replace(".csv", "")
    if appliance_name in target_appliances:
        file_path = os.path.join(new_appliance_input_folder, file_name)
        device_data = pd.read_csv(file_path)
        new_appliance_data[appliance_name] = device_data
        # print("new_appliance_data",new_appliance_data)

print(f"已成功讀取 {len(new_appliance_data)} 設備名稱並存入new_appliance_data。")

# 生成谐波列名
harmonics_columns = [f'Harmonics{i+1}' for i in range(7)]
phase_columns = [f'phase{i + 1}' for i in range(7)] 

# 假設你的數據存儲在 all_data 中
all_columns = [
    'RMS', 'Peak', 'Peak-to-Peak', 'Waveform Factor', 'Crest Factor',
    'Power', 'Power Std'
] + harmonics_columns + phase_columns + [
    'THD', 'Current Range','ZCR','vi area','P','Q','Skewness','Kurtosis',
    'Delta Current Mean','Delta Current Std'
]

valid_indices = all_data["WindowIndex"]
print("讀到的有效視窗編號:", valid_indices)

appliance_names = list(new_appliance_data.keys())

# # 按照 appliance_list 順序，依序從字典取出 DataFrame
devices = [new_appliance_data[name] for name in appliance_names]
print("devices",devices)

# 定義特徵名稱
feature_names = [
    'RMS', 'Peak', 'Peak-to-Peak', 'Waveform Factor', 'Crest Factor',
    'Power', 'Power Std', 'vi area', 'Current Range', 'ZCR',
    'Skewness', 'Kurtosis','Delta Current Mean','Delta Current Std'
] 

# 定義變化量特徵名稱 (加上 Delta_ 前綴)
delta_feature_names = [f"Delta_{name}" for name in feature_names]

# 提取特徵和標籤
X = all_data[feature_names]

# 監督二元分類 (假裝)
import joblib
from sklearn.mixture import BayesianGaussianMixture
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA

# 使用  GMM 進行聚類
print("GMM開始進行聚類")
num = 1  # 這裡只是示範 => GMM n_components=1
vbgmm = BayesianGaussianMixture(
    n_components=num, 
    covariance_type='spherical', 
    tol=1e-5, 
    random_state=42, 
    n_init=10, 
    max_iter=1000
)

print("GMM開始預測")
pseudo_labels = vbgmm.fit_predict(X)
print("GMM預測結束")

gmm_model_path = r"D:\graduate_info\Research\code\lab load\Machine Learning\GMM偽標籤分類.pkl"
joblib.dump(vbgmm, gmm_model_path)
print(f"GMM 模型已保存至: {gmm_model_path}")

# 将假標籤添加到数据中
all_data['pseudo_label'] = pseudo_labels
print("偽標籤分布: ")
print(all_data['pseudo_label'].value_counts())

# PCA (示範)
pca = PCA(n_components = Number_of_features)  
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
print(f"PCA 保留了 {X_pca.shape[1]} 個主成分")
print("PCA 各主成分的貢獻度:", explained_variance)

# 找出最重要的特徵
num_features_to_select = min(len(feature_names), X_pca.shape[1]) 
feature_importance = np.abs(pca.components_[:num_features_to_select, :]).sum(axis=0)
pca_top_features = np.argsort(feature_importance)[-num_features_to_select:]

# 選取最重要的特徵
selected_pca_features = [feature_names[i] for i in pca_top_features]
print("✅ PCA 選出的特徵:", selected_pca_features)

# 选取互信息值最高的特征(示範 => 直接把谐波加回)
top_features = selected_pca_features + harmonics_columns
print("互信息篩選后特徵:", top_features)

output_file_path = r"D:\graduate_info\Research\code\lab load\Machine learning\selected_features.txt"
with open(output_file_path, 'w', encoding='utf-8') as f:
    for feature in top_features:
        f.write(f"{feature}\n")
print(f"特徵名稱已成功保存至: {output_file_path}")

# 再次選取 => 用 top_features
X_selected = all_data[top_features].values
y_pseudo = np.array(pseudo_labels)

print("delta_feature_names:", delta_feature_names)

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_pseudo, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# 初始化模型 (示範)
model = Sequential()
print("NN開始訓練")

model.add(Dense(128, input_dim = X_train.shape[1], activation='relu', 
                kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))

model.add(Dense(64, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))

model.add(Dense(32, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))

# 輸出層 (假設只有 1 => 這裡只是示範)
model.add(Dense(num, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

class_weights = compute_class_weight('balanced', classes=np.unique(pseudo_labels), y=pseudo_labels)
class_weights = dict(enumerate(class_weights))
print("类别权重：", class_weights)

history = model.fit(
    X_train, y_train,
    epochs=1,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    class_weight=class_weights
)
print(model.summary())

model.save(r'D:\graduate_info\Research\code\lab load\Machine Learning\旋轉機器5分類.h5')
print("NN 訓練完成並已保存模型。")

# 預測
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=-1)
print('y_pred_classes: ', y_pred_classes)
print('y_test_classes: ', y_test)
accuracy = accuracy_score(y_test, y_pred_classes)
print(f'模型準確率: {accuracy * 100:.2f}%')

# 分類結果
class_1 = []
class_2 = []
class_3 = []
class_4 = []
class_5 = []
class_6 = []
class_7 = []
class_8 = []
class_9 = []
class_10 = []

class_1_features = []
class_2_features = []
class_3_features = []
class_4_features = []
class_5_features = []
class_6_features = []
class_7_features = []
class_8_features = []
class_9_features = []
class_10_features = []

class_1_delt_features = []
final_label_pred = []

# ======== 重點：多標籤 build 函式 ========
def build_multi_label_from_integers(integers):
    num_classes=2
    multi_labels = []
    for val in integers:
        # 若 val 是 numpy array，先轉成 python list
        if isinstance(val, np.ndarray):
            val = val.tolist()

        # 建立一個 2 維度的標籤 (e.g. [0,0])
        label = np.zeros(num_classes, dtype=int)

        if isinstance(val, list):
            # val 可能是 [0], [1], [0,1], []
            for sub_val in val:
                sub_val = int(sub_val)  # 確保是整數
                label[sub_val] = 1
        else:
            # val 若是單一整數或字串 => 轉成 int
            val = int(val)
            label[val] = 1

        multi_labels.append(label)

    return np.array(multi_labels, dtype=int)

def save_each_row_as_npy(dataframe, delt_dataframe, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    # 獲取特徵和負載數據
    features = dataframe.drop(columns=['LoadEncoded','load'], errors='ignore').to_numpy()
    delt_features = delt_dataframe.drop(columns=['LoadEncoded','load'], errors='ignore').to_numpy()
    load = dataframe['LoadEncoded'].to_numpy()

    for i, (feature_row, delt_feature_row, load_value) in enumerate(zip(features, delt_features, load)):
        feature_row_reshaped = np.expand_dims(feature_row, axis=0)
        delt_feature_row_reshaped = np.expand_dims(delt_feature_row, axis=0)

        # 保存單行特徵
        feature_filename = os.path.join(output_dir, f"{prefix}_load_features_{i}.npy")
        np.save(feature_filename, feature_row_reshaped)
        delt_feature_filename = os.path.join(output_dir, f"{prefix}_load_delt_features_{i}.npy")
        np.save(delt_feature_filename, delt_feature_row_reshaped)

        # 保存對應負載
        load_filename = os.path.join(output_dir, f"{prefix}_load_name_{i}.npy")
        np.save(load_filename, np.array([load_value]))
    
    print(f"{prefix} 的每一行特徵、變化量和負載已分别保存至 {output_dir}")

def save_load_column_as_csv(dataframe, output_csv_path):
    if 'load' not in dataframe.columns:
        print(f"DataFrame 裡沒有 'load' 欄位，無法輸出")
        return
    load_series = dataframe['load'].drop_duplicates()
    load_df = pd.DataFrame(load_series, columns=['load'])
    load_df.to_csv(output_csv_path, index=False, header=False, encoding='utf-8-sig')
    print(f"已將唯一的電器名稱存成 CSV：{output_csv_path}")

print('=====================================================================')

from PyEMD import EMD
from scipy.signal import hilbert
import pywt

class_1_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 1'
class_2_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 2' 
class_3_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 3' 
class_4_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 4'
class_5_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 5'
class_6_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 6' 
class_7_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 7'
class_8_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 8'
class_9_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 9'
class_10_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 10'

def wavelet_transform(data, wavelet='coif5', level=5):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    flattened_coeffs = np.hstack(coeffs)
    return flattened_coeffs

def HHT_transform(data, max_imfs=20):
    emd = EMD()
    imfs = emd(data)
    # 補 0 到指定數量 / 截斷
    if len(imfs) < max_imfs:
        padding = max_imfs - len(imfs)
        imfs = np.pad(imfs, ((0, padding), (0, 0)), 'constant')
    imfs = imfs[:max_imfs]
    
    hht_data = []
    for imf in imfs:
        analytic_signal = hilbert(imf)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * 1000
        # amplitude + frequency
        hht_data.append(amplitude_envelope)
        hht_data.append(instantaneous_frequency)
    
    flattened_hht = np.hstack(hht_data)
    return flattened_hht

def sliding_window_feature_extraction(data_array, label_array,
                                      window_size=5000, step=2500,
                                      wavelet='coif5', level=5,
                                      max_imfs=20,
                                      valid_indices=None
):
    """
    label_array 內每筆是 list -> [0], [1], [0,1], []...
    最後會投票出 majority_label (list)
    """
    n_samples = len(data_array)
    HHT_features_list = []
    DWT_features_list = []
    window_labels_list = []
    
    window_index = 0  # 用於跟 valid_indices 對照

    for start in range(0, n_samples - window_size + 1, step):
        end = start + window_size

                # 若指定了 valid_indices，且目前的 window_index 不在其中 => 直接跳過
        if (valid_indices is not None) and (window_index not in valid_indices):
            window_index += 1
            continue

        window_signal = data_array[start:end]
        window_label_seq = label_array[start:end]  # 這裡是一堆 list

        # ---- HHT 特徵 ----
        hht_fv = HHT_transform(window_signal, max_imfs=max_imfs)
        # ---- DWT 特徵 ----
        dwt_fv = wavelet_transform(window_signal, wavelet=wavelet, level=level)
        
        # 投票: 需要將 list => tuple 才能當 dict key
        count_dict = {}
        for lbl_list in window_label_seq:
            lbl_tuple = tuple(lbl_list)  # e.g. [0] => (0,), [0,1] => (0,1)
            count_dict[lbl_tuple] = count_dict.get(lbl_tuple, 0) + 1
        
        majority_label_tuple = max(count_dict, key=count_dict.get)
        # 轉回 list
        majority_label_list = list(majority_label_tuple)

        HHT_features_list.append(hht_fv)
        DWT_features_list.append(dwt_fv)
        window_labels_list.append(majority_label_list)
        window_index += 1

    HHT_features = np.array(HHT_features_list)
    DWT_features = np.array(DWT_features_list)
    labels = np.array(window_labels_list, dtype=object)  # 保留 list

    return HHT_features, DWT_features, labels

def check_and_save(data, filepath, expected_shape=None):
    if expected_shape is None:
        expected_shape = data.shape
    elif data.shape != expected_shape:
        print(f"形狀不一致！預期形狀: {expected_shape}, 實際形狀: {data.shape}")
        raise ValueError("數據形狀不一致，停止保存。")
    np.save(filepath, data)

print('=====================================================================')

for i, (device, group) in enumerate(zip(appliance_names, Device)):
    # 提取與模型一致的特徵
    X_group = group[top_features].values
    
    # 預測每個樣本的類別
    group_pred = model.predict(X_group)  # 預測概率
    mean_probs = np.mean(group_pred, axis=0) 

    cluster_label = np.argmax(mean_probs)  
    
    # 保存結果
    final_label_pred.append(cluster_label)
    print(f"設備 {device} 的聚類結果: {cluster_label}")

# 檢查最終的分類結果
print('final_label_pred:', final_label_pred)

final_label_pred = np.array(final_label_pred)
# print("最終分類結果:", final_label_pred)

LABEL = 'LoadEncoded'
print(LABEL)
le = preprocessing.LabelEncoder()

for i, label in enumerate(final_label_pred):
    device_data = Device[i][top_features].copy()
    device_delt_data = Device[i][delta_feature_names].copy()
    if label == 0:
        class_1.append(devices[i])
        device_data['load'] = appliance_names[i]
        class_1_features.append(device_data)
        device_delt_data['load'] = appliance_names[i]
        class_1_delt_features.append(device_delt_data)
        print("class_1_delt_features:",class_1_delt_features)

    elif label == 1:
        class_2.append(devices[i])
        device_data['load'] = appliance_names[i]
        class_2_features.append(device_data)

    elif label == 2:
        class_3.append(devices[i])
        device_data['load'] = appliance_names[i]
        class_3_features.append(device_data)

    elif label == 3:
        class_4.append(devices[i])
        device_data['load'] = appliance_names[i]
        class_4_features.append(device_data)

    elif label == 4:
        class_5.append(devices[i])
        device_data['load'] = appliance_names[i]
        class_5_features.append(device_data)

    elif label == 5:
        class_6.append(devices[i])
        device_data['load'] = appliance_names[i]
        class_6_features.append(device_data)

    elif label == 6:
        class_7.append(devices[i])
        device_data['load'] = appliance_names[i]
        class_7_features.append(device_data)

    elif label == 7:
        class_8.append(devices[i])
        device_data['load'] = appliance_names[i]
        class_8_features.append(device_data)

    elif label == 8:
        class_9.append(devices[i])
        device_data['load'] = appliance_names[i]
        class_9_features.append(device_data)
        
    elif label == 9:
        class_10.append(devices[i])
        device_data['load'] = appliance_names[i]
        class_10_features.append(device_data)

# ============== 這裡才是主要多標籤應用 =============
load_mapping = {
    "Juice": [0],
    "None": [],
    "Steam": [1],
}

def process_single_device(
    device_df, 
    device_features_df, 
    delt_device_features_df,
    output_dir,
    device_name,
    start_index=0,     
    LABEL='LoadEncoded',
    wavelet='coif5',
    level=5,
    max_imfs=10,
    window_size=2000,
    step=1000,
    num=1,
    valid_indices=None  
):
    # print("device_df:", device_df["load"])
    now_device = device_df["load"].iloc[0]
    print("now_device", now_device)

    # 這裡將 'load' -> list of int
    device_df["LoadEncoded"] = device_df["load"].apply(lambda x: load_mapping[x])
    device_features_df["LoadEncoded"] = device_features_df['load'].apply(lambda x: load_mapping[x])
    delt_device_features_df["LoadEncoded"] = delt_device_features_df['load'].apply(lambda x: load_mapping[x])

    print('device_df["LoadEncoded"] =', device_df["LoadEncoded"].head())
    print('device_features_df["LoadEncoded"] =', device_features_df["LoadEncoded"].head())
    print('delt_device_features_df["LoadEncoded"] =', delt_device_features_df["LoadEncoded"].head())

    # (可選) 輸出每行特徵
    save_each_row_as_npy(device_features_df, delt_device_features_df, output_dir, prefix=f"class{num}_{now_device}")
    
    # 繪圖 (可選)
    plt.figure(figsize=(6, 4))
    device_df['load'].value_counts().plot(kind='bar', title=now_device)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{now_device}.png')
    plt.savefig(save_path, dpi=200)
    plt.close()

    # sliding window
    data_array = device_df['current'].values.astype(float)
    # label_array = device_df[LABEL].values -> 這是 list (可能 [] or [0], [1], [0,1])
    label_array = device_df[LABEL].values

    valid_indices = valid_indices_per_appliance[now_device]

    # 執行 sliding_window_feature_extraction
    HHT_features, DWT_features, raw_labels = sliding_window_feature_extraction(
        data_array=data_array,
        label_array=label_array,
        window_size=window_size,
        step=step,
        wavelet=wavelet,
        level=level,
        max_imfs=max_imfs,
        valid_indices=valid_indices 
    )

    # raw_labels => shape=(num_windows,) 每個是 list => e.g. [0], [1], [0,1], []
    # 轉成 (N,2)
    labels_2d = build_multi_label_from_integers(raw_labels)

    print("labels after conversion (multi-label format):")
    print(labels_2d[:10])  

    # 逐 batch 輸出
    batch_counter = start_index
    for i in range(len(labels_2d)):
        hht_i = np.expand_dims(HHT_features[i], axis=0)
        dwt_i = np.expand_dims(DWT_features[i], axis=0)
        label_i = np.array([labels_2d[i]])  # shape (1,2)

 # **✅ 修改命名**
        hht_filename = os.path.join(output_dir, f"{now_device}_HHT_features_{batch_counter}.npy")
        dwt_filename = os.path.join(output_dir, f"{now_device}_DWT_features_{batch_counter}.npy")
        label_filename = os.path.join(output_dir, f"{now_device}_labels_{batch_counter}.npy")

        np.save(hht_filename, hht_i)
        np.save(dwt_filename, dwt_i)
        np.save(label_filename, label_i)
        batch_counter += 1

    return batch_counter

def class_process(num, class_num, class_features, class_delt_features, class_output_dir, 
                  wavelet='coif5', level=5, max_imfs=20, 
                  window_size=5000, step=2500,
                  valid_indices=None):
    if not class_num:
        clear_folder(class_output_dir)
        print(f"沒有找到任何 class{num}。")
        return
    
    os.makedirs(class_output_dir, exist_ok=True)
    clear_folder(class_output_dir)

    batch_counter = 0
    
    for i, (device_df, device_features_df, delt_device_features_df) in enumerate(zip(class_num, class_features, class_delt_features)):
        device_name = device_df['load'].iloc[0]
        batch_counter = process_single_device(
            device_df=device_df.copy(),
            device_features_df=device_features_df.copy(),
            delt_device_features_df=delt_device_features_df.copy(),
            output_dir=class_output_dir,
            device_name=f"{device_name}_device{i}",
            start_index=batch_counter,
            LABEL='LoadEncoded',
            wavelet=wavelet,
            level=level,
            max_imfs=max_imfs,
            window_size=window_size,
            step=step,
            num=num,
            valid_indices=valid_indices
        )
    print(f"Class{num}：所有設備都已處理完畢，最終 batch_counter={batch_counter}")

class_lists = {
    1: class_1,
    2: class_2,
    3: class_3,
    4: class_4,
    5: class_5,
    6: class_6,
    7: class_7,
    8: class_8,
    9: class_9,
    10: class_10
}

class_features_lists = {
    1: class_1_features,
    2: class_2_features,
    3: class_3_features,
    4: class_4_features,
    5: class_5_features,
    6: class_6_features,
    7: class_7_features,
    8: class_8_features,
    9: class_9_features,
    10: class_10_features
}

class_delt_features_lists = {
    1: class_1_delt_features
}

output_dirs_lists = {
    1: class_1_output_dir,
    2: class_2_output_dir,
    3: class_3_output_dir,
    4: class_4_output_dir,
    5: class_5_output_dir,
    6: class_6_output_dir,
    7: class_7_output_dir,
    8: class_8_output_dir,
    9: class_9_output_dir,
    10: class_10_output_dir
}

# 這裡示範只跑 class 1
for classnumber in range(1, num + 1):
    class_process( 
        num=classnumber,
        class_num=class_lists[classnumber],
        class_features=class_features_lists[classnumber],
        class_delt_features=class_delt_features_lists[classnumber],
        class_output_dir=output_dirs_lists[classnumber],
        wavelet='coif5',
        level=level,
        max_imfs=max_imfs,
        window_size=window_size,
        step=step_size,
        valid_indices=valid_indices
    )

# 結束計時
end_time = time.time()
execution_time = end_time - start_time
print(f"程式執行時間: {execution_time} 秒")
