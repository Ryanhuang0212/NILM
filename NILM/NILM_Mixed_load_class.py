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

# 設置參數 =============================================================================================

Number_of_features = 10
window_size = 2000  # 視窗大小
step_size = 2000  # 視窗滑動步長
max_imfs = 10
level = 5

# ======================================================================================================

# 資料集
csv_folder_path = r"D:\graduate_info\Research\code\lab load\2025 lab mix"
meta_json_path = r"D:\graduate_info\Research\code\lab load\2025 lab mix\2025_lab_15s_aggregated_Steam_juice.json"
input_folder = r"D:\graduate_info\Research\code\lab load\Devices_Mixed_load_Feature"

new_appliance_input_folder = r"D:\graduate_info\Research\code\lab load\new_appliance_output_folder_combine"

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

all_data = pd.concat(Device, ignore_index=True)

new_appliance_data = {}
# 定義您感興趣的特定負載名稱
target_appliances = ['Steam_Juice']

for file_name in os.listdir(new_appliance_input_folder):
    appliance_name = file_name.replace(".csv", "")
    file_path = os.path.join(new_appliance_input_folder, file_name)
    device_data = pd.read_csv(file_path)
    Device.append(device_data)

    new_appliance_data[appliance_name] = device_data

print(f"已成功讀取 {len(new_appliance_data)} 設備名稱並存入new_appliance_data。")

# 建立每個電器的 WindowIndex 集合
valid_indices_per_appliance = {}

for i, appliance in enumerate(target_appliances):
    # 獲取該設備的 `WindowIndex`
    valid_indices_per_appliance[appliance] = set(Device[i]["WindowIndex"].unique())

    print(f"📌 {appliance} 的有效 WindowIndex 數量: {len(valid_indices_per_appliance[appliance])}")


print("✅ 所有電器的有效 WindowIndex 已存入 `valid_indices_per_appliance`")

# 生成谐波列名
harmonics_columns = [f'Harmonics{i+1}' for i in range(7)]
phase_columns = [f'phase{i + 1}' for i in range(7)] 

# 假設你的數據存儲在 all_data 中
all_columns = ['RMS', 'Peak', 'Peak-to-Peak', 'Waveform Factor', 'Crest Factor', 
               'Power', 'Power Std'] + harmonics_columns + phase_columns + ['THD', 'Current Range','ZCR','vi area','GAF diag','P','Q','Skewness','Kurtosis','Delta Current Mean','Delta Current Std']

appliance_names = list(new_appliance_data.keys())
print(appliance_names)

# print('=================================================')
# 按照 appliance_list 順序，依序從字典取出 DataFrame
devices = [new_appliance_data[name] for name in appliance_names]

# print('devices:',devices)
# print('=================================================')

# 提取所有特徵
feature_names = ['RMS', 'Peak', 'Peak-to-Peak', 'Waveform Factor', 'Crest Factor', 
                 'Power', 'Power Std', 'vi area', 'Current Range', 'ZCR', 
                'Skewness' , 'Kurtosis','Delta Current Mean','Delta Current Std'] 

# 定義變化量特徵名稱 (加上 Delta_ 前綴)
delta_feature_names = [f"Delta_{name}" for name in feature_names]

# selected_features = ['P',  'Power', 'Q', 'THD']

features = feature_names

# 提取特徵和標籤
X = all_data[features]

# 監督二元分類
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

num = 1
# 使用  GMM 進行聚類
gmm_model_path =  r"D:\graduate_info\Research\code\lab load\Machine Learning\GMM偽標籤分類.pkl"

loaded_gmm = joblib.load(gmm_model_path)
# print("GMM開始預測")
pseudo_labels = loaded_gmm.fit_predict(X)
# pseudo_labels = gmm_fake.fit_predict(X)
print("GMM預測結束")

# 将假標籤添加到数据中
all_data['pseudo_label'] = pseudo_labels
print("偽標籤分布: ")
print(all_data['pseudo_label'].value_counts())

from sklearn.decomposition import PCA

# 設定要保留的主成分數量
pca = PCA(n_components = Number_of_features)  # 例如保留 10 維特徵
X_pca = pca.fit_transform(X)

explained_variance = pca.explained_variance_ratio_
print(f"PCA 保留了 {X_pca.shape[1]} 個主成分")
print("PCA 各主成分的貢獻度:", explained_variance)

# 找出最重要的特徵
num_features_to_select = min(len(features), X_pca.shape[1])  # 確保不超過原始特徵數
feature_importance = np.abs(pca.components_[:num_features_to_select, :]).sum(axis=0)  # 計算特徵影響力
pca_top_features = np.argsort(feature_importance)[-num_features_to_select:]  # 取影響力最大的特徵

# 選取最重要的特徵
selected_pca_features = [features[i] for i in pca_top_features]
print("✅ PCA 選出的特徵:", selected_pca_features)

top_features = selected_pca_features + harmonics_columns

print("top_features:", top_features)

# 使用讀取的特徵名稱進行後續操作
# 篩選數據中對應的特徵列
selected_data = all_data[top_features]

print("篩選後的數據:")
print(selected_data.head())

# Step 4: 使用筛选特征重新聚类
X_selected = all_data[top_features].values
y_pseudo = np.array(pseudo_labels)

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_pseudo, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

model_path =r'D:\graduate_info\Research\code\lab load\Machine Learning\旋轉機器5分類.h5'

model = load_model(model_path)

# 接下來將 y_pred_classes 用於設備分類
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

def save_each_row_as_npy(dataframe, delt_dataframe, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    # 獲取特徵和負載數據
    features = dataframe.drop(columns=['LoadEncoded','load']).to_numpy()
    delt_features = delt_dataframe.drop(columns=['load']).to_numpy()
    load = dataframe['LoadEncoded'].to_numpy()

    for i, (feature_row, delt_feature_row ,load_value) in enumerate(zip(features, delt_features,load)):
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
    
    # 只取 'load' 欄位 + 去重 (確保每個名稱只出現一次)
    load_series = dataframe['load'].drop_duplicates()
    # 將其轉成一個 DataFrame
    load_df = pd.DataFrame(load_series, columns=['load'])
    
    load_df.to_csv(output_csv_path, index=False, header=False, encoding='utf-8-sig')
    print(f"已將唯一的電器名稱存成 CSV：{output_csv_path}")

print('=====================================================================')

from PyEMD import EMD
from scipy.signal import hilbert
import pywt


class_1_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 1_mix'  
class_2_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 2_mix' 
class_3_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 3_mix' 
class_4_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 4_mix'
class_5_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 5_mix' 
class_6_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 6_mix'  
class_7_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 7_mix' 
class_8_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 8_mix' 
class_9_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 9_mix' 
class_10_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 10_mix' 


def wavelet_transform(data, wavelet='coif5', level = 5):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    flattened_coeffs = np.hstack(coeffs)
    return flattened_coeffs

def HHT_transform(data, max_imfs = 10):
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
        # amplitude & frequency
        hht_data.append(amplitude_envelope)
        hht_data.append(instantaneous_frequency)
    
    # flatten
    flattened_hht = np.hstack(hht_data)
    return flattened_hht

def sliding_window_feature_extraction(data_array, label_array,
                                      window_size=5000, step=2500,
                                      wavelet='coif5', level=5,
                                      max_imfs=20,
                                      valid_indices=None):
    n_samples = len(data_array)
    HHT_features_list = []
    DWT_features_list = []
    window_labels_list = []
    window_index = 0 
    
    for start in range(0, n_samples - window_size + 1, step):
        end = start + window_size
         # 若指定了 valid_indices，且目前的 window_index 不在其中 => 直接跳過
        if (valid_indices is not None) and (window_index not in valid_indices):
            window_index += 1
            continue
        # 1) 取該窗訊號
        window_signal = data_array[start:end]
        # 2) 取該窗對應的標籤序列
        window_label_seq = label_array[start:end]
        
        # ---- 執行 HHT 特徵 ----
        hht_fv = HHT_transform(window_signal, max_imfs=max_imfs)
        # ---- 執行 DWT 特徵 ----
        dwt_fv = wavelet_transform(window_signal, wavelet=wavelet, level=level)

        # ---- 決定該窗的整體標籤 (e.g. 多數決) ----
        #   這裡簡單用 value_counts().idxmax()
        unique_counts = {}
        for lbl in window_label_seq:
            unique_counts[lbl] = unique_counts.get(lbl, 0) + 1
        majority_label = max(unique_counts, key=unique_counts.get)

        # ---- 收集特徵 + 標籤 ----
        HHT_features_list.append(hht_fv)
        DWT_features_list.append(dwt_fv)
        window_labels_list.append(majority_label)
        window_index += 1
    
    # 轉成 numpy array
    HHT_features = np.array(HHT_features_list)
    DWT_features = np.array(DWT_features_list)
    labels = np.array(window_labels_list)
    
    return HHT_features, DWT_features, labels

# 檢查形狀是否一致的函數
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

print("開始做 DWT 與 HHT ")

def process_single_device(
    device_df, 
    device_features_df, 
    delt_device_features_df,
    output_dir,
    device_name,
    start_index=0, 
    LABEL='LoadEncoded',      # 這是我們每次呼叫時的「起始批次編號」
    wavelet='coif5',
    level=5,
    max_imfs=20,
    window_size=2000,
    step=2000,
    num = num,
    valid_indices=None 
):
   
    now_device = device_df["load"].iloc[0]
    print("now_device",now_device)

    le = preprocessing.LabelEncoder()
    device_df[LABEL] = le.fit_transform(device_df['load'].values.ravel())
    device_features_df[LABEL] = le.fit_transform(device_features_df['load'].values.ravel())

    # 為了 LabelEncode 'load'

    # (可選) 輸出每行特徵
    save_each_row_as_npy(device_features_df, delt_device_features_df, output_dir, prefix=f"class{num}_{device_name}")
    
    # 做簡單圖表 (可選)
    plt.figure(figsize=(6, 4))
    device_df['load'].value_counts().plot(kind='bar', title=device_name)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{device_name}.png')
    plt.savefig(save_path, dpi=200)
    plt.close()

    # sliding window
    data_array = device_df['current'].values.astype(float)
    label_array = device_df[LABEL].values.astype(int)
    valid_indices = valid_indices_per_appliance[now_device]

    HHT_features, DWT_features, labels = sliding_window_feature_extraction(
        data_array=data_array,
        label_array=label_array,
        window_size=window_size,
        step=step,
        wavelet=wavelet,
        level=level,
        max_imfs=max_imfs,
        valid_indices=valid_indices 

    )

    # 將結果逐 batch 輸出，但編號從 start_index 開始
    batch_counter = start_index
    for i in range(len(labels)):
        hht_i = np.expand_dims(HHT_features[i], axis=0)
        dwt_i = np.expand_dims(DWT_features[i], axis=0)
        label_i = np.array([labels[i]])

        np.save(os.path.join(output_dir, f"features_HHT_batch_{batch_counter}.npy"), hht_i)
        np.save(os.path.join(output_dir, f"features_DWT_batch_{batch_counter}.npy"), dwt_i)
        np.save(os.path.join(output_dir, f"labels_batch_{batch_counter}.npy"), label_i)

        batch_counter += 1  # 依序遞增

    # 回傳「本設備處理結束後的 batch_counter」
    return batch_counter

def class_process(num, class_num, class_features, class_delt_features,class_output_dir, 
                  wavelet='coif5', level=5,
                  max_imfs=20, window_size=5000, step=2500,
                  valid_indices=None):
    if not class_num:  # 如果沒有此類別
        clear_folder(class_output_dir)
        print(f"沒有找到任何 class{num}。")
        return
    
    # 建立 / 清空 class X 總目錄（可選，看您是否要這樣做）
    os.makedirs(class_output_dir, exist_ok=True)
    clear_folder(class_output_dir)

    batch_counter = 0  # 從 0 開始
    
    for i, (device_df, device_features_df, delt_device_features_df) in enumerate(zip(class_num, class_features, class_delt_features)):
        device_name = device_df['load'].iloc[0]  # 取第一筆 load 作設備名稱

        # 呼叫處理函式，並把 batch_counter 當成 start_index 傳進去
        batch_counter = process_single_device(
            device_df=device_df.copy(),
            device_features_df=device_features_df.copy(),
            delt_device_features_df = delt_device_features_df.copy(),
            output_dir=class_output_dir,
            device_name=f"{device_name}_device{i}",
            start_index = batch_counter,  # 這裡把目前的計數帶進去
            wavelet = wavelet,
            level = level,
            max_imfs = max_imfs,
            window_size = window_size,
            step = step,
            num = num,
            valid_indices=valid_indices
        )
        # process_single_device 會回傳「最新的 batch_counter」
        # 這樣下一台設備就會從上一次結束的編號繼續加
        
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

for classnumber in range(1, num + 1):
    class_process( 
        num = classnumber,
        class_num = class_lists[classnumber],
        class_features = class_features_lists[classnumber],
        class_delt_features = class_delt_features_lists[classnumber],
        class_output_dir = output_dirs_lists[classnumber],
        wavelet='coif5',
        level=level,
        max_imfs=max_imfs,
        window_size=window_size,
        step=step_size,
        valid_indices=valid_indices
        )

# 結束計時
end_time = time.time()

# 計算總執行時間                
execution_time = end_time - start_time
print(f"程式執行時間: {execution_time} 秒")