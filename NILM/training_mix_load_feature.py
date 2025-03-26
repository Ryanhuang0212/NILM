# 讀取必要函式庫
import numpy as np
import pandas as pd
import os
import time
from collections import deque
from tensorflow.keras.models import load_model
import tensorflow as tf

# 開始計時
start_time = time.time()
# 設置參數
window_size = 2000  
step_size = 2000  

# 資料夾路徑
class_1_output_dir = r'D:\graduate_info\Research\code\lab load\Class\Class 1_mix'  
new_appliance_input_folder = r"D:\graduate_info\Research\code\lab load\new_appliance_output_folder_combine"

# 載入已訓練的模型
model_path = r'D:\graduate_info\Research\code\lab load\Machine learning\1DCNN_model_DWT&HHT2_training_My_data_多分類旋轉機3.h5'
generator_path = r'D:\graduate_info\Research\code\lab load\Machine learning\generator.h5'
generator = load_model(generator_path)
trained_model = load_model(model_path)

print("✅ 已成功載入 Generator 和 CNN 模型！")

# 定義數據處理函式
def z_score_normalize(data):
    return (data - np.mean(data)) / (np.std(data) + 1e-8)

def process(folder_path):
    if not os.listdir(folder_path):
        print(f"資料夾 {folder_path} 為空，跳過處理。")
        return None
    
    features_HHT_list, labels_list, features_list, name_list, features_DWT_list, delt_features_list = [], [], [], [], [], []

    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        
        if 'features_HHT_batch' in file_name:
            features_HHT_list.append(np.load(file_path))
        elif 'labels_batch' in file_name:
            labels_list.append(np.load(file_path))
        elif 'load_features' in file_name:
            features_list.append(np.load(file_path))
        elif 'load_name' in file_name:
            name_list.append(np.load(file_path))
        elif 'features_DWT_batch' in file_name:
            features_DWT_list.append(np.load(file_path))
        elif 'load_delt_features' in file_name:
            delt_features_list.append(np.load(file_path))

    # 合併數據
    HHT_features = np.expand_dims(np.concatenate(features_HHT_list, axis=0), axis=-1)
    DWT_features = np.expand_dims(np.concatenate(features_DWT_list, axis=0), axis=-1)
    features = np.expand_dims(z_score_normalize(np.concatenate(features_list, axis=0)), axis=-1)
    delt_features = np.expand_dims(np.concatenate(delt_features_list, axis=0), axis=-1)
    labels = np.concatenate(labels_list, axis=0)
    names = np.expand_dims(np.concatenate(name_list, axis=0), axis=-1)

    return HHT_features, DWT_features, features, labels, names, delt_features

# 讀取數據
Class1_HHT, Class1_DWT, Class1_features, Class1_y, Class1_name, Class1_delt_features = process(class_1_output_dir)

# 讀取事件標籤
event_csv_path = r"D:\graduate_info\Research\code\lab load\new_appliance_output_folder_combine\Steam_Juice.csv"
df_event = pd.read_csv(event_csv_path)

if "event_status" not in df_event.columns:
    raise ValueError("❌ 錯誤: `event_status` 欄位不存在!")

event_status_list = df_event["event_status"].dropna().tolist()

# 平滑 `event_status_list`
def smooth_event_status(event_status_list, window_size=3):
    smoothed_status = []
    for i in range(len(event_status_list)):
        window = event_status_list[max(0, i - window_size + 1): i + 1]
        non_none_events = [status for status in window if status != "None"]
        smoothed_status.append(non_none_events[-1] if non_none_events else "None")
    return smoothed_status

filtered_event_status = smooth_event_status(event_status_list)

# 讀取最佳閾值
best_threshold_gmm = np.load(r"D:\graduate_info\Research\code\lab load\Machine learning\confidence_thresholds3.npy")
best_threshold_dict = {"Juice": best_threshold_gmm[1], "Steam": best_threshold_gmm[0]}
load_names = ["Juice", "Steam"]

# 自適應最佳閾值
def adjust_threshold(prob_values, initial_threshold=0.5, scale_factor=1.2):
    return max(initial_threshold, min(np.median(prob_values) * scale_factor, 0.9))

# 進行推理
def inference(HHT_test, DWT_test, features_test, delt_features_test):

    if delt_features_test.shape[1] < features_test.shape[1]:
        pad_size = 17 - delt_features_test.shape[1]
        delt_features_test = np.pad(delt_features_test, ((0, 0), (0, pad_size), (0, 0)), mode='constant')
        
    generated_features_test = generator.predict([features_test, features_test])

    predictions = trained_model.predict([HHT_test, DWT_test, features_test, delt_features_test, generated_features_test])

    for device in best_threshold_dict.keys():
        device_probs = predictions[:, load_names.index(device)]
        best_threshold_dict[device] = adjust_threshold(device_probs, best_threshold_dict[device])

    correct_predictions, total_samples = 0, len(predictions)
    active_devices = set()
    history = {device: deque(maxlen=10) for device in load_names}
    no_load_count = 0
    none_threshold_global = 0.01

    for i, vec in enumerate(predictions):
        current_active = set()
        no_load_detected = True

        for j, device in enumerate(load_names):
            device_prob = vec[j]
            device_threshold = best_threshold_dict[device]

            if device_prob > device_threshold:
                history[device].append(1)
                no_load_detected = False
            else:
                history[device].append(0)

            if sum(history[device]) / len(history[device]) >= 0.5:
                current_active.add(device)

        none_threshold_global = max(none_threshold_global, max(vec) * 0.5)

        if max(vec) < none_threshold_global:
            no_load_detected = True

        if no_load_detected:
            no_load_count += 1
            if no_load_count >= 5:
                active_devices.clear()
        else:
            no_load_count = 0

        turned_on = current_active - active_devices
        turned_off = active_devices - current_active
        active_devices = active_devices.union(turned_on) - turned_off

        csv_event_status = filtered_event_status[i]
        csv_event_set = set(csv_event_status.split("_")) if csv_event_status != "None" else set()

        if current_active == csv_event_set:
            correct_predictions += 1

    accuracy = correct_predictions / total_samples * 100
    print(f"📊 **模型準確率: {accuracy:.2f}%**")
    return predictions

# 執行推理
predictions = inference(Class1_HHT, Class1_DWT, Class1_features, Class1_delt_features)

# 計算執行時間
execution_time = time.time() - start_time
print(f"程式執行時間: {execution_time:.2f} 秒")