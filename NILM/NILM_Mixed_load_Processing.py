from __future__ import print_function
import numpy as np
import pandas as pd
import os
import time
import json
import shutil
from scipy.fft import fft, fftfreq
# 開始計時
start_time = time.time()
print('前處理程式開始執行=============================================')

# 資料集
csv_folder_path = r"D:\graduate_info\Research\code\lab load\2025 lab mix"
meta_json_path = r"D:\graduate_info\Research\code\lab load\2025 lab mix\2025_lab_15s_aggregated_Steam_juice.json"

# =======================================================================

threshold = 0.3
window_size = 2000  
step_size = 2000   
sampling_rate = 2000 
base_freq = 60  
harmonics = list(range(1, 8)) 

max_length = 90000
# =======================================================================
# 定義特徵名稱
feature_names = ['RMS', 'Peak', 'Peak-to-Peak', 'Waveform Factor', 'Crest Factor', 
                 'Power', 'Power Std', 'vi area', 'Current Range', 'ZCR', 
                'Skewness' , 'Kurtosis','Delta Current Mean','Delta Current Std'
                ,"WindowIndex"]
# =======================================================================

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

# 初始化字典存儲每種電器類型的數據
appliance_data = {}

# 加載 JSON 文件
with open(meta_json_path, "r") as file:
    json_data = json.load(file)

# 初始化空的 DataFrame 存放混合負載數據
mixed_load_data = pd.DataFrame()

# 遍歷 JSON 文件的每個條目
for key, entry in json_data.items():
    if "appliances" not in entry:
        continue  # 如果沒有 "appliances" 鍵，跳過

    appliances = entry["appliances"]

    # 生成設備組合的描述
    appliance_types = [appliance["type"] for appliance in entry["appliances"]]
    combination_label = "_".join(appliance_types)  # 用 "_" 拼接設備類型作為組合標籤

    # 找到對應的 CSV 文件
    csv_file_path = os.path.join(csv_folder_path, f"2025_steam_juice.csv")
    if not os.path.exists(csv_file_path):
        print(f"找不到對應的 CSV 文件: {csv_file_path}")
        continue

    # 讀取 CSV 文件
    df = pd.read_csv(csv_file_path, header=None, names=["voltage","current"])
    df["combination"] = combination_label  # 添加組合標籤列
    df["csv_id"] = key  # 添加文件 ID 列

     # 🛠️ 初始化電器狀態
    event_status_list = []
    current_status = set()

    # 遍歷 CSV 內的每個取樣點
    for row_idx in range(len(df)):
        for ap in appliances:
            dev_type = ap.get("type", "Unknown")

            on_list = [int(int(x))for x in ap["on"].strip("[]").split()]
            off_list = [int(int(x)) for x in ap["off"].strip("[]").split()]

            # 若 row_idx 到達開啟點，則加入設備
            if row_idx in on_list:
                current_status.add(dev_type)  # **開啟設備**
            if row_idx in off_list:
                current_status.discard(dev_type)  # **關閉設備**

        # 記錄當前狀態
        if current_status:
            event_status_list.append("_".join(sorted(current_status)))
        else:
            event_status_list.append("None")

    # print(f"{dev_type}: on={on_list}, off={off_list}")
    # 🛠️ 將事件資訊加入 DataFrame

    df["event_status"] = event_status_list

    # 🛠️ 將數據合併到總表
    mixed_load_data = pd.concat([mixed_load_data, df], ignore_index=True)

# 🔹 輸出檢查
print("✅ 合併後的 mixed_load_data:")
print(mixed_load_data.head(30))  # 查看前 30 筆
print(f"✅ 總資料筆數: {len(mixed_load_data)}")

# 分組並合併每種設備類型的數據
for appliance_type, group in mixed_load_data.groupby("combination"):
    # 每組是一個 DataFrame
    combined_data = group.reset_index(drop=True)
    print(f"已合併 {appliance_type} 的數據，共 {len(combined_data)} 行")

original_appliance_output_folder = r"D:\graduate_info\Research\code\lab load\original_appliance_output_folder_combine"
# 直接用 mixed_load_data 依 combination 分組輸出
current_data = {}
voltage_data = {}
event_data = {}
appliance_names = list(current_data.keys()) 
original_appliance_data = {}
#  按 "combination" 分組
for appliance_name, group in mixed_load_data.groupby("combination"):
    # 提取每種設備的 current 和 voltage 數據
    current_data[appliance_name] = group["current"].values
    voltage_data[appliance_name] = group["voltage"].values
    event_data[appliance_name] = group["event_status"].values
    appliance_names.append(appliance_name) 

    # 遍歷 appliance_names，處理每個電器的數據
for appliance_name in appliance_names:
    # 提取 current 和 voltage
    appliance_current = current_data[appliance_name]
    appliance_voltage = voltage_data[appliance_name]
    appliance_event = event_data[appliance_name]
    
    # 建立 DataFrame 並保存到 new_appliance_data
    original_appliance_data[appliance_name] = pd.DataFrame({
        "current": appliance_current,
        "voltage": appliance_voltage,
        "event_status": appliance_event,
        "load": appliance_name
    })

if not os.path.exists(original_appliance_output_folder):
    os.makedirs(original_appliance_output_folder)

clear_folder(original_appliance_output_folder)

for appliance_name, df in original_appliance_data.items():
        csv_filename = f"{appliance_name}.csv"
        csv_path = os.path.join(original_appliance_output_folder, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"{csv_filename} 已存至 {csv_path}")

# 定義一個函數，用於過採樣重複數據
def oversample_data(data, target_length):
    if len(data) == 0:
        return np.zeros(target_length)  # 避免空數據
    repeats = target_length // len(data)  # 計算完整重複的次數
    remainder = target_length % len(data)  # 計算不足的部分
    return np.concatenate([data] * repeats + [data[:remainder]])

new_appliance_output_folder = r"D:\graduate_info\Research\code\lab load\new_appliance_output_folder_combine"
new_appliance_data = {}

# 遍歷 appliance_names，處理每個電器的數據
for appliance_name in appliance_names:
    # 提取 current 和 voltage
    appliance_current = current_data[appliance_name]
    appliance_voltage = voltage_data[appliance_name]
    appliance_event = event_data[appliance_name]
    
    # 過採樣
    oversampled_current = oversample_data(appliance_current, max_length)
    oversampled_voltage = oversample_data(appliance_voltage, max_length)
    oversampled_event = oversample_data(appliance_event, max_length)
    
    # 建立 DataFrame 並保存到 new_appliance_data
    new_appliance_data[appliance_name] = pd.DataFrame({
        "current": oversampled_current,
        "voltage": oversampled_voltage,
        "event_status": oversampled_event,
        "load": appliance_name
    })

if not os.path.exists(new_appliance_output_folder):
    os.makedirs(new_appliance_output_folder)

clear_folder(new_appliance_output_folder)

for appliance_name, df in new_appliance_data.items():
    # print('appliance_name: ', appliance_name)
    # if appliance_name.startswith("Air Conditioner"):
        csv_filename = f"{appliance_name}.csv"
        csv_path = os.path.join(new_appliance_output_folder, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"{csv_filename} 已存至 {csv_path}")

# 驗證結構
for name, df in new_appliance_data.items():
    print(f"{name}: {df.shape}")
    print(df.head())

print("=======================================================================")

current_data_dict = {}
voltage_data_dict = {}

# 遍歷 appliance_names，處理每個電器的數據
for appliance_name, df in new_appliance_data.items():
    current_data_dict[appliance_name] = df["current"].values
    voltage_data_dict[appliance_name] = df["voltage"].values

# 將所有設備的 current_data 整合到列表中
current_data_list = list(current_data_dict.values())
voltage_data_list = list(voltage_data_dict.values())

# 移動視窗取特徵
# 計算 RMS, Peak 和 Peak-to-Peak 的函數
def calculate_metrics(window):
    window = np.array(window)
    window -= np.mean(window)  # 去除 DC 分量
    rms = np.sqrt(np.mean(window**2))
    peak = np.max(window)
    peak_to_peak = np.ptp(window)

    if np.mean(np.abs(window)) == 0:  
        waveform_factor = 0
    else:
        waveform_factor = rms / np.mean(np.abs(window))

    if rms == 0:
        crest_factor = 0
    else:
        crest_factor = peak / rms

    current_range = np.max(window) - np.min(window)

    zero_crossings = np.where(np.diff(np.sign(window)))[0].size
    zero_crossing_rate = zero_crossings / (len(window) - 1) if len(window) > 1 else 0

    X_min, X_max = np.min(window), np.max(window)
    if X_max != X_min:
        X_normalized = 2 * (window - X_min) / (X_max - X_min) - 1
    else:
        X_normalized = np.zeros_like(window)  # 避免 NaN
    
    angles = np.arccos(np.clip(X_normalized, -1, 1))
    GAF = np.cos(angles[:, None] + angles[None, :])
    diag_GAF = np.diag(GAF)

    # ✅ **防止 NaN**
    if np.isnan([rms, peak, peak_to_peak, waveform_factor, crest_factor, current_range, zero_crossing_rate]).any():
        print(f"⚠️ calculate_metrics() 內部發現 NaN，檢查數據 window: {window}")

    return rms, peak, peak_to_peak, waveform_factor, crest_factor, GAF, current_range, zero_crossing_rate

# 計算電流變化率 (ΔCurrent)
def compute_delta_current(window):
    window_shifted = window[:-1].copy()
    window_shifted[window_shifted == 0] = 1e-6  # 避免除以 0

    delta_current = np.diff(window) / window_shifted  # 計算變化率
    if delta_current.size > 0:
        delta_current = np.append(delta_current, delta_current[-1])  # 確保維度一致
    else:
        delta_current = np.zeros(1)  # 避免索引錯誤

    # 移除 NaN 或 Inf
    delta_current = np.nan_to_num(delta_current, nan=0.0, posinf=0.0, neginf=0.0)

    return np.mean(delta_current), np.std(delta_current)  # 返回平均變化率 & 標準差

def process_windowed_data(current_data, voltage_data, threshold, window_size, step_size):
    feature_list = []
    delta_feature_list = []
    prev_features = None
    window_indices = []
    window_index = 0  # 記錄視窗編號

    for i in range(0, len(current_data) - window_size + 1, step_size):
        current_window = current_data[i:i + window_size]
        voltage_window = voltage_data[i:i + window_size]

        if np.isnan(current_window).any() or np.isnan(voltage_window).any():
            print(f"⚠️ 窗口 {window_index} 包含 NaN，跳過！")
            continue  # **直接跳過 NaN 窗口**
        
        if len(current_window) < window_size:
            pad_length = window_size - len(current_window)
            current_window = np.pad(current_window, (0, pad_length), mode='constant', constant_values=0)
            voltage_window = np.pad(voltage_window, (0, pad_length), mode='constant', constant_values=0)
            print(f"⚠️ 窗口 {window_index} 點數不足，已補 0")


        if np.max(current_window) < threshold:
            window_index += 1  # 即使跳過也要遞增索引
            continue  # 不儲存該視窗的特徵
        try:
            rms, peak, peak_to_peak, waveform_factor, crest_factor, GAF, current_range, zcr = calculate_metrics(current_window)
            power = np.mean(current_window * voltage_window)
            power_std = np.std(current_window * voltage_window)
            vi_area = np.trapz(y=current_window, x=voltage_window)
            skewness = pd.Series(current_window).skew()
            kurtosis = pd.Series(current_window).kurt()
            delta_current_mean_val, delta_current_std_val = compute_delta_current(current_window)
            power = np.mean(current_window * voltage_window)

            # ✅ **防止 NaN 或極端數值**
            if np.isnan([power, power_std]).any():
                print(f"⚠️ 窗口 {window_index} 的 power 計算出現 NaN，current_window: {current_window}, voltage_window: {voltage_window}")

            if abs(power) > 1e10 or abs(power_std) > 1e10:
                print(f"⚠️ 窗口 {window_index} 的 power 出現異常數值，power: {power}, power_std: {power_std}")

            # 當前窗口特徵
            feature_vector = np.array([rms, peak, peak_to_peak, waveform_factor, crest_factor,
                                        power, power_std, vi_area, current_range, zcr, 
                                        skewness, kurtosis, delta_current_mean_val, delta_current_std_val])
            
            feature_vector = np.append(feature_vector, window_index)

            # **檢查 NaN**
            if np.isnan(feature_vector).any():
                print(f"⚠️ 警告: 在窗口 {window_index} 發現 NaN 值，可能來自計算錯誤")
                print(feature_vector)  # 列印問題視窗的特徵

            # **排除不該計算變化量的特徵**
            exclude_features = ["Delta Current Mean", "Delta Current Std", "WindowIndex"]
            exclude_indices = [feature_names.index(feat) for feat in exclude_features]

            if prev_features is not None:
                delta_vector = np.array([
                    feature_vector[i] - prev_features[i] if i not in exclude_indices else 0
                    for i in range(len(feature_vector))
                ])
            else:
                delta_vector = np.zeros_like(feature_vector)  # 第一個窗口沒有變化量，設為 0

            prev_features = feature_vector


            # 存儲原始特徵 & 變化量特徵
            feature_list.append(feature_vector)
            delta_feature_list.append(delta_vector)
            window_indices.append(window_index)
            window_index += 1  # 每次迴圈都遞增視窗索引

        except Exception as e:
            print(f"❌ 錯誤: 窗口 {window_index} 無法計算特徵，錯誤訊息: {e}")


    return np.array(feature_list), np.array(delta_feature_list), np.array(window_indices)

# 初始化結果存儲結構
all_features_results = []
all_delta_features_results = []
all_window_indices = []
# 遍歷所有數據集進行移動視窗處理
for idx, (current_data, voltage_data) in enumerate(zip(current_data_list, voltage_data_list)):
    print(f"正在處理 current_data{idx + 1}...")

    # 取得原始特徵 & 變化量特徵
    features, delta_features, window_box = process_windowed_data(
        current_data, voltage_data, threshold, window_size, step_size
    )

    # 轉為 DataFrame 存儲
    all_features_results.append(pd.DataFrame(features))
    all_delta_features_results.append(pd.DataFrame(delta_features))
    all_window_indices.append(pd.DataFrame(window_box))

print("所有數據集的移動視窗處理完成。")

# 提取特定窗口內的諧波特徵
def extract_harmonics(fft_vals, freqs, base_freq, harmonics):
    harmonic_amplitudes = []    
    harmonic_phases = []
    for h in harmonics:
        # 找到最接近 h 倍基波頻率的頻率分量
        idx = np.argmin(np.abs(freqs - h * base_freq))
        harmonic_amplitudes.append(np.abs(fft_vals[idx]))
        harmonic_phases.append(np.angle(fft_vals[idx]))  # 提取相位角
    return harmonic_amplitudes, harmonic_phases

# 計算每個窗口的FFT並提取諧波特徵
def get_harmonics_windowed(current_data, voltage_data, window_size, step_size, sampling_rate, base_freq, harmonics, threshold):
    harmonic_results = []
    thd_results = [] 
    phase_results = []
    pq_results = []
    window_indices = []

    window_index = 0

    for start in range(0, len(current_data) - window_size + 1, step_size):
        current_window  = current_data[start:start + window_size]
        voltage_window = voltage_data[start:start + window_size]
        if len(current_window) < window_size:
            current_window = np.pad(current_window, (0, window_size - len(current_window)), 'constant')
            voltage_window = np.pad(voltage_window, (0, window_size - len(voltage_window)), 'constant')

        if np.max(current_window) < threshold:
            window_index += 1  # 即使跳過也要遞增索引
            continue  

        # 计算 FFT
        fft_vals_current = fft(current_window)
        fft_vals_voltage = fft(voltage_window)

        fft_vals_current = fft_vals_current[:len(fft_vals_current) // 2]  # 取前半部分的 FFT 结果
        fft_vals_voltage = fft_vals_voltage[:len(fft_vals_voltage) // 2]
        # 提取頻率
        freqs = fftfreq(window_size, d=1.0 / sampling_rate)[:len(fft_vals_current)]

        # 提取谐波特征（幅值和相位角）
        harmonic_amplitudes_current, harmonic_phases_current = extract_harmonics(fft_vals_current, freqs, base_freq, harmonics)
        harmonic_amplitudes_voltage, harmonic_phases_voltage = extract_harmonics(fft_vals_voltage, freqs, base_freq, harmonics)

        harmonic_results.append(harmonic_amplitudes_current)
        phase_results.append(harmonic_phases_current)

        # 计算 PQ（有功和无功功率）
        P = 0
        Q = 0

        for v_amp, i_amp, v_phase, i_phase in zip(harmonic_amplitudes_voltage, harmonic_amplitudes_current, harmonic_phases_voltage, harmonic_phases_current):
            angle_diff = v_phase - i_phase
            P += v_amp * i_amp * np.cos(angle_diff)
            Q += v_amp * i_amp * np.sin(angle_diff)
        
        pq_results.append((P, Q))

        # 计算 THD
        V1 = harmonic_amplitudes_current[0]  # 基波
        if V1 != 0:  # 避免除以 0
            thd = np.sqrt(np.sum(np.array(harmonic_amplitudes_current[1:])**2)) / V1 * 100  # 计算 THD
        else:
            thd = 0  # 如果基波为 0，则 THD 为 0
        thd_results.append(thd)
        
        window_indices.append(window_index)  # 記錄視窗索引
        window_index += 1  # 每次迴圈都遞增視窗索引


    return np.array(harmonic_results), np.array(phase_results), np.array(thd_results), np.array(pq_results), np.array(window_indices)

# 定義設備列表
harmonics_results = []
phase_results = []
thd_results_all_devices = []
pq_results_all_devices = []
window_index = []

# 對每個設備的電流數據進行移動窗口 FFT 並提取諧波特徵

for current_data, voltage_data in zip(current_data_list, voltage_data_list):
    harmonics_current_data, phases_current_data, thd_current_data, pq_current_data, window_box = get_harmonics_windowed(
        current_data, voltage_data, window_size, step_size, sampling_rate, base_freq, harmonics, threshold
    )
    harmonics_results.append(pd.DataFrame(harmonics_current_data))
    phase_results.append(pd.DataFrame(phases_current_data))
    thd_results_all_devices.append(pd.DataFrame(thd_current_data))
    pq_results_all_devices.append(pd.DataFrame(pq_current_data, columns=["P", "Q"]))
    window_index.append(pd.DataFrame(window_box))




# 定義變化量特徵名稱 (加上 Delta_ 前綴)
delta_feature_names = [f"Delta_{name}" for name in feature_names]

for i, appliance_name in enumerate(appliance_names):
    print(f"🔍 檢查 {appliance_name}:")
    
    # 檢查 features 結構
    if i >= len(all_features_results) or all_features_results[i].empty:
        print(f"⚠️ {appliance_name} 的 all_features_results[{i}] 為空或超出範圍")
        continue

    if i >= len(all_delta_features_results) or all_delta_features_results[i].empty:
        print(f"⚠️ {appliance_name} 的 all_delta_features_results[{i}] 為空或超出範圍")
        continue

    if all_features_results[i].shape[1] != len(feature_names):
        print(f"⚠️ {appliance_name} 的 feature 數量 ({all_features_results[i].shape[1]}) 與定義的特徵數 ({len(feature_names)}) 不匹配")
    
    if all_delta_features_results[i].shape[1] != len(delta_feature_names):
        print(f"⚠️ {appliance_name} 的 delta feature 數量 ({all_delta_features_results[i].shape[1]}) 與定義的特徵數 ({len(delta_feature_names)}) 不匹配")

# 處理每個設備
devices = {}
for appliance_name, i in zip(appliance_names, range(len(all_features_results))):  
    if all_features_results[i].empty or all_delta_features_results[i].empty:
        print(f"⚠️ 警告: {appliance_name} 沒有有效特徵，跳過")
        continue

    # 建立 DataFrame
    device_data = {}

    for j, feature_name in enumerate(feature_names):
        if j >= all_features_results[i].shape[1]:  # 確保 j 在範圍內
            print(f"⚠️ 錯誤: {appliance_name} 的 features 沒有第 {j} 列")
            continue
        device_data[feature_name] = all_features_results[i].iloc[:, j]

    for j, delta_feature_name in enumerate(delta_feature_names):
        if j >= all_delta_features_results[i].shape[1]:  # 確保 j 在範圍內
            print(f"⚠️ 錯誤: {appliance_name} 的 delta_features 沒有第 {j} 列")
            continue 
        device_data[delta_feature_name] = all_delta_features_results[i].iloc[:, j]

    # 添加 Harmonics 和 Phase 特徵
    harmonics_data = harmonics_results[i]
    phase_data = phase_results[i]

    for j in range(7):  # 7 個諧波
        device_data[f'Harmonics{j + 1}'] = harmonics_data.iloc[:, j].values
        device_data[f'phase{j + 1}'] = phase_data.iloc[:, j].values

    # P, Q, THD
    device_data['P'] = pq_results_all_devices[i]['P'].values
    device_data['Q'] = pq_results_all_devices[i]['Q'].values    
    device_data['THD'] = thd_results_all_devices[i].iloc[:, 0]

    # 存入設備字典
    devices[f'{appliance_name}'] = pd.DataFrame(device_data)

# 驗證結構
for device_name, df in devices.items():
    print(f"{device_name}: {df.shape}")
    print(df.head())

# 確認 devices 結構
print(type(devices))
print(devices)
# 確認 devices 結構
print(type(devices))
print(devices)

# 初始化存儲特定負載的字典
filtered_devices = {}

# 遍歷所有設備
for appliance_name, df in devices.items(): # 判斷設備名稱是否以目標前綴開頭
    filtered_devices[appliance_name] = df
    print(f"提取的負載: {appliance_name}, 數據量: {df.shape[0]} 行")

# 確認篩選結果
print(f"共篩選出 {len(filtered_devices)} 種負載")

# 定義存檔路徑
filtered_output_folder = r"D:\graduate_info\Research\code\lab load\Devices_Mixed_load_Feature"
os.makedirs(filtered_output_folder, exist_ok=True)
clear_folder(filtered_output_folder)

# 保存篩選後的數據
for device_name, device_df in filtered_devices.items():
    df.fillna(0, inplace=True)
    file_name = f"{device_name}.csv"
    file_path = os.path.join(filtered_output_folder, file_name)
    device_df.to_csv(file_path, index=False)
    print(f"{device_name} 的數據已保存至: {file_path}")

# 結束計時
end_time = time.time()

# 計算總執行時間                
execution_time = end_time - start_time
print(f"程式執行時間: {execution_time} 秒")

