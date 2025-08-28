# Adversarial Learning 專案

## 專案概述
本專案是一個對抗式機器學習（Adversarial Learning）研究平台，專注於生成對抗式樣本（Adversarial Example）並測試深度學習模型的穩健性。該專案整合了模型訓練、對抗攻擊、防禦測試與結果分析等功能，並提供直觀的圖形化使用者介面，讓研究人員能夠輕鬆進行對抗式機器學習實驗。

## 資料集詳細說明

### Food101 資料集
- **路徑**: `Dataset/food101/`
- **描述**: 包含101種食物類別的大型資料集，本專案選用其中5個代表性類別
- **類別**: `tacos`, `takoyaki`, `tiramisu`, `tuna_tartare`, `waffles`
- **資料結構**:
  ```
  food101/
  ├── food101_trainSet/          # 訓練集
  │   ├── tacos/
  │   ├── takoyaki/
  │   ├── tiramisu/
  │   ├── tuna_tartare/
  │   └── waffles/
  ├── food101_testSet/           # 原始測試集
  └── ADV_test_*_food101_testSet/ # 對抗樣本測試集
  ```
- **對抗樣本**: 
  - `ADV_test_non_target_food101_testSet/`: Non-targeted 攻擊生成的對抗樣本
  - `ADV_test_target_food101_testSet/`: Targeted 攻擊生成的對抗樣本
  - 包含多輪攻擊結果 (round_39 到 round_50)

### IndiaFood80 資料集
- **路徑**: `Dataset/IndiaFood80/`
- **描述**: 專門針對印度料理的食物圖像資料集
- **資料結構**:
  ```
  IndiaFood80/
  ├── IndiaFood80_trainSet/      # 訓練集
  └── IndiaFood80_testSet/       # 測試集
  ```

## 模型架構與訓練策略

### 深度學習模型
- **基礎架構**: ResNet-18 (PyTorch實作)
- **預訓練**: 使用 ImageNet 預訓練權重
- **微調策略**: Transfer Learning
  - 凍結所有卷積層參數
  - 僅訓練最後的全連接分類層
  - 根據資料集動態調整輸出類別數

### 訓練配置
- **優化器**: Adam optimizer
- **學習率**: 自動調整
- **資料增強**: 標準化與尺寸調整
- **模型儲存**: `models/model_trained/food101_trainSet.pth`

## 對抗攻擊方法

### 1. FGSM (Fast Gradient Sign Method)
- **原理**: 單步梯度符號攻擊
- **特點**: 計算速度快，攻擊效果明顯
- **擾動強度**: ε參數可調

### 2. PGD (Projected Gradient Descent)
- **原理**: 多步迭代攻擊，更強的攻擊能力
- **特點**: 
  - 多輪迭代優化
  - 更精確的對抗式樣本生成
  - 較高的攻擊成功率

### 攻擊類型
- **Non-targeted Attack**: 讓模型錯誤分類到任意其他類別
- **Targeted Attack**: 讓模型錯誤分類到特定目標類別

## 核心功能模組

### 1. 模型訓練 (`src/train.py`)
- 支援多種資料集格式
- 自動化訓練流程
- 訓練進度追蹤
- 模型性能評估

### 2. 模型測試 (`src/test.py`)
- 計算分類準確率
- ROC-AUC性能指標
- 混淆矩陣(Confusion Matrix)分析
- 測試結果可視化

### 3. 對抗攻擊 (`src/attack.py`)
- 實作多種攻擊算法
- 批量對抗式樣本生成
- 攻擊參數自定義
- 攻擊效果評估

### 4. 結果記錄 (`src/csv_writer.py`)
- 實驗結果自動化記錄
- CSV格式數據輸出
- 支援大批量數據處理
- 結果追蹤與比較

### 5. 工具函數 (`src/utils.py`)
- 資料預處理
- 圖像變換
- 共用工具函數

## 圖形化使用者介面

### 主介面功能 (`GUIforAL.py`)
- **資料集選擇**: 支援多種資料集切換
- **模型管理**: 載入、訓練、儲存模型
- **攻擊配置**: 設定攻擊參數與目標
- **批量處理**: 大規模對抗樣本生成
- **進度監控**: 即時顯示處理進度
- **結果展示**: 攻擊成功率與模型性能統計

### 操作流程
1. 選擇資料集路徑
2. 載入或訓練模型
3. 設定攻擊參數
4. 執行對抗攻擊
5. 查看結果分析

## 實驗結果與輸出

### 結果文件
- `output_non_target_food101_testSet.csv`: Non-targeted攻擊結果
- `output_target_food101_testSet.csv`: Targeted攻擊結果

### 對抗樣本儲存
- 按攻擊輪次組織 (ADV_test_round_XX/)
- 保持原始資料夾結構
- 支援大批量樣本管理

## 安裝與環境需求

### Python依賴套件
```bash
torch>=1.9.0
torchvision>=0.10.0
numpy
PIL
tkinter
pandas
scikit-learn
```

### 系統需求
- Python 3.8+
- CUDA支援 (建議)
- 充足的儲存空間用於對抗樣本

## 使用指南

### 快速開始
1. **環境準備**
   ```bash
   pip install torch torchvision numpy pillow pandas scikit-learn
   ```

2. **資料準備**
   - 將資料集放置於 `Dataset/` 目錄下
   - 確保資料夾結構正確

3. **啟動介面**
   ```bash
   python GUIforAL.py
   ```

4. **執行實驗**
   - 透過GUI選擇資料集
   - 訓練或載入模型
   - 配置攻擊參數
   - 開始對抗攻擊實驗

### 進階使用
- **批量實驗**: 支援多輪次自動化攻擊
- **參數調優**: 可調整ε值、迭代次數等攻擊參數
- **結果分析**: 利用CSV輸出進行深度分析

## 專案結構
```
├── Adversarial_Learning.py    # 核心攻擊模組
├── GUIforAL.py               # 圖形化介面
├── Dataset/                  # 資料集目錄
├── models/                   # 模型儲存
├── src/                      # 核心程式碼
│   ├── attack.py            # 攻擊算法實作
│   ├── train.py             # 模型訓練
│   ├── test.py              # 模型測試
│   ├── csv_writer.py        # 結果記錄
│   └── utils.py             # 工具函數
└── README.md                # 專案說明
```

## 研究應用
- 深度學習模型穩健性評估
- 對抗樣本生成與分析
- 對抗訓練研究
- 模型安全性測試

## 技術特色
- 模組化設計，易於擴展
- 支援多種攻擊算法
- 完整的實驗記錄系統
- 直觀的圖形化操作介面
- 高效的批量處理能力

## 未來發展方向
- 支援更多攻擊算法 (C&W, AutoAttack等)
- 加入防禦機制評估
- 擴展至更多資料集
- 優化攻擊效率與品質
