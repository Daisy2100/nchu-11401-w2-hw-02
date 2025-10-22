# 房價預測多元線性回歸分析

**學號**: 114056036  
**GitHub Repository**: [https://github.com/Daisy2100/nchu-11401-w2-hw-02](https://github.com/Daisy2100/nchu-11401-w2-hw-02)  
**課程**: NCHU 11401 W2 HW02  
**日期**: 2025年10月22日  

## 專案概述
本專案使用 Kaggle 的房價資料集，遵循 CRISP-DM 流程完成多元線性回歸分析，預測房屋價格。

> 🎆 **作業選擇**: 本專案選擇上傳至 GitHub，並在 README.md 中整理流程與成果。

## 資料來源
- **資料集**: House Price Dataset
- **來源**: [Kaggle - House Price Dataset](https://www.kaggle.com/datasets/juhibhojani/house-price/discussion?sort=hotness)
- **特徵數量**: 21個特徵
- **資料筆數**: 187,531筆

## 專案結構
```
nchu-11401-w2-hw-02/
├── dataset/
│   └── house_prices.csv               # 原始資料集
├── 114056036_hw2.py                 # Python 主程式 (基礎版)
├── 114056036_hw2.ipynb              # Jupyter Notebook
├── 114056036_hw2_enhanced.py        # Python 主程式 (增強版，推薦)
├── 房價預測分析報告.md                # 完整分析報告
├── README.md                         # 專案說明文件
├── model_results.txt                 # 基礎版模型評估結果
├── enhanced_model_results.txt        # 增強版模型評估結果
├── house_price_model.pkl             # 基礎版訓練模型
├── best_house_price_model.pkl        # 最佳訓練模型 (增強版)
├── feature_scaler.pkl                # 基礎版特徵標準化器
├── best_feature_scaler.pkl           # 最佳特徵標準化器 (增強版)
├── feature_selector.pkl              # 基礎版特徵選擇器
├── best_feature_selector.pkl         # 最佳特徵選擇器 (增強版)
├── model_evaluation_plots.png        # 基礎版模型評估圖表
├── enhanced_model_analysis.png       # 增強版綜合分析圖表
└── prediction_intervals.png          # 預測區間圖表
```

## CRISP-DM 流程

### 1. Business Understanding (商業理解)
- **目標**: 建立準確的房價預測模型
- **商業價值**: 協助房地產業者、投資者和買家評估房屋價值
- **成功指標**: R² > 0.7，RMSE 合理範圍內

### 2. Data Understanding (資料理解)
- 資料集包含 21 個特徵，涵蓋房屋面積、位置、設施等
- 目標變數為房價 (Price in rupees)
- 檢查資料品質、缺失值和異常值

### 3. Data Preparation (資料準備)
- 移除不相關特徵 (Title, Description 等)
- 處理缺失值和異常值
- 特徵工程：數值化類別變數、提取數值特徵
- 使用 IQR 方法移除極端異常值

### 4. Modeling (建模)
- 特徵選擇：SelectKBest 和 RFE 方法
- 選擇最重要的 10 個特徵
- 標準化特徵
- 訓練多元線性回歸模型

### 5. Evaluation (評估)
- 評估指標：MSE, RMSE, MAE, R²
- 視覺化：實際值 vs 預測值、殘差圖
- 預測區間分析 (95% 信賴區間)
- 特徵重要性分析

### 6. Deployment (部署)
- 模型儲存和版本管理
- 結果文檔化
- 實務應用建議

## 使用方法

### 環境需求
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib
```

### 執行程式

#### 方法 1: 執行 Python 檔案
```bash
python 114056036_hw2.py
# 或執行增強版 (推薦)
python 114056036_hw2_enhanced.py
```

#### 方法 2: 使用 Jupyter Notebook
```bash
jupyter notebook 114056036_hw2.ipynb
```

## 模型效能

### 基礎版模型
- **訓練集 R²**: 0.1508
- **測試集 R²**: 0.1467
- **測試集 RMSE**: 2,871.02
- **預測區間覆蓋率**: 94.8%

### 增強版模型 (推薦)
- **訓練集 R²**: 0.2019
- **測試集 R²**: 0.2025
- **測試集 RMSE**: 3,558.72
- **交叉驗證 R²**: 0.2017 ± 0.0050
- **最佳模型**: Lasso Regression

### 特徵重要性
模型會輸出影響房價最重要的 10 個特徵及其重要性分數。

## 視覺化結果

### 1. 模型評估圖表
- 訓練集和測試集的實際值 vs 預測值散佈圖
- 殘差分析圖

### 2. 預測區間圖
- 顯示前 100 個樣本的預測值及 95% 預測區間
- 實際值與預測區間的比較

## AI 協助說明

### GPT 輔助內容
本專案在資料探索、程式架構、特徵工程、模型選擇、視覺化設計、錯誤排查（例如 GitHub 大檔案問題）等多個環節，使用了 GitHub Copilot（模型標示為 GPT-5 mini）協助。以下為整理後重點摘要，完整逐字對話請參閱專案檔案 `log.md`。

- 資料來源與檢視：確認 Kaggle 原始資料集與欄位（約 21 個特徵、187,531 筆），並檢查缺失值與資料型態。
- CRISP-DM 流程：依 Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment 完成分析流程。
- 資料清理：使用正規表示式抽取數值（如 "1200 sqft"、"2 BHK"），對類別欄位做熱門群組化（保留前 N 個地點，其他合成為 "Other"），並用 1%-99% 分位數或 IQR 處理極端值。
- 特徵工程與選擇：建立衍生特徵（設施總數、面積比等），並使用 SelectKBest (f_regression) 與 RFE 做特徵選擇。
- 建模策略：比較 Linear, Ridge, Lasso；嘗試對數轉換與正則化；使用標準化與交叉驗證評估模型穩定性。
- 成果摘要：基礎線性回歸 R² ≈ 0.15，增強版（含更嚴格的預處理與 Lasso）測試 R² 提升至 ≈ 0.2025，預測區間覆蓋率約 94.8%。
- 工程與版本控制：發現並解決 GitHub push 被拒的問題（csv 檔超過 100MB），更新 `.gitignore` 並移除該檔案於 commit 歷史後強制推送。

> 補充：本段為對話重點整理；若需逐字稿或完整紀錄，請打開 `log.md`（專案根目錄）。

### NotebookLM 研究摘要
專案同時準備了一份供 NotebookLM 上傳的研究摘要檔 `NotebookLM_summary.md`（位於專案根目錄），內含對網路上同題解法的整理與比較。

## 模型限制與改進建議

### 限制
1. 線性回歸假設特徵與目標呈線性關係
2. 對異常值敏感
3. 無法捕捉複雜的非線性關係

### 改進建議
1. 嘗試多項式回歸或其他非線性模型
2. 加入更多特徵工程 (如特徵交互作用)
3. 使用正則化方法 (Ridge, Lasso) 防止過擬合
4. 定期更新模型以反映市場變化

## 實務應用

### 適用場景
- 房地產初步估價
- 投資決策支援
- 市場趨勢分析

### 使用注意事項
- 模型基於歷史資料，需定期更新
- 預測結果應結合專業判斷使用
- 適用於類似市場環境的房產評估

## 作業要求檢查清單

- [x] 使用 10-20 個特徵的資料集
- [x] 遵循 CRISP-DM 流程
- [x] 實施特徵選擇
- [x] 執行模型評估
- [x] 提供預測圖與信賴區間
- [x] 準備完整的程式檔案 (.py 和 .ipynb)
- [ ] GPT 對話過程 PDF
- [ ] NotebookLM 研究摘要 (100字以上)
- [ ] 網路解法比較分析

## 參考資料
- [Kaggle House Price Dataset](https://www.kaggle.com/datasets/juhibhojani/house-price/discussion?sort=hotness)
- Scikit-learn 機器學習文檔
- CRISP-DM 方法論

## 聯絡資訊
- 學號: [請填入實際學號]
- 課程: NCHU 11401 W2
- 日期: 2025年10月22日