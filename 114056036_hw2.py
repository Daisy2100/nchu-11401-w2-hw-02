#!/usr/bin/env python
# coding: utf-8

"""
房價預測多元線性回歸分析
NCHU 11401 W2 HW02

資料來源: https://www.kaggle.com/datasets/juhibhojani/house-price/discussion?sort=hotness
作者: 114056036
日期: 2025年10月22日
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_explore_data():
    """
    1. Business Understanding & 2. Data Understanding
    載入和探索資料
    """
    print("=== 1. Business Understanding ===")
    print("專案目標: 建立多元線性回歸模型預測房價")
    print("商業價值: 幫助房地產業者、投資者和買家評估房屋價值")
    print("成功標準: 提供準確且可解釋的價格預測")
    
    print("\n=== 2. Data Understanding ===")
    
    # 讀取資料
    df = pd.read_csv('dataset/house_prices.csv')
    
    print(f"資料集形狀: {df.shape}")
    print("\n欄位名稱:")
    print(df.columns.tolist())
    
    print("\n前5筆資料:")
    print(df.head())
    
    print("\n資料型態:")
    print(df.dtypes)
    
    print("\n缺失值統計:")
    print(df.isnull().sum())
    
    print("\n目標變數統計 (Price in rupees):")
    print(df['Price (in rupees)'].describe())
    
    return df

def clean_and_preprocess_data(df):
    """
    3. Data Preparation
    清理和預處理資料
    """
    print("\n=== 3. Data Preparation ===")
    
    # 複製資料
    df_clean = df.copy()
    
    # 移除明顯不需要的欄位
    columns_to_drop = ['Index', 'Title', 'Description', 'Amount(in rupees)', 'Society']
    df_clean = df_clean.drop(columns=[col for col in columns_to_drop if col in df_clean.columns])
    
    # 處理目標變數 - 移除異常值和缺失值
    df_clean = df_clean.dropna(subset=['Price (in rupees)'])
    df_clean = df_clean[df_clean['Price (in rupees)'] > 0]
    
    # 移除極端異常值 (使用 IQR 方法)
    Q1 = df_clean['Price (in rupees)'].quantile(0.25)
    Q3 = df_clean['Price (in rupees)'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_clean = df_clean[(df_clean['Price (in rupees)'] >= lower_bound) & 
                       (df_clean['Price (in rupees)'] <= upper_bound)]
    
    print(f"清理前資料量: {df.shape[0]}")
    print(f"清理後資料量: {df_clean.shape[0]}")
    print(f"保留比例: {df_clean.shape[0]/df.shape[0]*100:.1f}%")
    
    return df_clean

def feature_engineering(df):
    """
    特徵工程
    """
    df_features = df.copy()
    
    # 處理數值型特徵
    numerical_features = []
    
    # 處理 Carpet Area (地毯面積)
    if 'Carpet Area' in df_features.columns:
        df_features['Carpet_Area_Numeric'] = pd.to_numeric(
            df_features['Carpet Area'].str.extract(r'(\d+)')[0], errors='coerce')
        numerical_features.append('Carpet_Area_Numeric')
    
    # 處理 Super Area
    if 'Super Area' in df_features.columns:
        df_features['Super_Area_Numeric'] = pd.to_numeric(
            df_features['Super Area'].str.extract(r'(\d+)')[0], errors='coerce')
        numerical_features.append('Super_Area_Numeric')
    
    # 處理 Bathroom, Balcony, Car Parking
    for col in ['Bathroom', 'Balcony', 'Car Parking']:
        if col in df_features.columns:
            col_name = col.replace(' ', '_') + '_Numeric'
            df_features[col_name] = pd.to_numeric(
                df_features[col].str.extract(r'(\d+)')[0], errors='coerce')
            numerical_features.append(col_name)
    
    # 處理 Floor
    if 'Floor' in df_features.columns:
        df_features['Floor_Numeric'] = pd.to_numeric(
            df_features['Floor'].str.extract(r'(\d+)')[0], errors='coerce')
        numerical_features.append('Floor_Numeric')
    
    # 處理類別型特徵
    categorical_features = ['location', 'Status', 'Transaction', 'Furnishing', 
                          'facing', 'overlooking', 'Ownership']
    
    # 對類別變數進行編碼
    le_dict = {}
    for col in categorical_features:
        if col in df_features.columns:
            le = LabelEncoder()
            df_features[col + '_Encoded'] = le.fit_transform(df_features[col].fillna('Unknown'))
            le_dict[col] = le
            numerical_features.append(col + '_Encoded')
    
    return df_features, numerical_features, le_dict

def perform_feature_selection(X, y, k=10):
    """
    執行特徵選擇
    """
    # 方法1: SelectKBest (使用F統計量)
    selector_f = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
    X_selected_f = selector_f.fit_transform(X, y)
    selected_features_f = X.columns[selector_f.get_support()].tolist()
    
    # 方法2: 遞歸特徵消除 (RFE)
    lr_temp = LinearRegression()
    rfe = RFE(estimator=lr_temp, n_features_to_select=min(k, X.shape[1]))
    X_selected_rfe = rfe.fit_transform(X, y)
    selected_features_rfe = X.columns[rfe.support_].tolist()
    
    return (selected_features_f, selected_features_rfe, 
            selector_f.scores_, selector_f)

def build_and_train_model(X_selected, y):
    """
    4. Modeling
    建立和訓練模型
    """
    print("\n=== 4. Modeling ===")
    
    # 分割訓練和測試資料
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
    print(f"訓練集大小: {X_train.shape}")
    print(f"測試集大小: {X_test.shape}")
    
    # 標準化特徵
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 建立線性回歸模型
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    print("模型訓練完成！")
    
    return model, scaler, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

def evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, selected_features):
    """
    5. Evaluation
    評估模型
    """
    print("\n=== 5. Evaluation ===")
    
    # 模型預測
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # 計算評估指標
    def calculate_metrics(y_true, y_pred, set_name):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n{set_name} 評估結果:")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.4f}")
        
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
    
    # 計算訓練和測試結果
    train_metrics = calculate_metrics(y_train, y_train_pred, "訓練集")
    test_metrics = calculate_metrics(y_test, y_test_pred, "測試集")
    
    # 特徵重要性分析
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("\n特徵重要性排序:")
    print(feature_importance)
    
    return train_metrics, test_metrics, y_train_pred, y_test_pred, feature_importance

def calculate_prediction_intervals(X_test_scaled, y_test, y_pred, confidence_level=0.95):
    """
    計算預測區間
    """
    # 計算殘差標準誤
    residuals = y_test - y_pred
    mse = np.mean(residuals**2)
    std_error = np.sqrt(mse)
    
    # 計算t統計量 (假設常態分布)
    alpha = 1 - confidence_level
    dof = len(y_test) - X_test_scaled.shape[1] - 1
    t_val = stats.t.ppf(1 - alpha/2, dof)
    
    # 預測區間
    margin_error = t_val * std_error
    lower_bound = y_pred - margin_error
    upper_bound = y_pred + margin_error
    
    return lower_bound, upper_bound

def create_visualizations(y_train, y_test, y_train_pred, y_test_pred, train_metrics, test_metrics):
    """
    建立視覺化圖表
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 實際值 vs 預測值 (訓練集)
    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.5)
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('實際價格')
    axes[0, 0].set_ylabel('預測價格')
    axes[0, 0].set_title(f'訓練集: 實際值 vs 預測值 (R² = {train_metrics["R2"]:.4f})')
    
    # 2. 實際值 vs 預測值 (測試集)
    axes[0, 1].scatter(y_test, y_test_pred, alpha=0.5, color='orange')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('實際價格')
    axes[0, 1].set_ylabel('預測價格')
    axes[0, 1].set_title(f'測試集: 實際值 vs 預測值 (R² = {test_metrics["R2"]:.4f})')
    
    # 3. 殘差圖 (訓練集)
    train_residuals = y_train - y_train_pred
    axes[1, 0].scatter(y_train_pred, train_residuals, alpha=0.5)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('預測價格')
    axes[1, 0].set_ylabel('殘差')
    axes[1, 0].set_title('訓練集殘差圖')
    
    # 4. 殘差圖 (測試集)
    test_residuals = y_test - y_test_pred
    axes[1, 1].scatter(y_test_pred, test_residuals, alpha=0.5, color='orange')
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('預測價格')
    axes[1, 1].set_ylabel('殘差')
    axes[1, 1].set_title('測試集殘差圖')
    
    plt.tight_layout()
    plt.savefig('model_evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_prediction_interval_plot(X_test_scaled, y_test, y_test_pred):
    """
    建立預測區間圖
    """
    # 計算95%預測區間
    lower_bound, upper_bound = calculate_prediction_intervals(X_test_scaled, y_test, y_test_pred)
    
    # 繪製帶預測區間的圖
    plt.figure(figsize=(12, 8))
    
    # 選擇前100個點來清楚顯示
    n_show = min(100, len(y_test))
    indices = range(n_show)
    
    plt.errorbar(indices, y_test_pred[:n_show], 
                 yerr=[y_test_pred[:n_show] - lower_bound[:n_show], 
                       upper_bound[:n_show] - y_test_pred[:n_show]], 
                 fmt='o', alpha=0.6, capsize=3, label='預測值與95%預測區間')
    
    plt.scatter(indices, y_test.iloc[:n_show], color='red', alpha=0.7, label='實際值')
    
    plt.xlabel('樣本序號')
    plt.ylabel('房價 (盧比)')
    plt.title('房價預測結果與95%預測區間 (前100個樣本)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('prediction_intervals.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 計算預測區間覆蓋率
    coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
    print(f"\n95%預測區間覆蓋率: {coverage:.1%}")
    
    return coverage

def save_model_and_results(model, scaler, selector, train_metrics, test_metrics, coverage):
    """
    6. Deployment
    儲存模型和結果
    """
    print("\n=== 6. Deployment ===")
    
    # 儲存模型
    joblib.dump(model, 'house_price_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    joblib.dump(selector, 'feature_selector.pkl')
    
    print("模型已儲存完成！")
    print("\n最終模型評估結果總結:")
    print(f"訓練集 R²: {train_metrics['R2']:.4f}")
    print(f"測試集 R²: {test_metrics['R2']:.4f}")
    print(f"測試集 RMSE: {test_metrics['RMSE']:.2f}")
    print(f"預測區間覆蓋率: {coverage:.1%}")
    
    # 儲存結果到文字檔
    with open('model_results.txt', 'w', encoding='utf-8') as f:
        f.write("房價預測多元線性回歸分析結果\n")
        f.write("="*50 + "\n\n")
        f.write(f"訓練集 R²: {train_metrics['R2']:.4f}\n")
        f.write(f"測試集 R²: {test_metrics['R2']:.4f}\n")
        f.write(f"測試集 RMSE: {test_metrics['RMSE']:.2f}\n")
        f.write(f"預測區間覆蓋率: {coverage:.1%}\n")

def main():
    """
    主要執行函數 - 遵循 CRISP-DM 流程
    """
    # 1 & 2. Business Understanding & Data Understanding
    df = load_and_explore_data()
    
    # 3. Data Preparation
    df_clean = clean_and_preprocess_data(df)
    df_processed, feature_columns, label_encoders = feature_engineering(df_clean)
    
    # 準備建模資料
    valid_features = []
    for col in feature_columns:
        if col in df_processed.columns:
            missing_ratio = df_processed[col].isnull().sum() / len(df_processed)
            if missing_ratio < 0.5:
                valid_features.append(col)
    
    print(f"\n有效特徵數量: {len(valid_features)}")
    print("有效特徵:", valid_features)
    
    X = df_processed[valid_features].fillna(df_processed[valid_features].median())
    y = df_processed['Price (in rupees)']
    
    # 特徵選擇
    features_f, features_rfe, f_scores, selector = perform_feature_selection(X, y, k=10)
    
    print("\nSelectKBest 選擇的特徵:")
    for i, (feature, score) in enumerate(zip(features_f, f_scores[selector.get_support()])):
        print(f"{i+1}. {feature}: {score:.2f}")
    
    selected_features = features_f
    X_selected = X[selected_features]
    
    # 4. Modeling
    model, scaler, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = build_and_train_model(X_selected, y)
    
    # 5. Evaluation
    train_metrics, test_metrics, y_train_pred, y_test_pred, feature_importance = evaluate_model(
        model, X_train_scaled, X_test_scaled, y_train, y_test, selected_features)
    
    # 建立視覺化圖表
    create_visualizations(y_train, y_test, y_train_pred, y_test_pred, train_metrics, test_metrics)
    
    # 建立預測區間圖
    coverage = create_prediction_interval_plot(X_test_scaled, y_test, y_test_pred)
    
    # 6. Deployment
    save_model_and_results(model, scaler, selector, train_metrics, test_metrics, coverage)

if __name__ == "__main__":
    main()