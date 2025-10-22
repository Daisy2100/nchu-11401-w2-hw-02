#!/usr/bin/env python
# coding: utf-8

"""
房價預測多元線性回歸分析 - 改進版
NCHU 11401 W2 HW02

主要改進：
1. 更好的資料預處理
2. 特徵工程優化
3. 對數轉換目標變數
4. 更嚴格的異常值處理
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def enhanced_data_preprocessing(df):
    """
    增強版資料預處理
    """
    print("\n=== 增強版資料預處理 ===")
    
    df_enhanced = df.copy()
    
    # 移除明顯無用的欄位
    columns_to_drop = ['Index', 'Title', 'Description', 'Amount(in rupees)', 'Society']
    df_enhanced = df_enhanced.drop(columns=[col for col in columns_to_drop if col in df_enhanced.columns])
    
    # 處理目標變數
    df_enhanced = df_enhanced.dropna(subset=['Price (in rupees)'])
    df_enhanced = df_enhanced[df_enhanced['Price (in rupees)'] > 0]
    
    # 使用更嚴格的異常值處理 (1st and 99th percentiles)
    price_lower = df_enhanced['Price (in rupees)'].quantile(0.01)
    price_upper = df_enhanced['Price (in rupees)'].quantile(0.99)
    df_enhanced = df_enhanced[(df_enhanced['Price (in rupees)'] >= price_lower) & 
                             (df_enhanced['Price (in rupees)'] <= price_upper)]
    
    print(f"原始資料: {df.shape[0]} 筆")
    print(f"清理後: {df_enhanced.shape[0]} 筆")
    print(f"保留比例: {df_enhanced.shape[0]/df.shape[0]*100:.1f}%")
    
    return df_enhanced

def advanced_feature_engineering(df):
    """
    進階特徵工程
    """
    print("\n=== 進階特徵工程 ===")
    
    df_features = df.copy()
    feature_list = []
    
    # 1. 數值型特徵提取和清理
    # Carpet Area
    if 'Carpet Area' in df_features.columns:
        df_features['Carpet_Area'] = pd.to_numeric(
            df_features['Carpet Area'].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
        feature_list.append('Carpet_Area')
    
    # Super Area  
    if 'Super Area' in df_features.columns:
        df_features['Super_Area'] = pd.to_numeric(
            df_features['Super Area'].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
        feature_list.append('Super_Area')
    
    # Bathroom, Balcony, Car Parking
    for col in ['Bathroom', 'Balcony', 'Car Parking']:
        if col in df_features.columns:
            col_clean = col.replace(' ', '_')
            df_features[col_clean] = pd.to_numeric(
                df_features[col].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
            feature_list.append(col_clean)
    
    # Floor
    if 'Floor' in df_features.columns:
        df_features['Floor_Number'] = pd.to_numeric(
            df_features['Floor'].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
        feature_list.append('Floor_Number')
    
    # 2. 類別型特徵編碼
    categorical_cols = ['location', 'Status', 'Transaction', 'Furnishing', 'facing', 'overlooking', 'Ownership']
    
    for col in categorical_cols:
        if col in df_features.columns:
            # 處理缺失值
            df_features[col] = df_features[col].fillna('Unknown')
            
            # 對於位置，保留出現次數較多的類別
            if col == 'location':
                location_counts = df_features[col].value_counts()
                top_locations = location_counts.head(20).index  # 保留前20個最常見位置
                df_features[col] = df_features[col].apply(
                    lambda x: x if x in top_locations else 'Other')
            
            # Label encoding
            le = LabelEncoder()
            df_features[col + '_Encoded'] = le.fit_transform(df_features[col])
            feature_list.append(col + '_Encoded')
    
    # 3. 創建新特徵
    # 面積比例
    if 'Carpet_Area' in df_features.columns and 'Super_Area' in df_features.columns:
        df_features['Area_Ratio'] = df_features['Carpet_Area'] / (df_features['Super_Area'] + 1e-6)
        feature_list.append('Area_Ratio')
    
    # 每平方米價格 (如果有面積資料)
    if 'Carpet_Area' in df_features.columns:
        df_features['Price_per_sqft'] = df_features['Price (in rupees)'] / (df_features['Carpet_Area'] + 1e-6)
    
    # 設施總數
    facility_cols = ['Bathroom', 'Balcony']
    valid_facility_cols = [col.replace(' ', '_') for col in facility_cols if col.replace(' ', '_') in df_features.columns]
    
    if valid_facility_cols:
        df_features['Total_Facilities'] = df_features[valid_facility_cols].sum(axis=1, skipna=True)
        feature_list.append('Total_Facilities')
    
    print(f"總共創建 {len(feature_list)} 個特徵")
    
    return df_features, feature_list

def enhanced_feature_selection(X, y, k=12):
    """
    增強版特徵選擇
    """
    print(f"\n=== 增強版特徵選擇 (選擇前 {k} 個特徵) ===")
    
    # 移除缺失值過多的特徵
    valid_features = []
    for col in X.columns:
        missing_ratio = X[col].isnull().sum() / len(X)
        if missing_ratio < 0.3:  # 保留缺失值少於30%的特徵
            valid_features.append(col)
    
    X_valid = X[valid_features]
    X_filled = X_valid.fillna(X_valid.median())
    
    print(f"有效特徵: {len(valid_features)} 個")
    
    # SelectKBest with F-statistics
    selector = SelectKBest(score_func=f_regression, k=min(k, len(valid_features)))
    selector.fit(X_filled, y)
    
    selected_features = X_filled.columns[selector.get_support()].tolist()
    feature_scores = selector.scores_[selector.get_support()]
    
    # 顯示特徵選擇結果
    feature_ranking = list(zip(selected_features, feature_scores))
    feature_ranking.sort(key=lambda x: x[1], reverse=True)
    
    print("選擇的特徵 (按重要性排序):")
    for i, (feature, score) in enumerate(feature_ranking):
        print(f"{i+1:2d}. {feature:20s}: {score:8.2f}")
    
    return selected_features, selector

def build_enhanced_model(X_train, X_test, y_train, y_test):
    """
    建立增強版模型 (包含正則化)
    """
    print("\n=== 建立增強版模型 ===")
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 嘗試不同的模型
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1)
    }
    
    best_model = None
    best_score = -np.inf
    best_name = ""
    
    results = {}
    
    for name, model in models.items():
        # 訓練模型
        model.fit(X_train_scaled, y_train)
        
        # 交叉驗證
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        # 預測
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # 評估
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'y_pred': y_test_pred
        }
        
        print(f"{name}:")
        print(f"  CV R² Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Test RMSE: {test_rmse:.2f}")
        
        if test_r2 > best_score:
            best_score = test_r2
            best_model = model
            best_name = name
    
    print(f"\n最佳模型: {best_name} (Test R²: {best_score:.4f})")
    
    return best_model, scaler, results, best_name

def create_comprehensive_plots(results, y_test, selected_features, best_name):
    """
    創建綜合性分析圖表
    """
    print("\n=== 創建綜合分析圖表 ===")
    
    # 設置圖表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    best_result = results[best_name]
    y_pred = best_result['y_pred']
    
    # 1. 實際值 vs 預測值
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, s=20)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 0].set_xlabel('實際房價')
    axes[0, 0].set_ylabel('預測房價')
    axes[0, 0].set_title(f'{best_name}: 實際值 vs 預測值\\nR² = {best_result["test_r2"]:.4f}')
    
    # 2. 殘差圖
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('預測房價')
    axes[0, 1].set_ylabel('殘差')
    axes[0, 1].set_title('殘差分析')
    
    # 3. 殘差分布
    axes[0, 2].hist(residuals, bins=50, alpha=0.7, density=True)
    axes[0, 2].set_xlabel('殘差')
    axes[0, 2].set_ylabel('密度')
    axes[0, 2].set_title('殘差分布')
    
    # 4. 模型比較
    model_names = list(results.keys())
    test_r2s = [results[name]['test_r2'] for name in model_names]
    
    bars = axes[1, 0].bar(model_names, test_r2s, color=['lightblue', 'lightgreen', 'lightcoral'])
    axes[1, 0].set_ylabel('Test R²')
    axes[1, 0].set_title('模型性能比較')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 在柱狀圖上標註數值
    for bar, r2 in zip(bars, test_r2s):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{r2:.3f}', ha='center', va='bottom')
    
    # 5. 特徵重要性 (使用最佳模型的係數)
    if hasattr(best_result['model'], 'coef_'):
        feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Coefficient': best_result['model'].coef_,
            'Abs_Coefficient': np.abs(best_result['model'].coef_)
        }).sort_values('Abs_Coefficient', ascending=True)
        
        # 取前10個最重要的特徵
        top_features = feature_importance.tail(10)
        
        y_pos = np.arange(len(top_features))
        axes[1, 1].barh(y_pos, top_features['Coefficient'])
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(top_features['Feature'])
        axes[1, 1].set_xlabel('回歸係數')
        axes[1, 1].set_title('Top 10 特徵重要性')
    
    # 6. 預測誤差分布
    percentage_error = np.abs((y_test - y_pred) / y_test) * 100
    percentage_error = percentage_error[percentage_error < 100]  # 移除極端值
    
    axes[1, 2].hist(percentage_error, bins=30, alpha=0.7)
    axes[1, 2].set_xlabel('絕對百分比誤差 (%)')
    axes[1, 2].set_ylabel('頻率')
    axes[1, 2].set_title(f'預測誤差分布\\n中位數誤差: {np.median(percentage_error):.1f}%')
    
    plt.tight_layout()
    plt.savefig('enhanced_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def log_transform_analysis(df_processed, selected_features):
    """
    對數轉換分析 (改善模型性能)
    """
    print("\n=== 對數轉換分析 ===")
    
    # 準備資料
    X = df_processed[selected_features].fillna(df_processed[selected_features].median())
    y = df_processed['Price (in rupees)']
    
    # 對目標變數進行對數轉換
    y_log = np.log1p(y)  # log(1+y) 避免 log(0)
    
    # 分割資料
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42)
    
    # 對應的原始目標變數
    y_train_orig = np.expm1(y_train_log)
    y_test_orig = np.expm1(y_test_log)
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 訓練模型 (使用對數轉換的目標變數)
    model_log = LinearRegression()
    model_log.fit(X_train_scaled, y_train_log)
    
    # 預測 (對數空間)
    y_train_pred_log = model_log.predict(X_train_scaled)
    y_test_pred_log = model_log.predict(X_test_scaled)
    
    # 轉換回原始空間
    y_train_pred_orig = np.expm1(y_train_pred_log)
    y_test_pred_orig = np.expm1(y_test_pred_log)
    
    # 評估 (原始空間)
    train_r2_log = r2_score(y_train_orig, y_train_pred_orig)
    test_r2_log = r2_score(y_test_orig, y_test_pred_orig)
    test_rmse_log = np.sqrt(mean_squared_error(y_test_orig, y_test_pred_orig))
    
    # 評估 (對數空間)
    train_r2_log_space = r2_score(y_train_log, y_train_pred_log)
    test_r2_log_space = r2_score(y_test_log, y_test_pred_log)
    
    print("對數轉換模型結果:")
    print(f"對數空間 - Train R²: {train_r2_log_space:.4f}, Test R²: {test_r2_log_space:.4f}")
    print(f"原始空間 - Train R²: {train_r2_log:.4f}, Test R²: {test_r2_log:.4f}")
    print(f"原始空間 - Test RMSE: {test_rmse_log:.2f}")
    
    return {
        'model': model_log,
        'scaler': scaler,
        'train_r2': train_r2_log,
        'test_r2': test_r2_log,
        'test_rmse': test_rmse_log,
        'y_test_orig': y_test_orig,
        'y_pred_orig': y_test_pred_orig
    }

def main_enhanced():
    """
    增強版主程式
    """
    print("房價預測多元線性回歸分析 - 增強版")
    print("=" * 50)
    
    # 載入資料
    df = pd.read_csv('dataset/house_prices.csv')
    print(f"原始資料: {df.shape}")
    
    # 增強版預處理
    df_clean = enhanced_data_preprocessing(df)
    
    # 進階特徵工程
    df_processed, feature_list = advanced_feature_engineering(df_clean)
    
    # 準備建模資料
    X = df_processed[feature_list]
    y = df_processed['Price (in rupees)']
    
    print(f"\n特徵矩陣: {X.shape}")
    print(f"目標變數: {y.shape}")
    
    # 增強版特徵選擇
    selected_features, selector = enhanced_feature_selection(X, y, k=12)
    
    # 準備最終資料
    X_selected = X[selected_features].fillna(X[selected_features].median())
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # 建立增強版模型
    best_model, scaler, results, best_name = build_enhanced_model(X_train, X_test, y_train, y_test)
    
    # 創建綜合分析圖表
    create_comprehensive_plots(results, y_test, selected_features, best_name)
    
    # 對數轉換分析
    log_results = log_transform_analysis(df_processed, selected_features)
    
    # 比較結果
    print("\n" + "=" * 50)
    print("最終結果比較:")
    print(f"最佳線性模型 ({best_name}): R² = {results[best_name]['test_r2']:.4f}")
    print(f"對數轉換模型: R² = {log_results['test_r2']:.4f}")
    
    # 儲存最佳模型
    if log_results['test_r2'] > results[best_name]['test_r2']:
        print("對數轉換模型表現較佳，儲存此模型")
        joblib.dump(log_results['model'], 'best_house_price_model.pkl')
        joblib.dump(log_results['scaler'], 'best_feature_scaler.pkl')
        final_r2 = log_results['test_r2']
    else:
        print(f"{best_name}表現較佳，儲存此模型")
        joblib.dump(best_model, 'best_house_price_model.pkl')
        joblib.dump(scaler, 'best_feature_scaler.pkl')
        final_r2 = results[best_name]['test_r2']
    
    joblib.dump(selector, 'best_feature_selector.pkl')
    
    # 儲存改進版結果
    with open('enhanced_model_results.txt', 'w', encoding='utf-8') as f:
        f.write("房價預測多元線性回歸分析 - 增強版結果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"最佳模型 R²: {final_r2:.4f}\n")
        f.write(f"資料量: {len(df_clean)} 筆\n")
        f.write(f"特徵數: {len(selected_features)} 個\n")
        f.write("\n模型比較:\n")
        for name, result in results.items():
            f.write(f"{name}: R² = {result['test_r2']:.4f}\n")
        f.write(f"對數轉換模型: R² = {log_results['test_r2']:.4f}\n")
    
    print(f"\n分析完成！最終 R² = {final_r2:.4f}")

if __name__ == "__main__":
    main_enhanced()