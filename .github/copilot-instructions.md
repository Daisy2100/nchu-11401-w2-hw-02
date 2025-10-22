一、作業目標：
本次作業延伸自 HW1，目標是讓同學能夠以實際資料集進行「多元線性回歸 (Multiple Linear Regression)」的完整分析，並遵循 CRISP-DM 流程完成從資料理解、建模到評估的全過程。
 
二、作業內容：
1. 資料來源
  至 Kaggle 選擇一個具有 10 至 20 個特徵 (features) 的公開資料集。
  類型不限（可為房價預測、醫療、車輛效能等主題）。
  請明確標示資料集來源與連結。
2. 分析任務
  使用線性回歸 (Linear Regression) 模型進行預測。
  可嘗試單純線性回歸、多元線性回歸或 Auto Regression。
  必須執行 特徵選擇 (Feature Selection) 與 模型評估 (Model Evaluation)。
  結果部分需包含請提供預測圖(加上信賴區間或預測區間)
3. CRISP-DM 流程說明
    Business Understanding
    Data Understanding
    Data Preparation
    Modeling
    Evaluation
    Deployment

4. AI協助要求
  所有與 ChatGPT 的對話請以 pdfCrowd 或其他方式須匯出為 PDF
  請使用 NotebookLM 對網路上同主題的解法進行研究，並撰寫一份 100 字以上的摘要，放入報告中。
  請在報告中明確標示「GPT 輔助內容」與「NotebookLM 摘要」
5. 繳交內容
  主程式：7114056XXX_hw2.py/.ipynb
  報告檔： PDF，需包含以下內容：
      按照 CRISP-DM 說明的分析流程
      GPT 對話過程（pdfCrowd 匯出）
      NotebookLM 研究摘要
      網路上主流或更優解法之比較與說明
  以上檔案與資料夾請壓縮為學號命名的一個zip（例如 7114056XXX_hw2.zip）上傳。

  (optional) 若上傳至 GitHub，或是以colab撰寫，需提供連結，並在 README.md 中整理流程與成果。

三、評分標準
  文件說明（50%）
    CRISP-DM 流程完整且邏輯清楚（25%）
    包含 GPT 對話與 NotebookLM 摘要（15%）
    有明確說明資料集來源與研究脈絡（10%）
  結果呈現（50%）
    模型正確可執行，具特徵選擇與評估（25%）
    結果合理、美觀且具有說服力（15%）
    呈現出Kaggle名次(若有)/預測結果評估(預測圖、評估指標)（10%）