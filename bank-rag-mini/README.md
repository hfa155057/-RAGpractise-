# 銀行內部知識查詢助理 Mini RAG

## 專案簡介

本專案是一個金融文件查詢助理，使用 RAG（Retrieval-Augmented Generation）架構，讓系統先從金融 FAQ 與開戶 SOP 文件中檢索相關段落，再將檢索結果提供給 LLM 產生回答。

此專案主要目標是模擬金融業內部知識查詢情境，展示文件讀取、文字切分、相似度檢索、LLM 回答、來源引用與資料不足拒答等 RAG 核心流程。

\---

## 專案背景

金融業內部文件通常包含 FAQ、SOP、產品條款、法遵規範與客服流程。傳統人工查詢文件較耗時，若直接使用生成式 AI 回答，又可能產生無依據內容或幻覺。

因此，本專案採用 RAG 架構，讓系統在回答前先從知識庫中找出相關文件段落，再根據參考資料整理回答，並附上來源，以提升回答的可靠性與可追溯性。

\---

## 專案目標

* 建立金融文件知識庫
* 讓使用者以自然語言查詢文件內容
* 使用 TF-IDF 與 Cosine Similarity 找出相關段落
* 將檢索結果提供給 LLM 整理回答
* 回答時附上來源文件、段落與相似度分數
* 當知識庫資料不足時拒答，降低 AI 幻覺風險

\---

## 核心功能

|功能|說明|
|-|-|
|文件讀取|讀取 `data/sample\\\_docs/` 中的 txt 文件|
|Chunk 切分|將長文件依段落切成較小的知識片段|
|TF-IDF 向量化|將文字轉換成可計算的數值特徵|
|相似度檢索|使用 Cosine Similarity 找出與問題最相關的段落|
|LLM 回答|將檢索到的段落提供給 LLM 整理回答|
|來源引用|顯示答案依據的文件名稱、段落編號與相似度分數|
|資料不足拒答|當檢索相似度過低時，系統會回覆資料不足|
|Streamlit 介面|提供互動式網頁 Demo，方便展示與操作|

\---

## 使用技術

* Python
* Streamlit
* scikit-learn
* TF-IDF
* Cosine Similarity
* OpenAI API
* python-dotenv

\---

## 系統流程

```text
金融文件
↓
讀取 txt 文件
↓
切分為 chunks
↓
使用者輸入問題
↓
TF-IDF 向量化
↓
Cosine Similarity 檢索 Top-k 相關段落
↓
將相關段落組成參考資料
↓
LLM 根據參考資料整理回答
↓
顯示回答與參考來源
```

\---

## 專案結構

```text
bank-rag-mini/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
└── data/
    └── sample\\\_docs/
        ├── bank\\\_faq.txt
        └── account\\\_opening\\\_sop.txt
```

\---

## 檔案說明

|檔案 / 資料夾|說明|
|-|-|
|`app.py`|Streamlit 主程式，包含文件讀取、檢索、LLM 回答與畫面顯示|
|`requirements.txt`|專案所需 Python 套件|
|`README.md`|專案說明文件|
|`.env.example`|API Key 設定範例|
|`.gitignore`|設定不需上傳 GitHub 的檔案|
|`data/sample\\\_docs/`|放置金融範例文件|

\---

## 安裝方式

### 1\. 下載專案

```bash
 bank-rag-mini
```

\---

### 2\. 建立虛擬環境

Windows：

```bash
python -m venv .venv
```

```bash
.venv\\\\\\\\Scripts\\\\\\\\activate
```

macOS / Linux：

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

\---

### 3\. 安裝套件

```bash
pip install -r requirements.txt
```

\---

## 環境變數設定

請在專案根目錄建立 `.env` 檔案。

內容範例：

```env
OPENAI\\\_API\\\_KEY=your\\\_api\\\_key\\\_here
OPENAI\\\_MODEL=your\\\_model\\\_here
```

\---

## 執行方式

在專案根目錄執行：

```bash
streamlit run app.py
```

成功後，瀏覽器會開啟 Streamlit 頁面，通常網址為：

```text
http://localhost:8501
```

\---

## Demo 測試問題

### 問題 1：開戶文件檢核

```text
開戶時客戶沒有提供身分證明文件，客服應該怎麼處理？
```

預期效果：  
系統應檢索到開戶 SOP 中與身分證明文件相關的段落，並回答客戶需補齊文件後才能重新辦理。

\---

### 問題 2：信用卡掛失

```text
信用卡遺失時要怎麼處理？
```

預期效果：  
系統應檢索到信用卡 FAQ 中與掛失、身分確認及補發新卡相關的段落。

\---

### 問題 3：資料不足拒答

```text
這間銀行明年會不會調高房貸利率？
```

預期效果：  
由於目前知識庫沒有房貸利率預測資料，系統應回覆：

```text
目前知識庫中沒有足夠資料可以可靠回答這個問題。
```

\---

## RAG 核心邏輯說明

本專案的 RAG 流程分為兩個階段：

### 1\. Retrieval：檢索階段

系統會先讀取金融文件，將文件切分為多個 chunks。  
當使用者輸入問題後，系統會使用 TF-IDF 將問題與文件段落轉換成向量，並透過 Cosine Similarity 計算相似度，找出最相關的 Top-k 段落。

### 2\. Generation：生成階段

系統會將檢索到的相關段落組成參考資料，交給 LLM 整理回答。  
LLM 被要求只能根據參考資料回答，若參考資料不足，必須回覆資料不足，避免產生無依據內容。

\---

## 風險控管設計

本專案特別加入以下設計，以模擬金融業導入生成式 AI 時需要注意的風險控管。

### 1\. 回答附來源

每次回答後，系統會顯示：

* 文件名稱
* Chunk 編號
* 相似度分數
* 原始段落內容

這可以讓使用者確認答案依據，提升可追溯性。

### 2\. 資料不足拒答

當最高相似度分數低於門檻時，系統不會強行回答，而是回覆目前知識庫資料不足。

### 3\. 限制 LLM 只能根據參考資料回答

Prompt 中明確要求模型不得自行編造金融規定、利率、費用、流程或不存在的文件內容。

\---

## 專案亮點

* 使用金融 FAQ 與開戶 SOP 作為應用情境，貼近金融業內部知識查詢需求
* 完整展示 RAG 的核心流程：文件讀取、chunk 切分、檢索、生成與來源引用
* 使用 TF-IDF 與 Cosine Similarity 建立最小可行版本，方便理解 RAG 基礎原理
* 加入資料不足拒答機制，降低 AI 幻覺風險
* 使用 Streamlit 建立互動式 Demo，方便作品展示與面試說明

\---

## 目前限制

* 目前僅支援 txt 文件
* 目前檢索方式為 TF-IDF，尚未使用正式 embedding 模型
* 尚未加入 Chroma 或 FAISS 向量資料庫
* 尚未支援 PDF / Word 文件上傳
* 尚未加入使用者登入、權限控管與查詢紀錄
* 尚未加入 Prompt Injection 測試頁面

\---

## 未來優化方向

* 支援 PDF / Word / txt 文件上傳
* 使用 Embedding 模型取代 TF-IDF
* 加入 Chroma 或 FAISS 向量資料庫
* 串接更完整的 LLM 回答流程
* 加入查詢紀錄與 SQLite 資料庫
* 加入使用者回饋按鈕
* 加入 Prompt Injection 測試
* 加入回答品質評估機制
* 部署至雲端平台，提供線上 Demo

\---

## 履歷描述範例

銀行內部知識查詢助理 Mini RAG｜Python、Streamlit、scikit-learn、OpenAI API

* 建立金融文件問答系統，模擬銀行 FAQ 與開戶 SOP 查詢情境
* 使用 TF-IDF 與 Cosine Similarity 檢索相關文件段落
* 將檢索結果傳入 LLM，要求模型根據來源整理回答
* 設計來源引用與資料不足拒答機制，降低 AI 幻覺風險
* 使用 Streamlit 建立互動式 Demo，方便展示 RAG 查詢流程

\---

## 專案定位

本專案為 Mini RAG Demo，重點在於展示 RAG 的核心概念與金融應用情境。  
後續可擴充為支援多格式文件、向量資料庫、查詢紀錄、使用者回饋與資安測試的金融內部知識管理系統。

