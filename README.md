# Facial Exercises Feedback System｜臉部運動即時回饋系統

結合 YOLO 輕量化表情偵測模型與 MediaPipe 臉部特徵點，即時引導使用者進行臉部訓練，並自動記錄動作幅度以評估短期間的訓練成效。

- 表情辨識類別：`angry`、`happy`、`neutral`、`sad`、`surprised`
- 4 種訓練動作：臉部表情訓練、臉部拉提訓練、嘴部開合訓練、皺眉緊眉訓練
- 推論方式：YOLOv7-tiny / YOLOv9-tiny 訓練後匯出為 ONNX，搭配 MediaPipe Face Mesh 做特徵點追蹤

## 系統架構（三階段）

```
1. 資料建置 (dataset_pipeline/)
      蔡旻均(2020)原始資料集 → 分層抽樣 train/val → 少數類別擴增 → 加入外部圖庫測試集
                        ↓
2. YOLO 模型訓練 (model_training/)
      YOLOv7-tiny / YOLOv9-tiny 訓練 → 比較各 batch size mAP → 選最佳權重 → 匯出 CPU ONNX
                        ↓
3. 臉部運動系統 (app/)
      將 ONNX 權重放入 app/weights/ → 即時偵測 + 引導使用者完成面部運動
```

## 資料夾結構

```
Facial-Exercises-Detection-System/
├── dataset_pipeline/                      # 1. 資料建置
│   ├── 01_train_val_split/
│   │   └── train_val_split.py             # 對 trainval pool 做分層抽樣切 train/val
│   ├── 02_balance_augmentation/
│   │   └── make_5_balanced_data.py        # 對 train set 做少數類別擴增（水平翻轉/明暗對比/模糊）
│   ├── 03_test_100_annotation_reliability/ # 外部圖庫測試集（test_100）的標註品質檢核
│   │   ├── prepare_blind_annotation.py     # 產生匿名盲標註樣本
│   │   ├── annotate_gui.py                 # 第二位標註者標註介面
│   │   ├── analyze_agreement.py            # 計算 Cohen's kappa / IoU 一致性
│   │   ├── agreement_summary.csv
│   │   └── test_100_sources.csv            # test_100 每張圖對應的來源圖庫平台
│   └── 04_test_train_dedup_check/          # 測試集(test_300)與訓練集(train)重複影像系統化查核
│       ├── check_test_train_duplicates.py  # MD5 精確比對 + pHash(256-bit) 近似比對
│       └── duplicate_report.csv            # 執行輸出（本次結果：0 組重複）
├── model_training/                        # 2. YOLO 模型訓練
│   ├── yolov7/
│   │   ├── face_dataset*.yaml             # 資料集設定（train/val/test 路徑、類別）
│   │   ├── make_5_balanced_data.py
│   │   ├── run.py                          # PyTorch 版即時系統（需搭配完整 YOLOv7 原始碼）
│   │   ├── commands.txt                    # 訓練/驗證/測試/匯出 ONNX 指令與各 batch size mAP 結果
│   │   └── runs/                           # 訓練/驗證/測試紀錄（圖表、log；權重檔另見雲端連結）
│   └── yolov9/
│       ├── face_dataset*.yaml
│       ├── commands.txt
│       └── runs/
├── app/                                    # 3. 臉部運動系統
│   ├── Facial_Exercises_Training.py         # 系統本體：即時引導使用者做面部運動
│   ├── Facial_Exercises_Evaluation.py       # 研究評估流程：受試者前測 → 訓練 → 後測
│   └── weights/
│       ├── v7best.onnx                     # YOLOv7-tiny 匯出.onnx權重
│       └── v9best.onnx                     # YOLOv9-tiny 匯出.onnx權重
├── statistical_analysis/                   # 4. 統計分析（論文用）
│   ├── 3.3_雙樣本比例Z檢定.py
│   ├── 4.6.1_*.py / 4.6.2_*.py / 4.6.3_*.py / 4.6.3附圖_*.py
│   ├── 總統計表(去識別化).xlsx              # 已移除真名對照表，僅留 P01~P15 代號
│   ├── fig_table24_gap.png
│   └── yolo原始資料/                        # YOLO 測試集標籤與彙總指標圖（confusion matrix / F1 / PR 曲線）
└── requirements.txt
```

> `evaluation_data/`（受試者前後測實測紀錄）與資料集實際影像/標註檔屬研究參與者個資與大型二進位檔案，不納入版本控制（見 `.gitignore`），僅保留在本機。`statistical_analysis/原始資料/`（受試者真名資料夾與訓練影片）與含真名對照表的檔案同樣不納入版本控制，只保留已去識別化的彙總結果。

## 安裝步驟

本專案三個部分請**各自建立獨立虛擬環境**，不要共用，避免套件版本互相衝突：

| 部分 | Python 版本 | 說明 |
|---|---|---|
| `app/`（臉部運動系統） | **3.9** | 開發/測試時實際使用的版本 |
| `model_training/yolov7/` | **3.12** | YOLOv7 官方 repo 所需套件版本 |
| `model_training/yolov9/` | **3.12** | YOLOv9 官方 repo 所需套件版本 |

**臉部運動系統（`app/`）**：

```bash
py -3.9 -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

> `requirements.txt` 僅涵蓋**即時偵測 App**（`app/`）所需套件（OpenCV、MediaPipe、ONNX Runtime 等）。

**YOLO 模型訓練（`model_training/`）**：YOLOv7、YOLOv9 官方 repo 各自要求的套件版本（PyTorch 等）不同，即使都用 Python 3.12，也請**各建一個獨立虛擬環境**，不要共用同一個環境訓練兩個版本，以免套件衝突。建立方式見下方「2. YOLO 模型訓練」。

## 1. 資料建置

資料集以 **蔡旻均 (2020)** 蒐集的表情資料集（[原始資料集雲端連結](https://drive.google.com/drive/folders/1kcvTD2jTDpjJ_9v79TEqog7bfczA0FpH)）為基礎（5 類別：angry / happy / neutral / sad / surprised），再依以下流程重新切分、平衡與擴充：

| 步驟 | 輸入 | 處理方式 | 輸出 |
|---|---|---|---|
| 0. 原始資料 | 蔡旻均(2020) `new_glasses_splidata`（1913 張，YOLO 格式標註） | 依原作者提供的 `train.txt`（1713 張）/ `test.txt`（200 張）切分 | trainval pool 1713 張、test 200 張 |
| 1. train/val 分層抽樣 | trainval pool 1713 張 | `dataset_pipeline/01_train_val_split/train_val_split.py`：依類別比例分層抽樣（`stratify`），val 比例 15%，`random_state=42` 可重現 | train 1456 張、val 257 張 |
| 2. 少數類別擴增 | train 1456 張 | `dataset_pipeline/02_balance_augmentation/make_5_balanced_data.py`：對未達 400 張的類別做水平翻轉、明暗對比調整、高斯模糊擴增（**僅對 train 做擴增，不涉及 test**） | 平衡後 train 2172 張 |
| 3. 加入外部圖庫測試集 | test 200 張（步驟 0 由 `test.txt` 切分而來，未經步驟 1/2 處理） | 另補 100 張取自 **Pexels、iStock、Pixabay、Dreamstime、Freepik** 五大圖庫的表情圖片，並以 `03_test_100_annotation_reliability/` 進行雙人盲標註一致性檢驗（Cohen's kappa、框選 IoU）確保標註品質 | test 200 + 100 = 300 張 |
| 4. 測試/訓練集重複影像查核 | test_300（300 張）× train（2172 張） | `dataset_pipeline/04_test_train_dedup_check/check_test_train_duplicates.py`：MD5 精確比對（位元組級） + pHash 256-bit 近似比對（漢明距離 ≤10%），確保測試集未與訓練集重複 | 0 組 MD5 完全相同、0 組 pHash 疑似重複；雜湊距離最接近之配對仍達 60/256 bits（約 23.4% 差異），經肉眼複查確認為不同真人 |

少數類別擴增（步驟 2）的中繼產物為 `v4_data_balanced/`（僅 train/val，尚未加入外部測試集）；加入外部圖庫測試集（步驟 3）後即為最終資料集 `v4_data_final/`（未納入本 repo，下載連結見下方「資料集下載」）：

```
v4_data_balanced/                             # 步驟 2 輸出（中繼產物）
├── images/{train, val}
└── labels/{train, val}

v4_data_final/                                # 步驟 3 輸出（最終，YOLO 訓練直接使用）
├── images/{train, val, test_100, test_200, test_300}
└── labels/{train, val, test_100, test_200, test_300}
```

- `train`：2172 張（平衡後）
- `val`：257 張
- `test_200`：原始 200 張測試集
- `test_100`：外部圖庫 100 張，來源平台對照見 `dataset_pipeline/03_test_100_annotation_reliability/test_100_sources.csv`
- `test_300`：`test_200` + `test_100` 合併，最終送入 YOLO 訓練/測試的完整測試集

`test_100` 各圖庫平台的表情類別張數（5 類共 100 張，每類 20 張）：

| 平台 | angry | happy | neutral | sad | surprised | 小計 |
|---|---|---|---|---|---|---|
| Pexels | 7 | 20 | 20 | 1 | 20 | 68 |
| iStock | 3 | 0 | 0 | 7 | 0 | 10 |
| Pixabay | 8 | 0 | 0 | 1 | 0 | 9 |
| Dreamstime | 2 | 0 | 0 | 5 | 0 | 7 |
| Freepik | 0 | 0 | 0 | 6 | 0 | 6 |
| **合計** | **20** | **20** | **20** | **20** | **20** | **100** |

`happy`、`neutral`、`surprised` 全數取自 Pexels；`angry`、`sad` 因 Pexels 該類別素材不足，另從 iStock、Pixabay、Dreamstime、Freepik 補足。

> 每張圖片對應的圖庫平台以檔名前綴判斷（例：`Freepik_sad1.jpg` → Freepik → sad），完整逐張對照（含表情類別）見 `test_100_sources.csv`。

`dataset_pipeline/04_test_train_dedup_check/check_test_train_duplicates.py` 需額外安裝 `Pillow`、`imagehash`（`pip install pillow imagehash`），對 `test_300` 與 `train` 進行 MD5 精確比對與 pHash（256-bit）近似比對，確認測試集未與訓練集重複；完整輸出見同目錄 `duplicate_report.csv`。

### 資料集下載

實際影像因檔案體積（原始資料 > 10 GB）與授權/個資考量（`蔡旻均(2020)` 原始資料集之受試者影像、`test_100` 之圖庫版權），未直接納入本 git repo，改以雲端硬碟連結提供各階段輸出（對應上表步驟）：

| 階段 | 檔案 | 下載連結 |
|---|---|---|
| 0. 原始資料 | `new_glasses_splidata_raw_data.zip` | [Google Drive](https://drive.google.com/file/d/1k-_NVOv6kA1L-wLmZwjg8FAxws8xnQK9/view?usp=sharing) |
| 0. train/test 切分 | `v4_data_train_test_split.zip` | [Google Drive](https://drive.google.com/file/d/1alhxDbqimRergwAwPtfl0uPbL86rUtlZ/view?usp=sharing) |
| 1. train/val 分層抽樣 | `v4_data_train_val_split.zip` | [Google Drive](https://drive.google.com/file/d/1bgTuFju1fe4iR-mlBIWud8Nwz5_m25Qb/view?usp=sharing) |
| 2. 少數類別擴增 | `v4_data_balanced.zip` | [Google Drive](https://drive.google.com/file/d/1193CTIyYW23W7lpTdWdnbbmGPGYgAqCg/view?usp=sharing) |
| 3. 加入外部圖庫測試集（最終訓練用） | `v4_data_final.zip` | [Google Drive](https://drive.google.com/file/d/1jjJXyhy8OAhbLLLo-ey4YrwNsiWZLfOa/view?usp=sharing) |

若只是要重新訓練 YOLO 模型，下載 `v4_data_final.zip` 即可（對應「2. YOLO 模型訓練」章節所需的 `v4_data_final/`）；若要重現完整資料建置流程，可從 `new_glasses_splidata_raw_data.zip` 開始依序執行。

如需取得資料集，請先確認用途符合研究倫理與圖庫授權範圍，即可直接點擊上方連結下載。

## 2. YOLO 模型訓練

偵測模型基於官方 [YOLOv7](https://github.com/WongKinYiu/yolov7) 與 [YOLOv9](https://github.com/WongKinYiu/yolov9) 微調，`model_training/` 內只保留**客製化的設定檔與腳本**，訓練前需先取得官方原始碼。YOLOv7、YOLOv9 官方 repo 要求的套件版本不同，**請分別建立獨立虛擬環境**，不要共用：

```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
py -3.12 -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
# 將 model_training/yolov7/ 內的檔案複製到 yolov7/ 目錄下
# 並將「1. 資料建置 → 資料集下載」的 v4_data_final.zip 解壓後放入 yolov7/ 目錄下
```

`yolov9` 作法相同，改用 [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)，並在另一個獨立目錄／虛擬環境下進行，避免跟 yolov7 的環境互相污染。

訓練 / 驗證 / 測試 / 匯出 ONNX 的完整指令與各 batch size（b16 / b32 / b64、250 epochs）之 mAP 結果，完整記錄在：
- `model_training/yolov7/commands.txt`
- `model_training/yolov9/commands.txt`

流程為：分別訓練 b16/b32/b64 → 在 val set 上比較 mAP@.5 選出最佳 batch size 權重 → 於 `test_100`/`test_200`/`test_300` 上測試泛化能力 → 將最佳權重匯出為 CPU 可用的 ONNX。

範例（YOLOv7-tiny，250 epochs）：

```bash
python train.py --workers 10 --device 0 --batch-size 32 \
    --data face_dataset.yaml --img 224 \
    --cfg cfg/training/yolov7-tiny.yaml --weights yolov7-tiny.pt \
    --name v4_data_e250_b32 --hyp data/hyp.scratch.tiny.yaml --epochs 250

python export.py --weights runs/train/v4_data_e250_b32/weights/best.pt \
    --img-size 224 224 --batch-size 1 --device cpu --grid --simplify
```

YOLOv9 對應指令（`train_dual.py` / `val_dual.py` / `export.py --include onnx`）見 `model_training/yolov9/commands.txt`。

### 訓練結果（`runs/`）

`model_training/yolov7/runs/`、`model_training/yolov9/runs/` 保留完整訓練/驗證/測試紀錄（confusion matrix、PR/F1/P/R 曲線、train/test batch 視覺化、TensorBoard log、`opt.yaml`/`hyp.yaml`、逐 epoch 結果表），方便直接比對三種 batch size 的訓練過程，不需重新跑訓練：

```
runs/
├── train/{b16_e250, b32_e250, b64_e250}/   # 各 batch size 訓練過程
├── val/{...}/                              # 個別驗證結果
└── test/{pooled_test100, ...200, ...300}/  # 在 test_100/200/300 上的最終測試結果
```

逐 epoch 的權重檔（`weights/*.pt`，YOLOv7 每組 19～25 個、YOLOv9 每組 10～12 個，總計約 3.4 GB）因體積過大未納入 git，改以雲端硬碟提供 `runs/` 完整備份（含上述所有圖表與 log，供對照）：

| 版本 | 檔案 | 下載連結 |
|---|---|---|
| YOLOv7（b16/b32/b64 全部 checkpoint） | `yolov7_runs.zip` | [Google Drive](https://drive.google.com/file/d/1_6u0pJvDNm35GxtI0uzMtHPfkZfQLkHE/view?usp=sharing) |
| YOLOv9（b16/b32/b64 全部 checkpoint） | `yolov9_runs.zip` | [Google Drive](https://drive.google.com/file/d/1vDYreE2ZvngDAJwOvjyi0Q3BRrf6FIB4/view?usp=sharing) |

> `train_batch*.jpg` / `test_batch*.jpg` 為訓練/測試批次視覺化圖，內含資料集人臉圖片（蔡旻均(2020)資料集或外部圖庫，非論文受試者真實個資）。

匯出的最佳權重（本 repo 已附上 `v7best.onnx`、`v9best.onnx`）放入 `app/weights/` 即可供第 3 階段的臉部運動系統使用。

## 3. 臉部運動系統

App 使用已訓練好的 ONNX 權重（`app/weights/`），不需 GPU，CPU 即可即時推論，搭配 MediaPipe 做臉部特徵點追蹤以量測動作幅度。

**`Facial_Exercises_Training.py`：系統本體**，即時引導使用者做面部運動並記錄幅度：

```bash
cd app
python Facial_Exercises_Training.py ^
    --weights weights/v7best.onnx ^
    --source 0 ^
    --participant 01 ^
    --session training ^
    --reps 10
```

程式啟動後會先進入**主選單**，訓練動作是在畫面上按數字鍵即時選擇，而非用啟動參數指定：

| 按鍵 | 訓練動作 | 判定方式 |
|---|---|---|
| `1` | 臉部表情訓練 | 五類表情循環訓練：生氣、開心、自然、難過與驚訝 |
| `2` | 臉部拉提訓練 | 微笑上提訓練：結合表情辨識與嘴角位置變化判斷 |
| `3` | 嘴部開合訓練 | 張嘴訓練：以上下嘴唇距離與表情狀態進行即時回饋 |
| `4` | 皺眉緊眉訓練 | 皺眉訓練：以眉頭水平靠攏與垂直下壓幅度搭配表情狀態判斷 |

主要啟動參數：

| 參數 | 說明 |
|---|---|
| `--weights` | ONNX 權重路徑，預設 `v7best.onnx` |
| `--source` | 攝影機編號或影片路徑，預設 `0` |
| `--session` | `pre_test`（前測）/ `training`（訓練）/ `post_test`（後測） |
| `--participant` | 受試者編號，用於記錄檔命名 |
| `--reps` | 每項動作回合數（第 2/3/4 項），前後測建議設 3 |

**`Facial_Exercises_Evaluation.py`：研究評估流程**，用於正式找受試者進行「前測 → 訓練 → 後測」自動化評估：

```bash
cd app
python Facial_Exercises_Evaluation.py --weights weights/v9best.onnx --source 0
```

執行紀錄會輸出至 `evaluation_data/`（校準基準值、動作幅度時序資料，屬受試者個資，未納入本 repo）。

## 4. 統計分析

`statistical_analysis/` 是論文統計分析用的程式與彙總結果：

| 檔案 | 內容 |
|---|---|
| `3.3_雙樣本比例Z檢定.py` | YOLOv7-tiny / YOLOv9-tiny 於不同測試子集之 Recall/Precision 雙樣本比例 Z 檢定 |
| `4.6.1_*.py` | Score 與動作幅度關係（Pearson / Spearman / rmcorr） |
| `4.6.2_*.py` | 前後測改善與逐回合趨勢（成對 t 檢定） |
| `4.6.3_*.py`、`4.6.3附圖_*.py` | 個人化校準必要性（CV、Wilson CI）與固定門檻誤判示意圖 |
| `總統計表(去識別化).xlsx` | 受試者總表、逐回合幅度、Score 明細、校準基準彙總，皆以 `P01`~`P15` 代號呈現 |
| `yolo原始資料/` | YOLOv7/v9 測試集標籤與彙總指標圖（confusion matrix、F1/PR/P/R 曲線） |

> ⚠️ 受試者真實姓名資料夾（含訓練影片 `.mp4`）與真名對照表**不納入本 repo**，僅保留在本機（見 `.gitignore`）。上面附的 `總統計表(去識別化).xlsx` 已移除「匿名對照」工作表，其餘工作表原本就是用 `P01`~`P15` 代號記錄，沒有真名。

## 授權與引用

本專案採用 **GPL-3.0** 授權（見 [LICENSE](LICENSE)），與 `model_training/` 所依賴的官方 [YOLOv7](https://github.com/WongKinYiu/yolov7) / [YOLOv9](https://github.com/WongKinYiu/yolov9)（皆為 GPL-3.0）保持一致。任何人使用、修改或發布本專案之衍生作品，皆須以 GPL-3.0 公開原始碼。

本專案資料集基礎為前人 **蔡旻均 (2020)** 蒐集之表情資料集（[原始資料集](https://drive.google.com/drive/folders/1kcvTD2jTDpjJ_9v79TEqog7bfczA0FpH)），使用時請一併標註原始資料集出處。

<!-- TODO: 補上本論文正式的引用格式（作者、標題、年份） -->
