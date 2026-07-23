# Facial Exercises Feedback System｜臉部運動即時回饋系統

*繁體中文 | [English](README.en.md)*

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
| `app/`（臉部運動系統） | **3.11** | 開發/測試時實際使用的版本 |
| `model_training/yolov7/` | **3.12** | YOLOv7 官方 repo 所需套件版本 |
| `model_training/yolov9/` | **3.12** | YOLOv9 官方 repo 所需套件版本 |

**臉部運動系統（`app/`）**：

```bash
py -3.11 -m venv venv
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

### 3.4 兩模型比較與最終選擇

根據上述訓練結果，本研究進一步比較 YOLOv7-tiny 與 YOLOv9-tiny 於獨立測試集之綜合辨識效能，以及兩模型於相同條件下之端到端效能表現，以決定最終整合至臉部運動即時回饋系統之模型。

**（一）辨識效能比較**

表 19、YOLOv7-tiny 與 YOLOv9-tiny 最佳模型測試集綜合比較

| 指標 | YOLOv7-tiny b32 e244 | YOLOv9-tiny b32 e225 |
|---|---|---|
| Precision | 0.862 | 0.859 |
| Recall | 0.860 | 0.886 |
| F1-score | 0.861 | 0.873 |
| mAP@0.5 | 0.865 | 0.896 |
| mAP@0.5:0.95 | 0.540 | 0.561 |
| 推論速度(FPS) | 159.2 | 70.4 |

就辨識效能而言，YOLOv9-tiny 在整體 F1-score（0.873 對比 0.861）與 mAP@0.5（0.896 對比 0.865）上均略優於 YOLOv7-tiny，尤其在 angry 等辨識難度較高的類別上，YOLOv9-tiny 之 Recall 表現更為突出；此差異可能與 PGI 機制、GELAN 架構等整體設計有關，惟本研究未進行 ablation study 拆解各機制之貢獻，無法將優勢單獨歸因於 PGI 機制。就推論速度而言，此階段所量測之 FPS 係以 PyTorch 驗證框架對測試集進行批次前向傳播計時所得，YOLOv7-tiny（159.2）與 YOLOv9-tiny（70.4）相差約 2.26 倍；惟此數值反映的是「模型匯出前」之效能基準，與系統實際部署所採用之 ONNX Runtime 單幀推論效能不完全相同，兩模型於實際部署情境下之表現，須以下述端到端測試結果為準。

**（二）相同條件下之端到端效能比較**

為避免僅以單一模型（YOLOv7-tiny）之端到端測試結果推論另一模型（YOLOv9-tiny）之部署可行性，本研究以延遲（ms）為量測單位，逐幀記錄原始樣本後計算平均延遲、標準差與 P95 延遲，如表 20 所示。

表 20、YOLOv7-tiny 與 YOLOv9-tiny 相同條件下端到端處理延遲比較表

| 量測項目 | 模型 | 平均延遲(ms) | 延遲SD(ms) | P95延遲(ms) | CPU平均(%) | CPU峰值(%) | 記憶體峰值(MB) |
|---|---|---|---|---|---|---|---|
| ONNX CPU | v7 | 4.85 | 0.27 | 5.38 | 69.9 | 85.1 | 569.5 |
| ONNX CPU | v9 | 7.38 | 0.41 | 8.14 | 67.3 | 82.6 | 566.9 |
| 臉部運動系統（YOLO+MediaPipe） | v7 | 10.16 | 0.99 | 11.72 | 38.0 | 84.7 | 1387.6 |
| 臉部運動系統（YOLO+MediaPipe） | v9 | 12.74 | 1.76 | 15.55 | 39.8 | 83.1 | 1387.9 |

> 測試硬體：12th Gen Intel(R) Core(TM) i5-12600KF，Windows 11；兩模型於同一次執行、同一攝影機來源、相同 onnxruntime 預設執行緒配置下依序測試，確保比較公平性。

由表 20 可知，YOLOv7-tiny 之 ONNX 純推論平均延遲為 4.85ms（P95 = 5.38ms），加入 MediaPipe FaceMesh 與前處理後之處理管線平均延遲上升至 10.16ms（P95 = 11.72ms）；YOLOv9-tiny 則由 7.38ms 上升至 12.74ms。兩模型於處理管線階段之延遲差距與純推論階段方向一致，顯示表 19 所觀察到之推論速度差距，於加入 MediaPipe 幾何特徵分析後依然存在，並非匯出評估階段之人工現象。惟需說明，上述處理延遲不含攝影機擷取等待與 GUI 繪製，系統實際顯示吞吐量另見（三）之討論（表 21）。

**（三）實際顯示吞吐量（含GUI）**

表 20 之處理延遲量測排除攝影機擷取等待與 GUI 繪製，以獨立評估模型本身之運算成本；然而使用者實際操作系統時所感受之流暢度，取決於「包含 GUI 繪製之完整系統」實際能達成之顯示更新速度。本研究進一步直接執行正式系統程式（而非獨立測速腳本），於背景自動完成校準並持續訓練，逐秒記錄實際顯示吞吐量，結果如表 21 所示。

表 21、系統實際顯示吞吐量比較表（攝影機擷取+YOLO推論+MediaPipe特徵分析+GUI）

| 量測項目 | 模型 | 平均FPS | SD(FPS) | P5(FPS) |
|---|---|---|---|---|
| i5-12600KF | v7 | 24.17 | 0.95 | 22.83 |
| i5-12600KF | v9 | 22.61 | 0.87 | 21.73 |
| i7-8565U（目標裝置） | v7 | 9.89 | 1.54 | 9.24 |
| i7-8565U（目標裝置） | v9 | 9.46 | 0.52 | 8.67 |

> 直接執行正式系統程式（非獨立測速腳本），自動完成校準並持續訓練，每組測試60秒，前5秒作為系統暖機期間不計入統計；逐秒記錄該秒內實際完成之幀數換算吞吐量。攝影機依實際規格為30 FPS。i7-8565U為Intel第8代U系列行動處理器（4核8緒，TDP 15W）；i5-12600KF為第12代桌上型K系列處理器（6P+4E共16緒，TDP 125W），兩者屬不同世代與不同功耗設計等級之處理器；本研究刻意選用前者作為目標測試裝置，用以模擬效能較低階、以續航力為優先考量之行動部署情境，而非以效能較弱裝置為研究限制。P5(FPS) 為逐秒吞吐量之第5百分位（最差情況），非第95百分位。

結果顯示，含 GUI 之實際顯示吞吐量遠低於表 20 所示之處理管線延遲換算而得之產能（i5-12600KF 上，YOLOv7-tiny 為 24.17 FPS，YOLOv9-tiny 為 22.61 FPS）。此一落差之可能成因為：表 20 之處理管線測試僅涵蓋模型推論與 MediaPipe 特徵分析，並未包含正式系統逐幀執行之鏡像翻轉、畫面縮放補邊等前處理，以及關鍵點標示、信心分數橫條圖與文字疊圖等畫面合成繪製工作；上述攝影機影像前處理與 GUI 畫面合成之整體成本，可能才是系統實際顯示吞吐量之主要瓶頸，而非模型推論本身或攝影機擷取等待。

於本系統實際目標部署裝置 i7-8565U 上，兩模型之實際顯示吞吐量分別僅為 9.89 FPS（YOLOv7-tiny）與 9.46 FPS（YOLOv9-tiny），兩模型間之差異較不具實質區別——顯示在效能較弱之目標裝置上，攝影機影像前處理與 GUI 畫面合成之整體成本已大幅超過 YOLO 模型推論成本本身之差異。此發現與（二）中，兩模型於純推論階段與處理管線階段之延遲差距（分別約 1.52 倍與 1.25 倍）方向一致但幅度有限形成對比，凸顯系統於目標裝置上之實際效能瓶頸主要在於前述影像處理與畫面合成成本，而非模型選擇本身；此為本研究測速方法逐層拆解（純推論→處理管線→含GUI實際顯示）後方才發現之重要限制，已一併納入論文 5.3 節研究限制與未來研究方向。

**（四）最終模型選擇**

綜合表 19 至表 21 之結果，本研究對模型選型之論述需一併考量處理延遲與實際顯示吞吐量兩個層次。就處理延遲而言，YOLOv7-tiny 於純推論與處理管線階段均快於 YOLOv9-tiny（純推論約 1.52 倍、處理管線約 1.25 倍），此一速度差距方向與兩模型之辨識效能差距（F1-score 差 0.012、mAP@0.5 差 0.031）並存。惟表 21 之實測結果顯示，此一處理延遲差距於加入 GUI 繪製後之實際顯示吞吐量層次已大幅被稀釋：於本系統之實際目標部署裝置 i7-8565U 上，兩模型之顯示吞吐量差距（9.89 對 9.46 FPS）已無實質意義，顯示現行系統於該裝置上之效能瓶頸主要在於攝影機影像處理與 GUI 渲染架構之整體成本，而非 YOLO 模型本身之運算成本差異。

因此，本研究之選模考量不再以「模型間之效能餘裕差異」為依據——因該差異於使用者實際可感知之顯示吞吐量層次並不成立——而改以辨識效能作為關鍵判準。YOLOv9-tiny 於整體 F1-score（0.873 對 0.861）與 mAP@0.5（0.896 對 0.865）均優於 YOLOv7-tiny，於 angry 等辨識難度較高類別上表現亦更為突出；在兩模型之實際顯示吞吐量已無實質差異的前提下，選擇辨識效能較佳者更符合系統設計目的。**本研究最終選擇 YOLOv9-tiny（batch size 32、epoch 225）作為臉部運動即時回饋系統整合之核心表情辨識模型**，並將其導出為 ONNX 格式透過 ONNX Runtime 於 CPU 環境執行。

惟需說明，兩模型於本系統目標裝置上之實際顯示吞吐量（9.46～9.89 FPS）均遠低於本研究原先預期之 24～30 FPS 即時互動門檻，此一限制之成因、已進行之優化（中文文字疊圖繪製方式優化，將原先每個帶外框文字標籤所需之9次獨立影像格式轉換合併為1次）與後續改善方向（如將攝影機擷取與GUI繪製解耦成獨立執行緒），詳見論文 5.3 節。

### 常見問題

**Q1：有關資料建置的部分，是否都有對應程式碼？**

- **分層抽樣 train/val**：有。`dataset_pipeline/01_train_val_split/train_val_split.py`，依類別比例分層抽樣，並固定 `random_state=42`，確保可重跑得到相同的切分結果。
- **少數類別擴增**：有。`dataset_pipeline/02_balance_augmentation/make_5_balanced_data.py`，僅對訓練集未達 400 張的類別做水平翻轉/明暗對比調整/高斯模糊擴增。
- **加入外部圖庫測試集（test_100）**：這部分**沒有程式碼**，僅附上 Google 雲端硬碟連結（圖片本身，因涉及圖庫版權與檔案體積未直接入 repo）。但標註品質有對應查核程式碼：`dataset_pipeline/03_test_100_annotation_reliability/`，透過雙人盲標註計算 Cohen's kappa 與框選 IoU 一致性，確保這 100 張圖的標籤可靠。

**Q2：是否提供 YOLOv7-tiny 與 YOLOv9-tiny 的訓練語法？各 batch size 挑選最佳權重、匯出 CPU ONNX 是否有對應程式？**

- **訓練與匯出語法**：有，完整記錄在 `model_training/yolov7/commands.txt` 與 `model_training/yolov9/commands.txt`，含 b16/b32/b64 三組訓練指令、驗證/測試指令，以及匯出 CPU ONNX 的指令（`export.py --device cpu`）。
- **各 batch size 挑選最佳權重**：不需要額外程式碼——YOLO 訓練框架本身會依驗證集 mAP 自動儲存最佳 epoch 權重（`best.pt`），三組 batch size 訓練完後直接比較各自的 `best.pt` 在驗證集上的表現即可選出最佳批次設定，此比較結果同樣記錄在上述 `commands.txt` 中。

## 3. 臉部運動系統

App 使用已訓練好的 ONNX 權重（`app/weights/`），不需 GPU，CPU 即可即時推論，搭配 MediaPipe 做臉部特徵點追蹤以量測動作幅度。系統最終採用 **YOLOv9-tiny**（見上方 3.4 節選型結論）。

**`Facial_Exercises_Training.py`：系統本體**，即時引導使用者做面部運動並記錄幅度：

```bash
cd app
python Facial_Exercises_Training.py ^
    --weights weights/v9best.onnx ^
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
| `--weights` | ONNX 權重路徑，預設 `v9best.onnx`（依 3.4 節最終選型結論） |
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
