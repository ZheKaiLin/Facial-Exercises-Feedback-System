"""
端到端效能測速腳本，供論文 3.4（二）「相同條件下之端到端效能比較」使用。

背景：舊版測速只留下 FPS 彙總數字（mean/SD/P95 FPS），沒有留下逐幀原始延遲，
無法正確回推 P95「延遲」（P95 FPS 不等於 1/P95 延遲，兩者是分布兩端，不能互相套公式）。
本腳本直接以「延遲」為量測單位（ms），逐幀記錄原始樣本，統計量從原始樣本算，
FPS 只在最後另外報一個「平均吞吐量」與「顯示模式下的實際 wall-clock 吞吐量」，不混用。

三種模式：
  onnx     ：僅 ONNX 推論（前處理 + session.run），用同一張已擷取好的畫面重複推論，
             排除攝影機擷取等待，對應論文「ONNX CPU」列。
  pipeline ：前處理 + ONNX 推論 + MediaPipe FaceMesh 之處理延遲。計時前會先向攝影機
             擷取一段真實影格緩衝（預設300張，含真實人臉與些微幀間變化，避免
             MediaPipe 因逐幀完全相同畫面而落入不真實的最佳情況追蹤），計時迴圈本身
             只在緩衝內循環讀取、不再呼叫 cap.read()，因此不含攝影機供幀等待時間，
             對應論文「臉部運動系統（YOLO+MediaPipe+前處理）」列的「處理延遲」。
             （攝影機供幀速度是否會成為系統瓶頸，改由 display 模式量測。）
  display  ：與 pipeline 相同的每幀工作內容，但改成量測整段測試期間的 wall-clock
             實際吞吐量（frames / 總秒數），用來檢驗「處理延遲的倒數」是否等於
             「攝影機實際能餵幀的速度」——若攝影機硬體上限低於處理速度，
             display 吞吐量會被攝影機幀率封頂，此時不能再用處理延遲倒數宣稱系統吞吐量。

輸出：
  - <out>/<mode>_<model>_<repeat>.csv：逐幀延遲原始樣本（ms）
  - 主控台摘要：測試幀數、warm-up 幀數、總測試時長、攝影機回報之實際 FPS
    （cap.get(CAP_PROP_FPS)）、平均延遲(ms)、延遲SD(ms)、P95延遲(ms)、
    平均吞吐量FPS(=1000/平均延遲，僅 onnx/pipeline 模式)、
    display 模式之實際 wall-clock 吞吐量FPS、CPU平均/峰值(%)、記憶體峰值(MB)。

用法範例：
  # ONNX 純推論（純模型效能，對應表20「ONNX CPU」列）
  python bench_e2e.py --weights ../weights/v7best.onnx --model YOLOv7-tiny --mode onnx --frames 1000 --warmup 100 --repeats 3

  # 完整管線延遲（對應表20「臉部運動系統」列，報 ms 而非 FPS）
  python bench_e2e.py --weights ../weights/v7best.onnx --model YOLOv7-tiny --mode pipeline --frames 1000 --warmup 100 --repeats 3

  # 實際顯示吞吐量（同一模型在不同裝置上比較，例如 i5-12600KF vs i7-8565U）
  python bench_e2e.py --weights ../weights/v7best.onnx --model YOLOv7-tiny --mode display --duration 30 --warmup 100
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

SCRIPT_DIR = Path(__file__).resolve().parent
APP_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(APP_DIR))

from Facial_Exercises_Training import ONNXDetector, ALL_EXPRESSIONS, DISPLAY_W, DISPLAY_H  # noqa: E402

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
except Exception as e:
    mp_face_mesh = None
    print("MediaPipe 載入失敗，pipeline/display 模式將無法使用：", e)


def build_face_mesh():
    return mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )


def open_camera(source):
    cap_source = int(source) if str(source).isdigit() else str(source)
    cap = cv2.VideoCapture(cap_source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟攝影機來源：{source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)
    return cap


def summarize(latencies_ms):
    arr = np.asarray(latencies_ms, dtype=float)
    mean_ms = float(arr.mean())
    sd_ms = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    p95_ms = float(np.percentile(arr, 95))
    mean_fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    # P5 FPS：對每一幀的延遲個別換算成瞬時 FPS 後取第 5 百分位，
    # 不是用 P95 延遲直接換算（兩者不等價，因為 1/x 是非線性轉換）。
    inst_fps = 1000.0 / arr
    p5_fps = float(np.percentile(inst_fps, 5))
    return {
        "n_frames": len(arr),
        "mean_latency_ms": round(mean_ms, 2),
        "sd_latency_ms": round(sd_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "mean_fps": round(mean_fps, 1),
        "p5_fps": round(p5_fps, 1),
    }


def run_onnx_mode(detector, cap, frames, warmup):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("攝影機讀不到畫面，無法取得測試用影格")

    for _ in range(warmup):
        detector.infer(frame)

    latencies = []
    for _ in range(frames):
        t0 = time.perf_counter()
        detector.infer(frame)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def capture_frame_buffer(cap, buffer_size):
    """預先擷取一段真實影格，讓計時迴圈不必在計時區間內呼叫 cap.read()，
    避免攝影機供幀節奏（可能受硬體幀率或驅動延遲影響）污染處理延遲量測。"""
    print(f"  擷取 {buffer_size} 張真實影格作為緩衝...")
    buf = []
    while len(buf) < buffer_size:
        ret, frame = cap.read()
        if ret:
            buf.append(frame.copy())
    return buf


def run_pipeline_mode(detector, face_mesh, cap, frames, warmup, buffer_size=300):
    buf = capture_frame_buffer(cap, buffer_size)
    idx = [0]

    def next_frame():
        f = buf[idx[0] % len(buf)]
        idx[0] += 1
        return f

    for _ in range(warmup):
        f = next_frame()
        detector.infer(f)
        face_mesh.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

    latencies = []
    for _ in range(frames):
        f = next_frame()
        t0 = time.perf_counter()
        detector.infer(f)
        face_mesh.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def run_display_mode(detector, face_mesh, cap, duration_s, warmup):
    for _ in range(warmup):
        ret, frame = cap.read()
        if not ret:
            continue
        detector.infer(frame)
        face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    n_frames = 0
    t_start = time.perf_counter()
    while (time.perf_counter() - t_start) < duration_s:
        ret, frame = cap.read()
        if not ret:
            continue
        detector.infer(frame)
        face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        n_frames += 1
    elapsed = time.perf_counter() - t_start
    return n_frames, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", required=True, help="標籤，例如 YOLOv7-tiny，只用於輸出檔名與摘要")
    ap.add_argument("--mode", required=True, choices=["onnx", "pipeline", "display"])
    ap.add_argument("--frames", type=int, default=1000, help="onnx/pipeline 模式：測試幀數")
    ap.add_argument("--duration", type=float, default=30.0, help="display 模式：測試秒數")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--source", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=str(SCRIPT_DIR / "results"))
    opt = ap.parse_args()

    out_dir = Path(opt.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = ONNXDetector(opt.weights, class_names=ALL_EXPRESSIONS, img_size=(opt.imgsz, opt.imgsz))
    cap = open_camera(opt.source)
    cam_reported_fps = cap.get(cv2.CAP_PROP_FPS)

    face_mesh = None
    if opt.mode in ("pipeline", "display"):
        if mp_face_mesh is None:
            raise RuntimeError("此模式需要 MediaPipe，但載入失敗")
        face_mesh = build_face_mesh()

    print(f"模型：{opt.model}　權重：{opt.weights}")
    print(f"攝影機來源：{opt.source}　cap.get(CAP_PROP_FPS) 回報值：{cam_reported_fps}")
    print(f"測試模式：{opt.mode}　warm-up：{opt.warmup}")

    import os
    proc = psutil.Process(os.getpid())

    for rep in range(1, opt.repeats + 1):
        import threading
        stop_flag = {"stop": False}
        # 用背景 thread 輪詢 CPU/記憶體，避免影響主執行緒的計時精度
        collected = {"cpu": [], "mem": []}

        def _collector():
            proc.cpu_percent(interval=None)
            while not stop_flag["stop"]:
                collected["cpu"].append(proc.cpu_percent(interval=0.2))
                collected["mem"].append(proc.memory_info().rss / (1024 * 1024))

        t = threading.Thread(target=_collector, daemon=True)
        t.start()

        t_wall_start = time.perf_counter()
        if opt.mode == "onnx":
            latencies = run_onnx_mode(detector, cap, opt.frames, opt.warmup)
            wall_elapsed = time.perf_counter() - t_wall_start
            stop_flag["stop"] = True
            t.join(timeout=1.0)

            stats = summarize(latencies)
            csv_path = out_dir / f"onnx_{opt.model}_rep{rep}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["frame_idx", "latency_ms"])
                for i, v in enumerate(latencies):
                    w.writerow([i, round(v, 4)])

            print(f"\n[rep {rep}] ONNX 推論　n={stats['n_frames']}　總測試時長(wall)={wall_elapsed:.1f}s")
            print(f"  平均延遲={stats['mean_latency_ms']}ms  SD={stats['sd_latency_ms']}ms  P95延遲={stats['p95_latency_ms']}ms")
            print(f"  平均吞吐量={stats['mean_fps']} FPS（=1000/平均延遲）　P5 FPS={stats['p5_fps']}")

        elif opt.mode == "pipeline":
            latencies = run_pipeline_mode(detector, face_mesh, cap, opt.frames, opt.warmup)
            wall_elapsed = time.perf_counter() - t_wall_start
            stop_flag["stop"] = True
            t.join(timeout=1.0)

            stats = summarize(latencies)
            csv_path = out_dir / f"pipeline_{opt.model}_rep{rep}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["frame_idx", "latency_ms"])
                for i, v in enumerate(latencies):
                    w.writerow([i, round(v, 4)])

            print(f"\n[rep {rep}] 完整管線（不含GUI繪製）　n={stats['n_frames']}　總測試時長(wall)={wall_elapsed:.1f}s")
            print(f"  平均延遲={stats['mean_latency_ms']}ms  SD={stats['sd_latency_ms']}ms  P95延遲={stats['p95_latency_ms']}ms")
            print(f"  平均吞吐量={stats['mean_fps']} FPS（=1000/平均延遲，處理延遲倒數，非攝影機實際餵幀速度）")
            print(f"  P5 FPS={stats['p5_fps']}")

        else:  # display
            n_frames, wall_elapsed = run_display_mode(detector, face_mesh, cap, opt.duration, opt.warmup)
            stop_flag["stop"] = True
            t.join(timeout=1.0)

            display_fps = n_frames / wall_elapsed if wall_elapsed > 0 else 0.0
            print(f"\n[rep {rep}] 實際顯示吞吐量（wall-clock，含攝影機擷取等待，不含GUI繪製）")
            print(f"  測試時長={wall_elapsed:.1f}s　實際處理幀數={n_frames}")
            print(f"  實際顯示吞吐量={display_fps:.1f} FPS　（攝影機回報上限={cam_reported_fps} FPS）")
            if cam_reported_fps and display_fps > cam_reported_fps * 0.95:
                print("  註：實際吞吐量已接近或超過攝影機回報幀率上限，此時吞吐量可能受攝影機硬體封頂，"
                      "不能單純視為處理效能的完整餘裕。")

        if cpu_samples := collected["cpu"]:
            print(f"  CPU 平均={np.mean(cpu_samples):.1f}%　峰值={np.max(cpu_samples):.1f}%")
        if mem_samples := collected["mem"]:
            print(f"  記憶體峰值={np.max(mem_samples):.1f}MB")

    cap.release()


if __name__ == "__main__":
    main()
