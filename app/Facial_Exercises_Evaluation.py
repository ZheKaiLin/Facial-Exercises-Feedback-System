

import sys, re, time, math
from pathlib import Path
import cv2
import numpy as np

# ── 從主程式匯入所有必要元件 ──────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from Facial_Exercises_Training import (
    ONNXDetector, FacialExercisesSystem, ExpressionTrainingSystem,
    FaceAlignmentChecker, SoundManager, DataLogger,
    draw_ui, draw_training_ui, draw_finished_summary,
    draw_filled_rect_with_alpha, draw_text_outline,
    resize_with_padding, resize_for_window, key_pressed,
    normalize_class_name,
    ALL_EXPRESSIONS, DISPLAY_W, DISPLAY_H, WINDOW_W, WINDOW_H,
    C_WHITE, C_GREEN, C_RED, C_BLUE, C_YELLOW, C_ORANGE,
    C_LIGHT_GRAY, C_BLACK, C_GRAY, C_CYAN,
    zh_exercise,
)

# ==========================================================
# 路徑輔助（PyInstaller 相容）
# ==========================================================
def _find_resource(filename: str) -> Path:
    """找到打包後的資源檔（ONNX 模型等唯讀檔案）。"""
    if getattr(sys, 'frozen', False):
        meipass = Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
        candidate = meipass / filename
        if candidate.exists():
            return candidate
        return Path(sys.executable).parent / filename
    return _HERE / filename


def _get_save_dir() -> Path:
    """evaluation_data 永遠放在 .exe 旁邊，不放在解壓暫存目錄。"""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent / "evaluation_data"
    return _HERE / "evaluation_data"


def _auto_participant_id(save_dir: Path) -> str:
    """掃描既有前測 CSV，自動產生下一組受試者編號（01, 02, …）。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    nums = []
    for f in save_dir.glob("P*_pre_test_exercises.csv"):
        m = re.match(r"P(\d+)_", f.name)
        if m:
            nums.append(int(m.group(1)))
    return str((max(nums) + 1) if nums else 1).zfill(2)


# ==========================================================
# 評估階段定義
# ==========================================================
EVAL_PHASES = [
    # (session,     type,     exercise,       reps,  display_name)
    ("training",  "expr",   None,            2, "訓練 ①  臉部表情練習"),
    ("pre_test",  "exercises",   "Face_Lift",     3, "前測 ①  臉部拉提"),
    ("pre_test",  "exercises",   "Double_Chin",   3, "前測 ②  下顎線條"),
    ("tutorial",  "wait",   None,            0, "系統說明（Tutorial）"),
    ("training",  "exercises",   "Face_Lift",    10, "訓練 ②  臉部拉提"),
    ("training",  "exercises",   "Double_Chin",  10, "訓練 ③  下顎線條"),
    ("post_test", "exercises",   "Face_Lift",     3, "後測 ①  臉部拉提"),
    ("post_test", "exercises",   "Double_Chin",   3, "後測 ②  下顎線條"),
]

SESSION_COLOR = {
    "pre_test":  (30,  160, 255),
    "tutorial":  (0,   210, 210),
    "training":  (50,  220,  80),
    "post_test": (200, 100, 255),
}

SESSION_LABEL = {
    "pre_test":  "前測",
    "tutorial":  "系統說明",
    "training":  "正式訓練",
    "post_test": "後測",
}

PHASE_INTRO_SEC = 4.0
PHASE_DONE_SEC  = 2.5


# ==========================================================
# 評估專屬畫面
# ==========================================================
def draw_waiting_start(frame, participant_id):
    h, w, _ = frame.shape
    col = (80, 200, 255)
    draw_filled_rect_with_alpha(frame, (0, 0), (w, h), (8, 10, 18), 0.92)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), col, 8)

    draw_text_outline(frame, "臉部運動系統評估",
                      (w // 2 - 230, 170), cv2.FONT_HERSHEY_DUPLEX, 1.55, col, 2)
    draw_text_outline(frame, f"受試者編號：P{participant_id}",
                      (w // 2 - 160, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.85, C_WHITE, 1)
    draw_text_outline(frame, f"共 {len(EVAL_PHASES)} 個評估階段，請準備好後開始",
                      (w // 2 - 245, 318), cv2.FONT_HERSHEY_SIMPLEX, 0.72, C_LIGHT_GRAY, 1)

    if int(time.time() * 2) % 2 == 0:
        cv2.rectangle(frame, (w // 2 - 200, 410), (w // 2 + 200, 472), col, -1)
        draw_text_outline(frame, "按  Enter  開始評估",
                          (w // 2 - 165, 454), cv2.FONT_HERSHEY_DUPLEX, 0.95, (8, 10, 18), 2)
    else:
        cv2.rectangle(frame, (w // 2 - 200, 410), (w // 2 + 200, 472), col, 2)
        draw_text_outline(frame, "按  Enter  開始評估",
                          (w // 2 - 165, 454), cv2.FONT_HERSHEY_DUPLEX, 0.95, col, 2)


def draw_phase_intro(frame, phase, phase_idx, total, countdown):
    session, ptype, exercise, reps, name = phase
    h, w, _ = frame.shape
    col = SESSION_COLOR.get(session, C_WHITE)

    draw_filled_rect_with_alpha(frame, (0, 0), (w, h), (10, 12, 20), 0.90)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), col, 8)

    # 頂部進度列
    cell_w = (w - 80) // total
    for i in range(total):
        cx = 40 + i * cell_w + cell_w // 2
        c  = SESSION_COLOR.get(EVAL_PHASES[i][0], (80, 80, 80))
        if i < phase_idx:
            cv2.circle(frame, (cx, 42), 12, c, -1)
        elif i == phase_idx:
            cv2.circle(frame, (cx, 42), 14, col, 3)
            cv2.circle(frame, (cx, 42), 7,  col, -1)
        else:
            cv2.circle(frame, (cx, 42), 10, (60, 60, 80), 1)
        if i < total - 1:
            cv2.line(frame, (cx + 14, 42), (cx + cell_w - 14, 42), (60, 60, 80), 1)

    draw_text_outline(frame, SESSION_LABEL.get(session, session),
                      (w // 2 - 60, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.76, col, 1)
    draw_text_outline(frame, name,
                      (w // 2 - 240, 185), cv2.FONT_HERSHEY_DUPLEX, 1.22, col, 3)

    hints = {
        "pre_test":  "前測不提供額外指導，請自然完成動作",
        "tutorial":  "研究者將說明系統操作方式（約 5-10 分鐘）",
        "training":  "請依照系統提示完成訓練，系統回饋可幫助修正動作",
        "post_test": "後測流程與前測相同，請自然完成動作",
    }
    draw_text_outline(frame, hints.get(session, ""),
                      (w // 2 - 280, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.72, C_LIGHT_GRAY, 1)

    if ptype == "exercises":
        draw_text_outline(frame,
                          f"動作：{zh_exercise(exercise)}    回合數：{reps} 回合",
                          (w // 2 - 200, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.74, C_WHITE, 1)
    elif ptype == "expr":
        draw_text_outline(frame,
                          f"五類表情（angry / happy / neutral / sad / surprised）各 {reps} 輪",
                          (w // 2 - 290, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.67, C_WHITE, 1)
    elif ptype == "wait":
        draw_text_outline(frame, "說明結束後按  空白鍵  繼續",
                          (w // 2 - 185, 312), cv2.FONT_HERSHEY_DUPLEX, 0.88, C_GREEN, 2)

    if ptype != "wait":
        if countdown > 0:
            draw_text_outline(frame, f"{math.ceil(countdown)}  秒後自動開始",
                              (w // 2 - 130, 410), cv2.FONT_HERSHEY_DUPLEX, 1.05, col, 2)
        draw_text_outline(frame, "按  Enter  立即開始",
                          (w // 2 - 130, 478), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (200, 200, 100), 1)


def draw_tutorial_wait(frame):
    h, w, _ = frame.shape
    col = SESSION_COLOR["tutorial"]
    draw_filled_rect_with_alpha(frame, (0, 0), (w, h), (10, 14, 14), 0.90)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), col, 8)

    draw_text_outline(frame, "系統說明  Tutorial",
                      (w // 2 - 175, 185), cv2.FONT_HERSHEY_DUPLEX, 1.35, col, 2)
    draw_text_outline(frame, "研究者正在說明系統操作方式",
                      (w // 2 - 215, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.80, C_WHITE, 1)

    items = [
        "主介面功能選擇方式",
        "個人化基準校準步驟（C 鍵）",
        "訓練介面各區塊說明（倒數、達標提示、分數）",
        "動作修正引導文字的解讀方式",
    ]
    for i, item in enumerate(items):
        draw_text_outline(frame, f"  .  {item}",
                          (w // 2 - 240, 320 + i * 36), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (170, 200, 170), 1)

    if int(time.time() * 2) % 2 == 0:
        cv2.rectangle(frame, (w // 2 - 225, 510), (w // 2 + 225, 572), col, -1)
        draw_text_outline(frame, "說明結束後按  空白鍵  繼續",
                          (w // 2 - 198, 554), cv2.FONT_HERSHEY_DUPLEX, 0.88, (10, 14, 14), 2)
    else:
        cv2.rectangle(frame, (w // 2 - 225, 510), (w // 2 + 225, 572), col, 2)
        draw_text_outline(frame, "說明結束後按  空白鍵  繼續",
                          (w // 2 - 198, 554), cv2.FONT_HERSHEY_DUPLEX, 0.88, col, 2)


def draw_phase_done(frame, phase, saved_path=None):
    session, _, _, _, name = phase
    h, w, _ = frame.shape
    col = SESSION_COLOR.get(session, C_GREEN)
    draw_filled_rect_with_alpha(frame, (0, 0), (w, h), (6, 16, 10), 0.90)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), C_GREEN, 6)

    draw_text_outline(frame, "V  階段完成",
                      (w // 2 - 130, 230), cv2.FONT_HERSHEY_DUPLEX, 1.45, C_GREEN, 2)
    draw_text_outline(frame, name,
                      (w // 2 - 210, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.84, col, 1)
    if saved_path:
        draw_text_outline(frame, f"已儲存：{Path(saved_path).name}",
                          (w // 2 - 280, 374), cv2.FONT_HERSHEY_SIMPLEX, 0.62, C_LIGHT_GRAY, 1)
    draw_text_outline(frame, "即將進入下一階段...",
                      (w // 2 - 155, 444), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (150, 150, 150), 1)


def draw_all_done(frame, participant_id, save_dir: Path):
    h, w, _ = frame.shape
    draw_filled_rect_with_alpha(frame, (0, 0), (w, h), (10, 10, 20), 0.92)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), C_YELLOW, 6)

    draw_text_outline(frame, "評估全部完成！",
                      (w // 2 - 170, 195), cv2.FONT_HERSHEY_DUPLEX, 1.5, C_YELLOW, 2)
    draw_text_outline(frame, f"受試者  P{participant_id}  感謝您的參與",
                      (w // 2 - 240, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.82, C_WHITE, 1)

    draw_text_outline(frame, "已儲存的資料檔案：",
                      (w // 2 - 240, 346), cv2.FONT_HERSHEY_SIMPLEX, 0.68, C_LIGHT_GRAY, 1)
    csvs = sorted(save_dir.glob(f"P{participant_id}_*.csv")) if save_dir.exists() else []
    for i, f in enumerate(csvs[:6]):
        draw_text_outline(frame, f"  - {f.name}",
                          (w // 2 - 240, 380 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 100), 1)

    draw_text_outline(frame, "按  Q  或  Esc  關閉程式",
                      (w // 2 - 175, 578), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (200, 200, 100), 1)


# ==========================================================
# 主評估流程
# ==========================================================
def _pretest_calibrate(exercises_sys, exercise):
    """前後測使用寬鬆門檻自動校準，不需要使用者手動按 C。"""
    dbg = exercises_sys.debug_vals
    if exercise == "Face_Lift":
        y0 = dbg.get("y_diff", 0.02)
        exercises_sys.calib_baseline["Face_Lift"] = y0
        exercises_sys.custom_thresh["Face_Lift"] = (y0 + 0.005) * 1
    elif exercise == "Double_Chin":
        D0 = dbg.get("m_height", 0.03)
        exercises_sys.calib_D0["Double_Chin"] = max(D0, 1e-6)
        exercises_sys.calib_baseline["Double_Chin"] = D0
        exercises_sys.custom_thresh["Double_Chin"] = D0 * 1
    exercises_sys.calib_confirm_time = time.time()
    exercises_sys.is_calibrated = True
    exercises_sys.state = "START"
    exercises_sys.reset_round_timer()
    exercises_sys.reset_prep_timer()


def _make_phase_objects(phase):
    session, ptype, exercise, reps, name = phase
    exercises_sys = expr_trainer = expr_align = None
    if ptype == "exercises":
        exercises_sys = FacialExercisesSystem()
        exercises_sys.max_reps = reps
        return "exercises", exercises_sys, expr_trainer, expr_align
    elif ptype == "expr":
        expr_trainer = ExpressionTrainingSystem(rounds=reps)
        expr_align   = FaceAlignmentChecker()
        return "EXPR", exercises_sys, expr_trainer, expr_align
    elif ptype == "wait":
        return "TUTORIAL_WAIT", exercises_sys, expr_trainer, expr_align
    return "PHASE_DONE", exercises_sys, expr_trainer, expr_align


def evaluation_run(weights=None, source="0"):
    save_dir  = _get_save_dir()
    auto_pid  = _auto_participant_id(save_dir)
    onnx_path = str(weights or _find_resource("v7best.onnx"))

    print(f"========== 評估流程啟動 ==========")
    print(f"受試者編號：P{auto_pid}")
    print(f"模型路徑  ：{onnx_path}")
    print(f"資料目錄  ：{save_dir}")

    if not Path(onnx_path).exists():
        print("錯誤：找不到模型檔案：", onnx_path); return

    model = ONNXDetector(onnx_path, class_names=ALL_EXPRESSIONS, img_size=(224, 224))

    # ── 三個獨立 DataLogger，pre/training/post 資料完全分開存檔 ─
    loggers = {
        sess: DataLogger(participant_id=auto_pid, session=sess, save_dir=save_dir)
        for sess in ("pre_test", "training", "post_test")
    }

    # ── 攝影機 ────────────────────────────────────────────────
    cap_src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(cap_src, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        print("錯誤：無法開啟攝影機"); return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)

    window_name = "臉部運動系統評估"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_W, WINDOW_H)

    sound_mgr = SoundManager()

    # ── 錄影輔助 ──────────────────────────────────────────────
  
    _fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    _rec_fps  = 25.0
    _rec_size = (DISPLAY_W, DISPLAY_H)

    def _start_rec(session, label):
        """開啟新的 VideoWriter，回傳物件。"""
        fname = save_dir / f"P{auto_pid}_{session}_{label}.mp4"
        vw = cv2.VideoWriter(str(fname), _fourcc, _rec_fps, _rec_size)
        print(f"[錄影開始] {fname.name}")
        return vw

    def _stop_rec(vw):
        if vw is not None and vw.isOpened():
            vw.release()
            print("[錄影結束]")

    video_writer: cv2.VideoWriter | None = None

    # ── 流程狀態 ──────────────────────────────────────────────
    ctrl              = "WAITING_START"
    phase_idx         = 0
    phase_intro_start = time.time()

    exercises_sys         = None
    expr_trainer     = None
    expr_align       = None

    prev_exercises_state  = None
    prev_prep_ceil   = None
    expr_started     = False
    phase_logged     = False
    phase_saved_path = None
    phase_done_time  = None

    # ── 即時動作幅度追蹤──────────────────────
    # 「維持中」狀態逐幀收集當下幾何訊號，回合完成（reps 增加）時結算峰值/平均值，
    amp_buffer            = []
    amp_round_start_time  = None
    prev_reps_for_amp     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # 鏡像顯示，符合使用者面對攝影機的直覺

        im0, pad_x, pad_y, disp_scale = resize_with_padding(
            frame, DISPLAY_W, DISPLAY_H, return_info=True)

        # ── ONNX 推論（僅訓練畫面）────────────────────────────
        yolo_results = {"class": "none", "conf": 0.0, "all_conf": {}, "probs": None}
        if ctrl in ("exercises", "EXPR"):
            onnx_out = model.infer(im0)
            yolo_results = {
                "class":    normalize_class_name(onnx_out.get("class", "none")),
                "conf":     float(onnx_out.get("conf", 0.0)),
                "all_conf": {normalize_class_name(k): float(v)
                             for k, v in onnx_out.get("all_conf", {}).items()},
                "probs":    onnx_out.get("probs", None),
            }

        # ── 畫面渲染 ──────────────────────────────────────────
        if ctrl == "WAITING_START":
            draw_waiting_start(im0, auto_pid)

        elif ctrl == "PHASE_INTRO":
            phase     = EVAL_PHASES[phase_idx]
            elapsed   = time.time() - phase_intro_start
            remaining = max(0.0, PHASE_INTRO_SEC - elapsed)
            draw_phase_intro(im0, phase, phase_idx, len(EVAL_PHASES), remaining)
            if remaining <= 0 and phase[1] != "wait":
                ctrl, exercises_sys, expr_trainer, expr_align = _make_phase_objects(phase)
                expr_started = False
                phase_logged = False
                prev_exercises_state = prev_prep_ceil = None
                amp_buffer, amp_round_start_time, prev_reps_for_amp = [], None, 0
                # 開始錄影
                _stop_rec(video_writer)
                s, _, ex, _, _ = phase
                label = ex if ex else "expression"
                video_writer = _start_rec(s, label)

        elif ctrl == "TUTORIAL_WAIT":
            draw_tutorial_wait(im0)

        elif ctrl == "exercises":
            phase = EVAL_PHASES[phase_idx]
            session, _, exercise, reps, name = phase

            state, yreps, dbg, y_cls, y_conf = exercises_sys.analyze(
                exercise, im0, yolo_results,
                mp_frame=frame, mp_offset=(pad_x, pad_y), mp_scale=disp_scale)

            if session in ("pre_test", "post_test") and state == "校正":
                if exercises_sys.debug_vals.get("is_aligned", False):
                    _pretest_calibrate(exercises_sys, exercise)

            dbg["max_reps"] = reps
            # UI 標題：前後測顯示「前測 / 後測 + 動作」，不顯示「訓練」
            dbg["exercise_display"] = name

            is_tr = dbg.get("is_target_reached", False)

            # ── 即時動作幅度收集：狀態機精確掌握「維持中」真實起訖幀，
            # 逐幀收集當下幾何訊號，回合完成（reps 增加）時直接結算峰值/平均值，
            
            if state == "維持中":
                if prev_exercises_state != "維持中":
                    amp_buffer = []
                    amp_round_start_time = time.time()
                raw_val = dbg.get("y_diff") if exercise == "Face_Lift" else dbg.get("m_height")
                if raw_val is not None:
                    amp_buffer.append(float(raw_val))

            if exercises_sys.reps != prev_reps_for_amp:
                # reps 剛增加：代表上一回合為「有效記錄」（Tₛ=0 之重做回合不會使 reps 增加，
                # amp_buffer 會在下次重新進入「維持中」時自然被清空，不會誤記為有效回合）
                baseline = exercises_sys.calib_baseline.get(exercise)
                if baseline is not None and amp_buffer:
                    buf = np.asarray(amp_buffer, dtype=float)
                    if exercise == "Face_Lift":
                        extremum = float(np.min(buf))
                        amp      = max(baseline - extremum, 0.0)
                        amp_mean = max(baseline - float(np.mean(buf)), 0.0)
                    else:  # Double_Chin
                        extremum = float(np.max(buf))
                        amp      = max(extremum - baseline, 0.0)
                        amp_mean = max(float(np.mean(buf)) - baseline, 0.0)
                    duration = (time.time() - amp_round_start_time) if amp_round_start_time else 0.0
                    loggers[session].log_amplitude(
                        exercise, exercises_sys.reps, amp, amp_mean, extremum, duration, len(buf))
                amp_buffer = []
                prev_reps_for_amp = exercises_sys.reps

            if state != prev_exercises_state:
                if state == "準備中":     prev_prep_ceil = None
                elif state == "維持中":   sound_mgr.play_training_start()
                elif state == "RESET":    sound_mgr.play_rest()
                elif state == "FINISHED": sound_mgr.play_finished()
                prev_exercises_state = state
            if state == "準備中":
                pc = math.ceil(dbg.get("prep_remaining", 3.0))
                if pc != prev_prep_ceil and 1 <= pc <= 3:
                    sound_mgr.play_prep_tick(pc)
                    prev_prep_ceil = pc
            if state == "維持中" and not is_tr:
                sound_mgr.play_error_continuous()

            draw_ui(im0, state, yreps, dbg, y_cls, y_conf,
                    exercise, exercises_sys.custom_thresh,
                    exercises_sys.hold_duration, exercises_sys.hold_start_time,
                    exercises_sys.rest_start_time, exercises_sys.rest_duration)

            if state == "FINISHED" and not phase_logged:
                _stop_rec(video_writer); video_writer = None
                lg = loggers[session]
                for i, (sc, ts) in enumerate(
                        zip(exercises_sys.quality_history, exercises_sys.ts_history), 1):
                    lg.log_round(exercise, i, sc, ts)
                saved = lg.save()
                phase_saved_path = saved[0] if saved else None
                phase_logged    = True
                ctrl            = "PHASE_DONE"
                phase_done_time = time.time()
                print(f"[儲存] {session}/{exercise}: {saved}")

        elif ctrl == "EXPR":
            phase   = EVAL_PHASES[phase_idx]
            session = phase[0]

            is_aligned, align_msg = expr_align.check(im0)
            status, feedback, color, debug = expr_trainer.analyze(yolo_results, is_aligned)
            confs = yolo_results.get("all_conf", {})

            if not expr_started and expr_trainer.state == "IDLE":
                expr_trainer.start()
                expr_started = True

            if expr_trainer.state == "DONE":
                draw_finished_summary(im0, expr_trainer)
                if not phase_logged:
                    _stop_rec(video_writer); video_writer = None
                    lg = loggers[session]
                    for idx, (expr, sc) in enumerate(
                            zip(expr_trainer.sequence, expr_trainer.scores), 1):
                        lg.log_expression_round(expr, idx, sc)
                    saved = lg.save()
                    phase_saved_path = saved[0] if saved else None
                    phase_logged    = True
                    ctrl            = "PHASE_DONE"
                    phase_done_time = time.time()
                    print(f"[儲存] {session}/expression: {saved}")
            else:
                draw_training_ui(im0, expr_trainer, status, feedback, color,
                                 debug, is_aligned, align_msg, confs)

        elif ctrl == "PHASE_DONE":
            draw_phase_done(im0, EVAL_PHASES[phase_idx], phase_saved_path)
            if time.time() - phase_done_time >= PHASE_DONE_SEC:
                phase_idx += 1
                if phase_idx >= len(EVAL_PHASES):
                    ctrl = "ALL_DONE"
                else:
                    nxt = EVAL_PHASES[phase_idx]
                    ctrl = "TUTORIAL_WAIT" if nxt[1] == "wait" else "PHASE_INTRO"
                    if ctrl == "PHASE_INTRO":
                        phase_intro_start = time.time()

        elif ctrl == "ALL_DONE":
            draw_all_done(im0, auto_pid, save_dir)

        # ── 錄影寫入（exercises / EXPR 狀態才寫幀）────────────────────
        if ctrl in ("exercises", "EXPR") and video_writer is not None:
            video_writer.write(im0)

        cv2.imshow(window_name, resize_for_window(im0))
        key = cv2.waitKeyEx(30)
        if key == -1:
            continue

        if key_pressed(key, "q") or key == 27:
            _stop_rec(video_writer)
            break

        if ctrl == "WAITING_START":
            if key == 13:  # Enter：進入第一階段介紹
                phase_intro_start = time.time()
                ctrl = "PHASE_INTRO"

        elif ctrl == "PHASE_INTRO":
            if key == 13:  # Enter：立即開始
                phase = EVAL_PHASES[phase_idx]
                ctrl, exercises_sys, expr_trainer, expr_align = _make_phase_objects(phase)
                expr_started = False
                phase_logged = False
                prev_exercises_state = prev_prep_ceil = None
                amp_buffer, amp_round_start_time, prev_reps_for_amp = [], None, 0
                # Enter 提前開始也要啟動錄影
                _stop_rec(video_writer)
                s, _, ex, _, _ = phase
                video_writer = _start_rec(s, ex if ex else "expression")

        elif ctrl == "TUTORIAL_WAIT":
            if key == 32:
                phase_idx += 1
                if phase_idx >= len(EVAL_PHASES):
                    ctrl = "ALL_DONE"
                else:
                    ctrl = "PHASE_INTRO"
                    phase_intro_start = time.time()

        elif ctrl == "exercises":
            session  = EVAL_PHASES[phase_idx][0]
            exercise = EVAL_PHASES[phase_idx][2]
            # 校準只在正式訓練階段由使用者按 C 確認
            if session == "training" and key_pressed(key, "c") \
                    and exercises_sys and exercises_sys.state == "校正":
                exercises_sys.confirm_calibration(exercise)
                lg = loggers.get(session)
                if lg:
                    if exercise == "Face_Lift":
                        lg.log_calibration("Face_Lift",
                            exercises_sys.debug_vals.get("y_diff", 0.0),
                            exercises_sys.custom_thresh.get("Face_Lift", 0.0))
                    elif exercise == "Double_Chin":
                        lg.log_calibration("Double_Chin",
                            exercises_sys.calib_D0.get("Double_Chin", 0.0),
                            exercises_sys.custom_thresh.get("Double_Chin", 0.0))

        elif ctrl == "EXPR":
            if key_pressed(key, "r") and expr_trainer:
                expr_trainer.reset()
                expr_started = False

    cap.release()
    cv2.destroyAllWindows()


# ==========================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="臉部運動系統評估流程自動化")
    parser.add_argument("--source",  type=str, default="0")
    parser.add_argument("--weights", type=str, default=None)
    opt = parser.parse_args()
    evaluation_run(weights=opt.weights, source=opt.source)
