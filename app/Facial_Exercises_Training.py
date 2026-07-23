import argparse
import os
import sys
import time
import math
import threading
import winsound
import csv
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort

# ==========================================================
# 中文文字顯示工具：OpenCV 內建 putText 不支援中文，因此改用 PIL 繪製中文
# ==========================================================
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

_ORIG_CV2_PUTTEXT = cv2.putText
_ORIG_CV2_GETTEXTSIZE = cv2.getTextSize

_CJK_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msjh.ttc",
    r"C:\Windows\Fonts\mingliu.ttc",
    r"C:\Windows\Fonts\NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]

_CJK_FONT_PATH = None
for _font_path in _CJK_FONT_CANDIDATES:
    if os.path.exists(_font_path):
        _CJK_FONT_PATH = _font_path
        break

_FONT_CACHE = {}

def _has_cjk(text):
    text = str(text)
    return any(ord(ch) > 127 for ch in text)

def _get_cjk_font(font_scale=0.6, thickness=1):
    size = max(12, int(float(font_scale) * 32 + int(thickness) * 2))
    key = (size, _CJK_FONT_PATH)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    if ImageFont is not None and _CJK_FONT_PATH is not None:
        try:
            font = ImageFont.truetype(_CJK_FONT_PATH, size)
        except Exception:
            font = ImageFont.load_default()
    elif ImageFont is not None:
        font = ImageFont.load_default()
    else:
        font = None

    _FONT_CACHE[key] = font
    return font

_TEXT_SIZE_CACHE = {}


def _measure_cjk_text(text, fontFace, fontScale, thickness=1):
    """
    量測中文文字尺寸，並加入快取，避免每一幀重複量測造成卡頓。
    """
    text = str(text)
    key = (text, float(fontScale), int(thickness), _CJK_FONT_PATH)
    if key in _TEXT_SIZE_CACHE:
        return _TEXT_SIZE_CACHE[key]

    font = _get_cjk_font(fontScale, thickness)
    if Image is None or ImageDraw is None or font is None:
        size = _ORIG_CV2_GETTEXTSIZE(text, fontFace, fontScale, thickness)
        _TEXT_SIZE_CACHE[key] = size
        return size

    dummy = Image.new("RGB", (8, 8))
    draw = ImageDraw.Draw(dummy)

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = max(1, bbox[2] - bbox[0])
        h = max(1, bbox[3] - bbox[1])
    except Exception:
        w = int(len(text) * float(fontScale) * 20)
        h = int(float(fontScale) * 34)

    size = ((w, h), 0)
    _TEXT_SIZE_CACHE[key] = size
    return size


def _put_chinese_text(img, text, org, fontFace, fontScale, color, thickness=1, lineType=None, bottomLeftOrigin=False):
    text = str(text)

    # 沒有中文時仍使用 OpenCV 原生繪字，維持數字與英文顯示效果
    if Image is None or ImageDraw is None or not _has_cjk(text):
        return _ORIG_CV2_PUTTEXT(img, text, org, fontFace, fontScale, color, thickness, lineType or cv2.LINE_AA, bottomLeftOrigin)

    h_img, w_img = img.shape[:2]
    x, y = int(org[0]), int(org[1])
    (text_w, text_h), _ = _measure_cjk_text(text, fontFace, fontScale, thickness)

    # OpenCV 的 org 是 baseline；PIL 的 y 是文字上緣，因此稍微往上修正
    y_top = y - text_h

    # 只轉換文字附近的小區域，不再每次把整張 1280x960 影像轉成 PIL。
    pad = 6
    x1 = max(0, x - pad)
    y1 = max(0, y_top - pad)
    x2 = min(w_img, x + text_w + pad)
    y2 = min(h_img, y_top + text_h + pad)

    if x1 >= x2 or y1 >= y2:
        return img

    roi = img[y1:y2, x1:x2]
    pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_roi)
    font = _get_cjk_font(fontScale, thickness)
    rgb_color = (int(color[2]), int(color[1]), int(color[0]))

    draw.text((x - x1, y_top - y1), text, font=font, fill=rgb_color)
    img[y1:y2, x1:x2] = cv2.cvtColor(np.array(pil_roi), cv2.COLOR_RGB2BGR)
    return img


def _get_chinese_text_size(text, fontFace, fontScale, thickness=1):
    text = str(text)

    if Image is None or ImageDraw is None or not _has_cjk(text):
        return _ORIG_CV2_GETTEXTSIZE(text, fontFace, fontScale, thickness)

    return _measure_cjk_text(text, fontFace, fontScale, thickness)

cv2.putText = _put_chinese_text
cv2.getTextSize = _get_chinese_text_size


# MediaPipe 版本相容處理：新版/部分版本可能沒有 mp.solutions 屬性
try:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
except Exception:
    try:
        mp_face_mesh = mp.solutions.face_mesh
    except Exception:
        mp_face_mesh = None
from collections import deque, Counter, defaultdict

# --- 環境初始化與路徑設定 ---
FILE = Path(__file__).resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))



# --- 顏色定義 (BGR 格式) ---
C_WHITE = (245, 245, 245)
C_BLACK = (20, 20, 20)
C_DARK = (35, 35, 35)
C_GRAY = (80, 80, 80)
C_LIGHT_GRAY = (180, 180, 180)
C_ORANGE = (0, 165, 255)   # Calibration / Pout
C_BLUE = (255, 150, 50)    # Ready
C_GREEN = (80, 220, 100)   # Holding
C_CYAN = (220, 220, 0)     # Resting / Chin lift
C_RED = (0, 0, 255)        # Warning
C_PURPLE = (180, 80, 255)
C_YELLOW = (0, 255, 255)


def draw_text_outline(frame, text, org, font, scale, color, thickness=2, outline_color=(0, 0, 0), outline_thickness=None):
    """
    在文字外加描邊，提升畫面可讀性。

    注意：
    本程式已將 cv2.putText 改成支援中文的 PIL 繪字版本。
    若用「更大的 thickness」畫黑色底字，中文字型大小也會跟著改變，
    會看起來像黑色文字與彩色文字重疊、錯位。
    因此這裡改用「同樣 thickness、不同座標偏移」來做描邊，
    避免中文文字出現重影。

    效能注意：中文文字若逐一呼叫 cv2.putText 9 次（8 個描邊位移 + 1 個本體），
    monkey-patch 過的中文版 cv2.putText 每次都要完整跑一輪 PIL 轉換
    （裁切 ROI → BGR轉RGB → 畫字 → 轉回BGR → 貼回），等於一個帶外框的
    中文標籤要做 9 次獨立 PIL 轉換；UI 一幀要畫幾十個標籤時，這是實測的
    主要效能瓶頸。因此中文文字改成「只裁切一次、只轉換一次 PIL」，
    9 層文字都畫在同一個 PIL 物件上才轉換貼回一次。
    """
    t = max(1, int(thickness))
    offsets = [
        (-2, -2), (-2, 0), (-2, 2),
        (0, -2),           (0, 2),
        (2, -2),  (2, 0),  (2, 2),
    ]
    x, y = int(org[0]), int(org[1])
    text_s = str(text)

    if Image is not None and ImageDraw is not None and _has_cjk(text_s):
        (text_w, text_h), _ = _measure_cjk_text(text_s, font, scale, t)
        y_top = y - text_h
        pad = 8  # 6(原本內距) + 2(描邊位移範圍)
        h_img, w_img = frame.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y_top - pad)
        x2 = min(w_img, x + text_w + pad)
        y2 = min(h_img, y_top + text_h + pad)
        if x1 >= x2 or y1 >= y2:
            return frame

        roi = frame[y1:y2, x1:x2]
        pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_roi)
        pil_font = _get_cjk_font(scale, t)
        rgb_outline = (int(outline_color[2]), int(outline_color[1]), int(outline_color[0]))
        rgb_color = (int(color[2]), int(color[1]), int(color[0]))

        for dx, dy in offsets:
            draw.text((x + dx - x1, y_top + dy - y1), text_s, font=pil_font, fill=rgb_outline)
        draw.text((x - x1, y_top - y1), text_s, font=pil_font, fill=rgb_color)

        frame[y1:y2, x1:x2] = cv2.cvtColor(np.array(pil_roi), cv2.COLOR_RGB2BGR)
        return frame

    for dx, dy in offsets:
        cv2.putText(frame, text_s, (x + dx, y + dy), font, scale, outline_color, t)
    cv2.putText(frame, text_s, (x, y), font, scale, color, t)
    return frame



# ==========================================================
# 臉部位置框線與位置檢查
# ==========================================================
def get_face_guide_box(frame):
    """
    取得臉部建議放置框線位置。
    只用於畫面提示與位置檢查，不會影響 ONNX 模型輸入。
    """
    h, w = frame.shape[:2]
    guide_w = int(w * 0.30)
    guide_h = int(h * 0.54)
    cx = int(w * 0.50)
    cy = int(h * 0.53)

    x1 = max(0, cx - guide_w // 2)
    y1 = max(0, cy - guide_h // 2)
    x2 = min(w - 1, cx + guide_w // 2)
    y2 = min(h - 1, cy + guide_h // 2)
    return x1, y1, x2, y2


def draw_face_position_guide(frame, face_points=None):
    """
    在畫面上畫出臉部位置框線，並檢查臉部是否位於框線中。
    face_points 為已映射到顯示畫面的 landmark 座標 list[(x, y)]。
    回傳：(是否在框線中, 中文提示文字)
    """
    x1, y1, x2, y2 = get_face_guide_box(frame)
    guide_w = x2 - x1
    guide_h = y2 - y1

    ok = False
    message = "請將臉放入框線中"

    if face_points:
        xs = [int(p[0]) for p in face_points]
        ys = [int(p[1]) for p in face_points]

        fx1, fy1 = max(0, min(xs)), max(0, min(ys))
        fx2, fy2 = min(frame.shape[1] - 1, max(xs)), min(frame.shape[0] - 1, max(ys))
        face_w = fx2 - fx1
        face_h = fy2 - fy1
        face_cx = (fx1 + fx2) / 2.0
        face_cy = (fy1 + fy2) / 2.0

        center_ok = (x1 <= face_cx <= x2) and (y1 <= face_cy <= y2)
        not_too_large = face_w <= guide_w * 1.18 and face_h <= guide_h * 1.18
        not_too_small = face_w >= guide_w * 0.32 and face_h >= guide_h * 0.32

        ok = center_ok and not_too_large and not_too_small

        if ok:
            message = "臉部位置正常"
        elif not center_ok:
            message = "請將臉移到框線中央"
        elif not not_too_large:
            message = "請稍微後退，讓臉完整進入框線"
        elif not not_too_small:
            message = "請靠近一點，將臉放在框線中"

    color = C_GREEN if ok else C_ORANGE

    # 半透明框線區域（只處理框線所在的小區域，不複製整張畫面）
    roi = frame[y1:y2, x1:x2]
    if roi.size > 0:
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (x2 - x1 - 1, y2 - y1 - 1), color, 2)
        cv2.addWeighted(overlay, 0.65, roi, 0.35, 0, roi)

    # 角落加粗，讓使用者更容易看到定位框
    corner = 48
    thickness = 4
    cv2.line(frame, (x1, y1), (x1 + corner, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner), color, thickness)
    cv2.line(frame, (x2, y1), (x2 - corner, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner), color, thickness)
    cv2.line(frame, (x1, y2), (x1 + corner, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - corner, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner), color, thickness)

    # 框線上方提示
    cv2.putText(
        frame,
        message,
        (x1, max(35, y1 - 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2
    )

    return ok, message


# --- 顯示畫面尺寸 ---
DISPLAY_W = 1280
DISPLAY_H = 960

# 只影響 OpenCV 顯示視窗大小，不影響模型輸入與推論精度
WINDOW_W = 960
WINDOW_H = 720


ALL_EXPRESSIONS = ["angry", "happy", "neutral", "sad", "surprised"]

EXPRESSION_ZH = {
    "angry": "生氣",
    "happy": "開心",
    "neutral": "自然",
    "sad": "難過",
    "surprised": "驚訝",
    "none": "無",
}

EXERCISE_ZH = {
    "Face_Lift": "臉部拉提訓練",
    "Double_Chin": "嘴部開合訓練",
    "Frown_Tighten": "皺眉緊眉訓練",
}

def zh_expr(name):
    return EXPRESSION_ZH.get(str(name).lower(), str(name))

def zh_exercise(name):
    return EXERCISE_ZH.get(str(name), str(name).replace("_", " "))


def normalize_class_name(cls_name):
    cls_name = str(cls_name).lower()
    if cls_name in ["surprise", "surprised"]:
        return "surprised"
    return cls_name


def resize_with_padding(frame, target_w=DISPLAY_W, target_h=DISPLAY_H, return_info=False):
    """
    將攝影機畫面等比例放進固定 1280x960 畫布，避免影像被拉伸變形。
    return_info=True 時會額外回傳 offset 與 scale，供 MediaPipe landmark 映射到顯示畫面。
    """
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized

    if return_info:
        return canvas, x, y, scale

    return canvas


def resize_for_window(frame, window_w=WINDOW_W, window_h=WINDOW_H):
    """
    只用於 cv2.imshow 的顯示縮放。
    注意：這不會回傳給 ONNX 或 MediaPipe，因此不會影響模型推論與判斷結果。
    """
    if frame is None:
        return frame
    return cv2.resize(frame, (window_w, window_h), interpolation=cv2.INTER_AREA)


class ONNXDetector:
    """
    CPU ONNX Runtime 推論器。
    支援：
    1. YOLOv9 常見輸出：(1, 9, N) 或 (1, N, 9)，9 = 4 bbox + 5 class scores
    2. YOLOv7 合併輸出：(1, 25200, 10)，10 = 4 bbox + objectness + 5 class scores

    注意：
    YOLOv7 的最終分數必須使用 objectness * class_score，
    並且 confidence bar 應顯示同一個最佳候選框的五類分數，
    不能對每一類各自從 25200 個候選框取最大值，否則會造成多類別同時接近 1.00。
    """
    def __init__(self, onnx_path, class_names, img_size=(224, 224)):
        self.onnx_path = str(onnx_path)
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)

        # img_size 是「實際影像內容」想要先縮放的尺寸，例如 640x480。
        # 但 ONNX 模型本身若是固定 640x640，最後送進 session 的 tensor 仍必須是 640x640。
        self.content_size = tuple(img_size)

        self.session = ort.InferenceSession(
            self.onnx_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        try:
            tensor_h = int(input_shape[2])
            tensor_w = int(input_shape[3])
        except Exception:
            tensor_w, tensor_h = self.content_size

        self.tensor_size = (tensor_w, tensor_h)

        print("已載入 ONNX 模型：", self.onnx_path)
        print("ONNX 執行提供者：", self.session.get_providers())
        print("ONNX 輸入形狀：", self.session.get_inputs()[0].shape)
        print("ONNX 輸出形狀：", [o.shape for o in self.session.get_outputs()])
        print("類別順序：", self.class_names)
        print("模型內容尺寸：", f"{self.content_size[0]}x{self.content_size[1]}")
        print("ONNX 張量尺寸：", f"{self.tensor_size[0]}x{self.tensor_size[1]}")

    def preprocess(self, frame):
        content_w, content_h = self.content_size
        tensor_w, tensor_h = self.tensor_size

        # 先把「影像內容」縮成 640x480，再轉黑白。
        # 若 ONNX 模型固定吃 640x640，則把 640x480 內容置中補黑邊成 640x640，避免維度錯誤。
        img = cv2.resize(frame, (content_w, content_h))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        if (content_w, content_h) != (tensor_w, tensor_h):
            canvas = np.zeros((tensor_h, tensor_w, 3), dtype=np.uint8)
            x0 = max(0, (tensor_w - content_w) // 2)
            y0 = max(0, (tensor_h - content_h) // 2)
            paste_w = min(content_w, tensor_w)
            paste_h = min(content_h, tensor_h)
            canvas[y0:y0 + paste_h, x0:x0 + paste_w] = img[:paste_h, :paste_w]
            img = canvas

        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        return img

    def _empty_result(self):
        return {
            "class": "none",
            "conf": 0.0,
            "all_conf": {normalize_class_name(name): 0.0 for name in self.class_names},
            "probs": np.zeros(self.num_classes, dtype=np.float32),
            "box": None
        }

    def _to_candidates(self, pred):
        """
        將常見 YOLO ONNX output 轉成 [num_candidates, channels]。
        """
        pred = np.asarray(pred)

        # batch 維度移除，例如 [1, 25200, 10] -> [25200, 10]
        while pred.ndim >= 3 and pred.shape[0] == 1:
            pred = pred[0]

        # YOLOv9 常見：[9, 6300]，轉成 [6300, 9]
        if pred.ndim == 2:
            if pred.shape[0] in (4 + self.num_classes, 5 + self.num_classes) and pred.shape[1] > pred.shape[0]:
                pred = pred.T
            return pred

        # 若仍是高維度 raw head，例如 [3, 80, 80, 10]，先攤平成 [19200, 10]
        # 注意：這不負責 anchor decode；若 export 已加 --grid，通常不會進到這裡。
        if pred.ndim >= 3 and pred.shape[-1] >= 4 + self.num_classes:
            pred = pred.reshape(-1, pred.shape[-1])
            return pred

        return pred

    def _parse_single_output(self, pred, conf_thres=0.25):
        pred = self._to_candidates(pred)

        if pred.ndim != 2:
            return self._empty_result()

        ch = pred.shape[1]
        expected_v9 = 4 + self.num_classes
        expected_v7 = 5 + self.num_classes

        if ch < expected_v9:
            return self._empty_result()

        boxes = pred[:, 0:4]

        # YOLOv7 merged output: [x, y, w, h, objectness, class1..class5]
        if ch >= expected_v7:
            obj_conf = pred[:, 4:5]
            class_scores = pred[:, 5:5 + self.num_classes]
            scores = obj_conf * class_scores
            output_type = "YOLOv7"

        # YOLOv9 output: [x, y, w, h, class1..class5]
        else:
            class_scores = pred[:, 4:4 + self.num_classes]
            scores = class_scores
            output_type = "YOLOv9"

        if scores.size == 0:
            return self._empty_result()

        # 清掉 NaN / Inf，避免異常值造成 argmax 錯誤
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        scores = np.clip(scores, 0.0, 1.0)

        # 找出「所有候選框、所有類別」中最高的分數
        best_box_idx, best_cls_id = np.unravel_index(np.argmax(scores), scores.shape)
        best_conf = float(scores[best_box_idx, best_cls_id])

        if best_conf < conf_thres:
            return self._empty_result()

        best_cls = normalize_class_name(self.class_names[int(best_cls_id)])
        best_box = boxes[best_box_idx]

        # confidence bar 只顯示同一個最佳候選框的五類分數
        # 不要對每一類各自取 max，否則 YOLOv7 會出現多類別同時 1.00。
        best_scores = scores[best_box_idx]
        all_conf = {}
        for i, name in enumerate(self.class_names):
            cls_name = normalize_class_name(name)
            all_conf[cls_name] = float(best_scores[i]) if i < len(best_scores) else 0.0

        probs = np.asarray([all_conf.get(normalize_class_name(name), 0.0) for name in self.class_names], dtype=np.float32)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, 0.0, 1.0)
        prob_sum = float(np.sum(probs))
        if prob_sum > 1e-9:
            probs = probs / prob_sum

        return {
            "class": best_cls,
            "conf": best_conf,
            "all_conf": all_conf,
            "probs": probs,
            "box": best_box,
            "output_type": output_type
        }

    def infer(self, frame, conf_thres=0.25):
        im = self.preprocess(frame)

        outputs = self.session.run(None, {self.input_name: im})
        outputs = [np.asarray(o) for o in outputs]

        # 常見情況：v7 merged 一個 output，v9 取第一個 output 即可。
        if len(outputs) == 1:
            return self._parse_single_output(outputs[0], conf_thres=conf_thres)

        # 若有多個 output，逐一解析，選出最高 confidence 的結果。
        # 這也可避免 v9 有輔助輸出時固定讀錯 output。
        best = self._empty_result()
        for out in outputs:
            result = self._parse_single_output(out, conf_thres=conf_thres)
            if float(result.get("conf", 0.0)) > float(best.get("conf", 0.0)):
                best = result

        return best


# ==========================================================
# UI 小工具
# ==========================================================
def draw_filled_rect_with_alpha(frame, pt1, pt2, color, alpha=0.75):
    """
    半透明填色矩形。
    只對矩形所在的小區域做 copy + addWeighted，不複製整張畫面，
    這個函式在每一幀會被呼叫十幾次，若每次都複製整張 1280x960 畫面會嚴重拖慢 FPS。
    """
    h, w = frame.shape[:2]
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    overlay = np.empty_like(roi)
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def draw_landmark_dots(frame, lm, color=C_LIGHT_GRAY, src_w=None, src_h=None, scale=1.0, offset=(0, 0)):
    """
    以 numpy 向量化方式一次畫出全部 landmark 小點，
    取代逐點呼叫 cv2.circle（每幀 468 次 Python/OpenCV 呼叫太慢，是卡頓主因之一）。
    src_w/src_h：landmark 正規化座標所依據的來源畫面寬高（預設為 frame 自身)。
    scale/offset：座標從來源畫面映射到 frame 顯示畫面的縮放與位移（對應 resize_with_padding 的 offset/scale）。
    """
    h, w = frame.shape[:2]
    n = len(lm)
    if n == 0:
        return

    sw = src_w if src_w is not None else w
    sh = src_h if src_h is not None else h
    off_x, off_y = offset

    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    for i, pt in enumerate(lm):
        xs[i] = pt.x
        ys[i] = pt.y

    cx = np.clip((xs * sw * scale + off_x).astype(np.int32), 0, w - 1)
    cy = np.clip((ys * sh * scale + off_y).astype(np.int32), 0, h - 1)

    frame[cy, cx] = color
    frame[cy, np.clip(cx + 1, 0, w - 1)] = color
    frame[np.clip(cy + 1, 0, h - 1), cx] = color

def key_pressed(key, *chars):
    """
    支援小寫、大寫，以及 cv2.waitKey / cv2.waitKeyEx 的不同回傳格式。
    """
    if key == -1:
        return False

    key_low = key & 0xFF

    for ch in chars:
        if key == ord(ch) or key_low == ord(ch):
            return True

        if ch.isalpha():
            if key == ord(ch.upper()) or key_low == ord(ch.upper()):
                return True

    return False

def put_fit_text(frame, text, org, font, max_scale, min_scale, color, thickness, max_width):
    """
    自動縮小文字，避免文字超出卡片範圍。
    """
    scale = max_scale

    while scale >= min_scale:
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        if text_size[0] <= max_width:
            break
        scale -= 0.05

    cv2.putText(
        frame,
        text,
        org,
        font,
        scale,
        color,
        thickness
    )


def draw_center_text(frame, text, y, scale, color, thickness=2):
    h, w, _ = frame.shape
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)[0]
    x = int((w - text_size[0]) / 2)

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        color,
        thickness
    )




def draw_top_bar(frame, title, subtitle="", right_text=""):
    """
    表情訓練畫面上方資訊列。
    原本此函式存在於 Facial_expression_training.py，整合版需要保留，
    否則進入臉部表情訓練時會發生 NameError。
    """
    h, w, _ = frame.shape

    draw_filled_rect_with_alpha(
        frame,
        (0, 0),
        (w, 112),
        C_BLACK,
        0.75
    )

    put_fit_text(
        frame,
        title,
        (30, 48),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        0.55,
        C_WHITE,
        2,
        max(420, w - 520)
    )

    if subtitle:
        put_fit_text(
            frame,
            subtitle,
            (32, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            0.38,
            C_LIGHT_GRAY,
            1,
            max(420, w - 520)
        )

    if right_text:
        put_fit_text(
            frame,
            right_text,
            (w - 520, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            0.35,
            C_LIGHT_GRAY,
            1,
            490
        )

def draw_card(frame, x, y, w, h, title, subtitle, key_text, color):
    """
    主畫面選單卡片。
    """
    # shadow
    draw_filled_rect_with_alpha(
        frame,
        (x + 8, y + 8),
        (x + w + 8, y + h + 8),
        (0, 0, 0),
        0.35
    )

    # card body
    draw_filled_rect_with_alpha(
        frame,
        (x, y),
        (x + w, y + h),
        (38, 38, 38),
        0.92
    )

    # left color bar
    cv2.rectangle(frame, (x, y), (x + 10, y + h), color, -1)

    # key circle
    cv2.circle(frame, (x + 48, y + int(h / 2)), 27, color, -1)

    cv2.putText(
        frame,
        key_text,
        (x + 39, y + int(h / 2) + 9),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        C_BLACK,
        2
    )

    # title
    put_fit_text(
        frame,
        title,
        (x + 95, y + 42),
        cv2.FONT_HERSHEY_DUPLEX,
        0.85,
        0.55,
        C_WHITE,
        2,
        w - 130
    )

    # subtitle
    put_fit_text(
        frame,
        subtitle,
        (x + 95, y + 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        0.35,
        C_LIGHT_GRAY,
        1,
        w - 130
    )


def draw_main_menu(frame):
    """
    主畫面：整合臉部表情訓練與臉部運動訓練功能。
    """
    h, w, _ = frame.shape

    draw_filled_rect_with_alpha(frame, (0, 0), (w, h), (15, 15, 15), 0.72)

    draw_center_text(
        frame,
        "臉部運動訓練系統",
        100,
        1.3,
        C_WHITE,
        2
    )

    draw_center_text(
        frame,
        "請選擇要開始的訓練項目",
        145,
        0.75,
        C_LIGHT_GRAY,
        1
    )

    cv2.putText(
        frame,
        "主選單",
        (w - 250, 60),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        C_CYAN,
        2
    )

    card_w = min(920, w - 180)
    card_h = 105
    x = int((w - card_w) / 2)

    y1 = 205
    y2 = y1 + 118
    y3 = y2 + 118
    y4 = y3 + 118

    draw_card(
        frame,
        x,
        y1,
        card_w,
        card_h,
        "臉部表情訓練",
        "五類表情循環訓練：生氣、開心、自然、難過與驚訝",
        "1",
        C_GREEN
    )

    draw_card(
        frame,
        x,
        y2,
        card_w,
        card_h,
        "臉部拉提訓練",
        "微笑上提訓練：結合表情辨識與嘴角位置變化判斷",
        "2",
        C_CYAN
    )

    draw_card(
        frame,
        x,
        y3,
        card_w,
        card_h,
        "嘴部開合訓練",
        "張嘴訓練：以上下嘴唇距離與表情狀態進行即時回饋",
        "3",
        C_ORANGE
    )

    draw_card(
        frame,
        x,
        y4,
        card_w,
        card_h,
        "皺眉緊眉訓練",
        "皺眉訓練：以眉頭水平靠攏與垂直下壓幅度搭配表情狀態判斷",
        "4",
        C_RED
    )

    draw_filled_rect_with_alpha(
        frame,
        (0, h - 70),
        (w, h),
        C_BLACK,
        0.7
    )

    cv2.putText(
        frame,
        "按 1 / 2 / 3 / 4 選擇      |      Q：離開",
        (55, h - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        C_WHITE,
        1
    )



# ==========================================================
# 臉部表情訓練功能（由 Facial_expression_training.py 整合）
# ==========================================================
# ==========================================================
# 表情資訊
# ==========================================================
EXPRESSION_INFO = {
    "neutral": {
        "title": "自然",
        "face": ":|",
        "color": C_CYAN,
        "tips": [
            "放鬆眉毛",
            "放鬆嘴巴",
            "正視攝影機"
        ]
    },
    "happy": {
        "title": "開心",
        "face": ":)",
        "color": C_GREEN,
        "tips": [
            "上提兩側嘴角",
            "輕輕微笑",
            "眼睛保持自然"
        ]
    },
    "sad": {
        "title": "難過",
        "face": ":(",
        "color": C_BLUE,
        "tips": [
            "下拉嘴角",
            "放鬆眼睛",
            "臉部稍微向下用力"
        ]
    },
    "angry": {
        "title": "生氣",
        "face": ">:|",
        "color": C_RED,
        "tips": [
            "眉毛向中間靠近",
            "眼神集中",
            "嘴巴保持緊實"
        ]
    },
    "surprised": {
        "title": "驚訝",
        "face": ":O",
        "color": C_ORANGE,
        "tips": [
            "睜大眼睛",
            "嘴巴微微張開",
            "抬高眉毛"
        ]
    }
}

ALL_EXPRESSIONS = ["angry", "happy", "neutral", "sad", "surprised"]
TRAIN_EXPRESSIONS = ALL_EXPRESSIONS

EXPRESSION_ZH = {
    "angry": "生氣",
    "happy": "開心",
    "neutral": "自然",
    "sad": "難過",
    "surprised": "驚訝",
    "none": "無",
}

STATUS_ZH = {
    "READY": "準備",
    "INSTRUCTION": "說明",
    "TRY IT": "開始嘗試",
    "HOLDING": "維持中",
    "ADJUST": "請調整",
    "REST": "休息",
    "FINISHED": "完成",
    "臉部對齊": "臉部對齊",
    "FREE": "自由偵測",
}

def zh_expr(name):
    return EXPRESSION_ZH.get(str(name).lower(), str(name))

def zh_status(name):
    return STATUS_ZH.get(str(name), str(name))


def normalize_class_name(cls_name):
    cls_name = str(cls_name).lower()

    if cls_name in ["surprise", "surprised"]:
        return "surprised"

    return cls_name




# ==========================================================
# 臉部擺正檢查：MediaPipe FaceMesh
# ==========================================================
class FaceAlignmentChecker:
    def __init__(self):
        self.enabled = False
        self.face_mesh = None
        self.error_msg = ""

        try:
            face_mesh_module = None

            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                face_mesh_module = mp.solutions.face_mesh
            else:
                from mediapipe.python.solutions import face_mesh as face_mesh_module

            self.face_mesh = face_mesh_module.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )

            self.enabled = True
            print("MediaPipe FaceMesh loaded successfully")

        except Exception as e:
            self.enabled = False
            self.face_mesh = None
            self.error_msg = str(e)
            print("MediaPipe FaceMesh disabled:", e)

        self.roll_threshold_deg = 8.0
        self.yaw_ratio_threshold = 0.75

    def check(self, frame):
        if not self.enabled or self.face_mesh is None:
            return False, "MediaPipe 關閉"

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            draw_face_position_guide(frame, None)
            return False, "請將臉放入框線中"

        lm = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        face_points = [(int(pt.x * w), int(pt.y * h)) for pt in lm]
        position_ok, position_msg = draw_face_position_guide(frame, face_points)

        # 顯示全臉灰色偵測點，確認 MediaPipe 正常運作
        draw_landmark_dots(frame, lm, C_LIGHT_GRAY)

        left_eye = lm[33]
        right_eye = lm[263]
        nose = lm[1]

        lx, ly = int(left_eye.x * w), int(left_eye.y * h)
        rx, ry = int(right_eye.x * w), int(right_eye.y * h)
        nx, ny = int(nose.x * w), int(nose.y * h)

        # 畫出重點輔助點
        cv2.circle(frame, (lx, ly), 7, C_CYAN, -1)
        cv2.circle(frame, (rx, ry), 7, C_CYAN, -1)
        cv2.circle(frame, (nx, ny), 8, C_ORANGE, -1)

        cv2.circle(frame, (lx, ly), 10, C_WHITE, 2)
        cv2.circle(frame, (rx, ry), 10, C_WHITE, 2)
        cv2.circle(frame, (nx, ny), 11, C_WHITE, 2)

        cv2.putText(frame, "左眼", (lx + 8, ly - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1)
        cv2.putText(frame, "右眼", (rx + 8, ry - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1)
        cv2.putText(frame, "鼻子", (nx + 8, ny - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1)

        cv2.line(frame, (lx, ly), (rx, ry), C_CYAN, 2)

        # Roll：左右眼連線角度，判斷頭有沒有歪
        roll_deg = math.degrees(math.atan2(ry - ly, rx - lx))
        is_level = abs(roll_deg) <= self.roll_threshold_deg

        # Yaw：鼻子到左右眼距離比例，判斷是否正對鏡頭
        dist_left = abs(nose.x - left_eye.x)
        dist_right = abs(nose.x - right_eye.x)

        if max(dist_left, dist_right) == 0:
            return False, "臉部對齊失敗"

        yaw_ratio = min(dist_left, dist_right) / max(dist_left, dist_right)
        is_centered = yaw_ratio >= self.yaw_ratio_threshold

        if not position_ok:
            return False, position_msg

        if not is_level:
            return False, "請保持臉部水平，不要歪頭"

        if not is_centered:
            return False, "請正對攝影機"

        return True, "臉部已對齊"


def draw_alignment_status(frame, is_aligned, message):
    h, w, _ = frame.shape

    color = C_GREEN if is_aligned else C_RED
    label = "臉部正常" if is_aligned else "臉部對齊"

    x, y = 35, 230
    box_w, box_h = 500, 54

    draw_filled_rect_with_alpha(
        frame,
        (x, y),
        (x + box_w, y + box_h),
        C_BLACK,
        0.65
    )

    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), color, 2)

    cv2.putText(
        frame,
        label,
        (x + 15, y + 34),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        color,
        2
    )

    cv2.putText(
        frame,
        message,
        (x + 150, y + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        C_WHITE,
        1
    )



# ==========================================================
# 表情指引圖
# ==========================================================
def draw_expression_face(frame, center, expression, scale=1.0):
    x, y = center
    r = int(55 * scale)

    info = EXPRESSION_INFO.get(expression, EXPRESSION_INFO["neutral"])
    color = info["color"]

    cv2.circle(frame, (x, y), r, (120, 120, 120), 2)

    eye_y = y - int(18 * scale)
    eye_dx = int(20 * scale)

    if expression == "angry":
        cv2.line(frame, (x - eye_dx - 10, eye_y - 8), (x - eye_dx + 8, eye_y - 2), C_WHITE, 2)
        cv2.line(frame, (x + eye_dx - 8, eye_y - 2), (x + eye_dx + 10, eye_y - 8), C_WHITE, 2)

    elif expression == "surprised":
        cv2.circle(frame, (x - eye_dx, eye_y), 6, C_WHITE, 2)
        cv2.circle(frame, (x + eye_dx, eye_y), 6, C_WHITE, 2)
        cv2.line(frame, (x - eye_dx - 10, eye_y - 15), (x - eye_dx + 10, eye_y - 18), C_WHITE, 2)
        cv2.line(frame, (x + eye_dx - 10, eye_y - 18), (x + eye_dx + 10, eye_y - 15), C_WHITE, 2)

    else:
        cv2.circle(frame, (x - eye_dx, eye_y), 4, C_WHITE, -1)
        cv2.circle(frame, (x + eye_dx, eye_y), 4, C_WHITE, -1)

    mouth_y = y + int(25 * scale)

    if expression == "happy":
        pts = np.array([
            [x - 28, mouth_y - 5],
            [x, mouth_y + 18],
            [x + 28, mouth_y - 5]
        ], np.int32)
        cv2.polylines(frame, [pts], False, color, 3)

    elif expression == "sad":
        pts = np.array([
            [x - 28, mouth_y + 12],
            [x, mouth_y - 10],
            [x + 28, mouth_y + 12]
        ], np.int32)
        cv2.polylines(frame, [pts], False, color, 3)

    elif expression == "angry":
        cv2.line(frame, (x - 25, mouth_y + 5), (x + 25, mouth_y + 5), color, 3)

    elif expression == "surprised":
        cv2.ellipse(frame, (x, mouth_y), (16, 24), 0, 0, 360, color, 3)

    else:
        cv2.line(frame, (x - 25, mouth_y), (x + 25, mouth_y), color, 3)


def draw_guide_panel(
    frame,
    target,
    status,
    confidence,
    hold_remaining,
    hold_duration,
    is_target_reached,
    y_start=145
):
    h, w, _ = frame.shape

    # 小型導引提示框，放在右上方
    panel_w = 260
    panel_h = 290
    x = w - panel_w - 28
    y = y_start

    info = EXPRESSION_INFO.get(target, EXPRESSION_INFO["neutral"])
    color = info["color"]

    draw_filled_rect_with_alpha(
        frame,
        (x, y),
        (x + panel_w, y + panel_h),
        (25, 25, 25),
        0.88
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + panel_w, y + panel_h),
        color,
        2
    )

    cv2.putText(
        frame,
        "指引",
        (x + 18, y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        C_WHITE,
        2
    )

    # 小型表情圖示
    draw_expression_face(
        frame,
        (x + 70, y + 100),
        target,
        0.65
    )

    # 表情名稱
    cv2.putText(
        frame,
        info["title"].upper(),
        (x + 130, y + 95),
        cv2.FONT_HERSHEY_DUPLEX,
        0.6,
        color,
        2
    )

    # 小提示文字
    tips_y = y + 145

    for i, tip in enumerate(info["tips"]):
        put_fit_text(
            frame,
            f"{i + 1}. {tip}",
            (x + 18, tips_y + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            0.32,
            C_WHITE,
            1,
            panel_w - 35
        )

    # Holding 秒數提示
    if status == "HOLDING":
        remain = max(0.0, hold_remaining)

        cv2.rectangle(
            frame,
            (x + 18, y + panel_h - 48),
            (x + panel_w - 18, y + panel_h - 18),
            C_DARK,
            -1
        )

        cv2.putText(
            frame,
            f"Hold: {math.ceil(remain)}s",
            (x + 32, y + panel_h - 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            C_GREEN if is_target_reached else C_RED,
            2
        )

# ==========================================================
# 五類表情信心分數橫條圖
# ==========================================================
def draw_confidence_bars(frame, confs, target=None, y_end=None):
    """
    小型五類表情信心分數橫條圖，顯示在畫面右下方。
    y_end: 面板底部的 y 座標，預設為 h-24。
    """
    h, w, _ = frame.shape

    panel_w = 300
    panel_h = 175
    x = w - panel_w - 28
    _y_end = y_end if y_end is not None else (h - 24)
    y = _y_end - panel_h

    draw_filled_rect_with_alpha(
        frame,
        (x, y),
        (x + panel_w, y + panel_h),
        C_BLACK,
        0.72
    )

    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), C_GRAY, 2)

    cv2.putText(
        frame,
        "信心分數",
        (x + 18, y + 28),
        cv2.FONT_HERSHEY_DUPLEX,
        0.58,
        C_WHITE,
        1
    )

    bar_x = x + 105
    bar_y = y + 45
    bar_w = 145
    bar_h = 12
    gap = 24

    for i, exp in enumerate(ALL_EXPRESSIONS):
        info = EXPRESSION_INFO[exp]
        color = info["color"]

        yy = bar_y + i * gap
        conf = float(confs.get(exp, 0.0))
        conf = max(0.0, min(conf, 1.0))

        label_color = color if exp == target else C_LIGHT_GRAY

        cv2.putText(
            frame,
            zh_expr(exp),
            (x + 16, yy + 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            label_color,
            1
        )

        cv2.rectangle(
            frame,
            (bar_x, yy),
            (bar_x + bar_w, yy + bar_h),
            C_GRAY,
            -1
        )

        fill_w = int(conf * bar_w)

        cv2.rectangle(
            frame,
            (bar_x, yy),
            (bar_x + fill_w, yy + bar_h),
            color,
            -1
        )

        cv2.putText(
            frame,
            f"{conf:.2f}",
            (bar_x + bar_w + 8, yy + 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            C_WHITE,
            1
        )


# ==========================================================
# 訓練系統：改為 expression_exercises.py 的 FSM 計分邏輯
# ==========================================================
QUALITY_TIERS = [
    (0.85, "完美！", C_GREEN),
    (0.70, "很好！", C_CYAN),
    (0.55, "差不多", C_ORANGE),
    (0.00, "繼續加油", C_RED),
]


class ExpressionTrainingSystem:
    """
    表情訓練狀態機。

    邏輯參考 expression_exercises.py：
    1. 依序訓練五類表情：angry, happy, neutral, sad, surprised
    2. 每個表情開始前有 PREP 倒數
    3. predicted_class == target 且 target confidence >= 0.60 時才進入 HOLD
    4. 連續維持 HOLD_DURATION 秒後，計為一次完成
    5. 每次完成時，以維持期間的平均 target confidence 作為品質分數
    """
    def __init__(self, rounds=2):
        self.rounds = int(rounds)
        self.sequence = ALL_EXPRESSIONS * self.rounds
        self.total_reps = len(self.sequence)

        self.current_idx = 0
        self.reps_done = 0

        self.state = "IDLE"

        self.prep_duration = 2.0
        self.prep_start_time = None
        self.prep_remaining = self.prep_duration

        self.hold_duration = 2.0
        self.hold_threshold = 0.60
        self.hold_start_time = None
        self.hold_elapsed = 0.0
        self.hold_remaining = self.hold_duration
        self.hold_confs = deque(maxlen=240)

        self.result_duration = 1.5
        self.result_start_time = None
        self.result_score = 0.0
        self.result_label = ""

        self.feedback = "按空白鍵開始表情訓練"

        self.probs_ema = None
        self.ema_alpha = 0.35

        self.debug_enabled = False

        self.stats = defaultdict(lambda: {
            "success": 0,
            "best_conf": 0.0,
            "total_score": 0.0,
            "count": 0
        })
        self.scores = []

    def reset(self):
        rounds = self.rounds
        self.__init__(rounds=rounds)

    def get_current_target(self):
        if self.current_idx >= len(self.sequence):
            return self.sequence[-1]
        return self.sequence[self.current_idx]

    def get_target_idx(self):
        return ALL_EXPRESSIONS.index(self.get_current_target())

    def start(self):
        if self.state in ["IDLE", "DONE"]:
            self.state = "PREP"
            self.prep_start_time = time.time()
            self.prep_remaining = self.prep_duration
            self.feedback = f"準備做：{zh_expr(self.get_current_target())}"

    def _quality_label(self, score):
        for thresh, label, _ in QUALITY_TIERS:
            if score >= thresh:
                return f"{label} 品質 {score * 100:.0f}%"
        return f"品質 {score * 100:.0f}%"

    def _quality_color(self, score):
        for thresh, _, color in QUALITY_TIERS:
            if score >= thresh:
                return color
        return C_RED

    def _advance(self):
        self.current_idx += 1

        if self.current_idx >= len(self.sequence):
            self.state = "DONE"
            avg = float(np.mean(self.scores)) if self.scores else 0.0
            self.feedback = f"全部完成！平均品質 {avg * 100:.0f}%"
            return

        self.state = "PREP"
        self.prep_start_time = time.time()
        self.prep_remaining = self.prep_duration
        self.hold_start_time = None
        self.hold_elapsed = 0.0
        self.hold_remaining = self.hold_duration
        self.hold_confs.clear()
        self.feedback = f"準備做：{zh_expr(self.get_current_target())}"

    def update_smoothed_probs(self, probs):
        probs = np.asarray(probs, dtype=np.float32)

        if self.probs_ema is None:
            self.probs_ema = probs.copy()
        else:
            self.probs_ema = self.ema_alpha * probs + (1.0 - self.ema_alpha) * self.probs_ema

        return self.probs_ema

    def analyze(self, yolo_results, is_aligned):
        raw_probs = yolo_results.get("probs", None)

        if raw_probs is None:
            raw_probs = np.array([yolo_results.get("all_conf", {}).get(e, 0.0) for e in ALL_EXPRESSIONS], dtype=np.float32)

        probs = self.update_smoothed_probs(raw_probs)

        pred_idx = int(np.argmax(probs))
        pred_class = ALL_EXPRESSIONS[pred_idx]
        pred_conf = float(probs[pred_idx])

        target = self.get_current_target()
        target_idx = self.get_target_idx()
        target_conf = float(probs[target_idx])

        now = time.time()

        is_target_reached = (
            is_aligned
            and pred_idx == target_idx
            and target_conf >= self.hold_threshold
        )

        status = self.state
        color = C_WHITE

        # 臉部未對齊時，除了 IDLE / DONE 以外都暫停判斷
        if not is_aligned and self.state not in ["IDLE", "DONE"]:
            self.feedback = "請先將臉部擺正並放入框線中"
            status = "ALIGN"
            color = C_RED

            debug = self._make_debug(target, pred_class, pred_conf, target_conf, is_target_reached, probs)
            return status, self.feedback, color, debug

        if self.state == "IDLE":
            self.feedback = "按空白鍵開始表情訓練"
            color = C_LIGHT_GRAY

        elif self.state == "PREP":
            if self.prep_start_time is None:
                self.prep_start_time = now

            elapsed = now - self.prep_start_time
            self.prep_remaining = max(0.0, self.prep_duration - elapsed)
            self.feedback = f"準備：{zh_expr(target)}，剩 {self.prep_remaining:.1f} 秒"
            color = C_ORANGE

            if elapsed >= self.prep_duration:
                self.state = "DETECT"
                self.feedback = f"請做出：{zh_expr(target)}"
                color = EXPRESSION_INFO[target]["color"]

        elif self.state == "DETECT":
            color = EXPRESSION_INFO[target]["color"]

            if is_target_reached:
                self.state = "HOLD"
                self.hold_start_time = now
                self.hold_elapsed = 0.0
                self.hold_remaining = self.hold_duration
                self.hold_confs.clear()
                self.hold_confs.append(target_conf)
                self.feedback = "很好，請維持！"
                color = C_GREEN
            else:
                self.feedback = (
                    f"請做出：{zh_expr(target)} "
                    f"（目標信心需 ≥ {self.hold_threshold:.0%}，目前 {target_conf:.0%}）"
                )

        elif self.state == "HOLD":
            color = C_GREEN if is_target_reached else C_RED

            if is_target_reached:
                if self.hold_start_time is None:
                    self.hold_start_time = now

                self.hold_elapsed = now - self.hold_start_time
                self.hold_remaining = max(0.0, self.hold_duration - self.hold_elapsed)
                self.hold_confs.append(target_conf)
                self.feedback = f"維持！剩 {self.hold_remaining:.1f} 秒"

                if self.hold_elapsed >= self.hold_duration:
                    score = float(np.mean(self.hold_confs)) if self.hold_confs else target_conf
                    self.result_score = score
                    self.result_label = self._quality_label(score)
                    self.scores.append(score)

                    self.stats[target]["success"] += 1
                    self.stats[target]["best_conf"] = max(self.stats[target]["best_conf"], score)
                    self.stats[target]["total_score"] += score
                    self.stats[target]["count"] += 1

                    self.reps_done += 1
                    self.state = "RESULT"
                    self.result_start_time = now
                    self.feedback = self.result_label
                    color = self._quality_color(score)
            else:
                self.state = "DETECT"
                self.hold_start_time = None
                self.hold_elapsed = 0.0
                self.hold_remaining = self.hold_duration
                self.hold_confs.clear()
                self.feedback = f"表情中斷，請重新做出：{zh_expr(target)}"
                color = C_RED

        elif self.state == "RESULT":
            color = self._quality_color(self.result_score)
            self.feedback = self.result_label

            if self.result_start_time is None:
                self.result_start_time = now

            if now - self.result_start_time >= self.result_duration:
                self.result_start_time = None
                self._advance()

        elif self.state == "DONE":
            avg = float(np.mean(self.scores)) if self.scores else 0.0
            self.feedback = f"全部完成！平均品質 {avg * 100:.0f}%"
            color = C_YELLOW

        debug = self._make_debug(target, pred_class, pred_conf, target_conf, is_target_reached, probs)
        return self.state, self.feedback, color, debug

    def _make_debug(self, target, pred_class, pred_conf, target_conf, is_target_reached, probs):
        return {
            "target": target,
            "target_idx": self.get_target_idx(),
            "current_class": pred_class,
            "current_conf": pred_conf,
            "target_conf": target_conf,
            "is_target_reached": is_target_reached,
            "hold_remaining": self.hold_remaining,
            "hold_elapsed": self.hold_elapsed,
            "hold_progress": min(1.0, self.hold_elapsed / self.hold_duration) if self.hold_duration > 0 else 0.0,
            "prep_remaining": self.prep_remaining,
            "prep_progress": min(1.0, (self.prep_duration - self.prep_remaining) / self.prep_duration) if self.prep_duration > 0 else 0.0,
            "probs": probs,
            "result_score": self.result_score,
            "result_label": self.result_label
        }



# ==========================================================
# 訓練畫面 UI
# ==========================================================
def _progress_bar(frame, x, y, w, h, progress, color):
    progress = max(0.0, min(1.0, float(progress)))
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_GRAY, -1)
    cv2.rectangle(frame, (x, y), (x + int(w * progress), y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_LIGHT_GRAY, 1)


def draw_training_ui(
    frame,
    trainer,
    status,
    feedback,
    color,
    debug,
    is_aligned,
    align_msg,
    confs
):
    h, w, _ = frame.shape

    TOP_H    = 190
    BOTTOM_H = 82
    BOT_Y    = h - BOTTOM_H

    target            = debug["target"]
    current_class     = debug["current_class"]
    current_conf      = debug["current_conf"]
    target_conf       = debug["target_conf"]
    is_target_reached = debug["is_target_reached"]

    _expr_info = EXPRESSION_INFO.get(target, EXPRESSION_INFO["neutral"])
    _tips      = _expr_info.get("tips", [])

    # ── 狀態文字決定 ────────────────────────────────────────────
    status_lbl = {
        "IDLE":   "等待開始",
        "PREP":   "準備倒數",
        "DETECT": "偵測中",
        "HOLD":   "動作達標" if is_target_reached else "請調整",
        "RESULT": "完成一次",
        "DONE":   "全部完成",
        "ALIGN":  "調整位置",
    }.get(status, status)

    main_text  = feedback
    sub_text   = ""
    expr_color = color

    if not is_aligned or status == "ALIGN":
        expr_color = C_RED
        status_lbl = "調整位置"
        main_text  = align_msg or "請正對攝影機，將臉放入框線中"
        sub_text   = "請確保臉部出現在畫面中央框線內"
    elif status == "IDLE":
        expr_color = C_LIGHT_GRAY
        main_text  = f"按  空白鍵  開始訓練｜目標表情：{zh_expr(target)}"
        sub_text   = "M：主選單    Q：離開"
    elif status == "PREP":
        expr_color = C_ORANGE
        prep_s     = math.ceil(debug.get("prep_remaining", 3.0))
        main_text  = f"倒數  {prep_s}  秒後開始  ─  準備做出  {zh_expr(target)}  表情"
        sub_text   = _tips[0] if _tips else ""
    elif status == "DETECT":
        expr_color = C_BLUE
        main_text  = f"請做出  {zh_expr(target)}  表情"
        sub_text   = feedback
    elif status == "HOLD":
        if is_target_reached:
            expr_color = C_GREEN
            main_text  = f"很棒！請維持  {zh_expr(target)}  表情，分數持續累積"
            sub_text   = "  |  ".join(_tips[:2]) if _tips else ""
        else:
            expr_color = C_RED
            main_text  = f"請調整！目前偵測到  {zh_expr(current_class)}，需要  {zh_expr(target)}"
            sub_text   = feedback
    elif status == "RESULT":
        score      = debug.get("result_score", 0)
        expr_color = C_GREEN if score >= 0.6 else C_YELLOW
        main_text  = (f"太棒了！本次品質：{score * 100:.0f}%" if score >= 0.6
                      else f"繼續加油！本次品質：{score * 100:.0f}%")
        sub_text   = "下一組即將開始..."
    elif status == "DONE":
        expr_color = C_BLUE
        main_text  = "所有回合完成！"
        sub_text   = "按  M  返回主選單"

    # ══════════════════════════════════════════════════════════
    # 區域 1：頂部指令區
    # ══════════════════════════════════════════════════════════
    draw_filled_rect_with_alpha(frame, (0, 0), (w, TOP_H), (8, 10, 18), 0.93)

    # 第一行：狀態 pill + 目標表情 + 進度計數
    _ptw = cv2.getTextSize(status_lbl, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0][0]
    _px1, _px2 = 16, 16 + _ptw + 30
    cv2.rectangle(frame, (_px1, 10), (_px2, 68), expr_color, -1)
    draw_text_outline(frame, status_lbl, (_px1 + 13, 57),
                      cv2.FONT_HERSHEY_DUPLEX, 1.0, (8, 10, 18), 2)
    draw_text_outline(frame, f"目標：{zh_expr(target)}", (_px2 + 18, 60),
                      cv2.FONT_HERSHEY_DUPLEX, 1.05, _expr_info["color"], 2)
    _prog_text = f"第  {trainer.reps_done} / {trainer.total_reps}  次"
    _prog_tw   = cv2.getTextSize(_prog_text, cv2.FONT_HERSHEY_DUPLEX, 0.96, 2)[0][0]
    draw_text_outline(frame, _prog_text, (w - _prog_tw - 18, 60),
                      cv2.FONT_HERSHEY_DUPLEX, 0.96, C_WHITE, 2)

    cv2.line(frame, (16, 74), (w - 16, 74), (55, 55, 68), 1)

    # 第二行：主要指引
    draw_text_outline(frame, main_text, (18, 128),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.92, expr_color, 2)

    # 第三行：補充說明 + 快捷鍵提示
    if sub_text:
        draw_text_outline(frame, sub_text, (18, 166),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_LIGHT_GRAY, 1)
    draw_text_outline(frame, "空白鍵：開始   R：重置   M：主選單   Q：離開",
                      (w - 434, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 100), 1)

    cv2.line(frame, (0, TOP_H), (w, TOP_H), expr_color, 3)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), expr_color, 6)

    # ══════════════════════════════════════════════════════════
    # 浮動計時圓圈（相機區左側，不干擾右側指引面板）
    # ══════════════════════════════════════════════════════════
    _TCX = w - 420
    _TCY = TOP_H + 130
    _RC  = 68

    _gx1, _gy1 = max(0, _TCX - _RC - 16), max(0, _TCY - _RC - 16)
    _gx2, _gy2 = min(frame.shape[1], _TCX + _RC + 16), min(frame.shape[0], _TCY + _RC + 16)
    _groi = frame[_gy1:_gy2, _gx1:_gx2]
    _ov = _groi.copy()
    cv2.circle(_ov, (_TCX - _gx1, _TCY - _gy1), _RC + 16, (6, 8, 16), -1)
    cv2.addWeighted(_ov, 0.80, _groi, 0.20, 0, _groi)
    cv2.circle(frame, (_TCX, _TCY), _RC, (55, 55, 55), 5)

    _circle_text = ""
    _circ_color  = expr_color

    if trainer.state == "PREP":
        prog = debug.get("prep_progress", 0.0)
        cv2.ellipse(frame, (_TCX, _TCY), (_RC, _RC), 0, -90, -90 + int(prog * 360), C_ORANGE, 6)
        _circle_text = f"{math.ceil(debug.get('prep_remaining', 3.0))}s"
        _circ_color  = C_ORANGE
    elif trainer.state == "HOLD":
        prog    = debug.get("hold_progress", 0.0)
        _gc     = C_GREEN if is_target_reached else C_RED
        _gr2    = int(_RC + 10 * math.sin(time.time() * 8))
        cv2.circle(frame, (_TCX, _TCY), _gr2, _gc, 2)
        cv2.ellipse(frame, (_TCX, _TCY), (_RC, _RC), 0, -90, -90 + int(prog * 360), _gc, 6)
        _circle_text = f"{math.ceil(debug.get('hold_remaining', 3.0))}s"
        _circ_color  = _gc
    elif trainer.state == "RESULT":
        score   = debug.get("result_score", 0)
        _gc     = trainer._quality_color(score)
        cv2.circle(frame, (_TCX, _TCY), _RC + 4, _gc, 5)
        _circle_text = f"{score * 100:.0f}%"
        _circ_color  = _gc
    else:
        _circle_text = f"{trainer.reps_done}/{trainer.total_reps}"

    if _circle_text:
        _ts3 = cv2.getTextSize(_circle_text, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)[0]
        draw_text_outline(frame, _circle_text,
                          (_TCX - _ts3[0] // 2, _TCY + _ts3[1] // 2 + 4),
                          cv2.FONT_HERSHEY_DUPLEX, 1.1, _circ_color, 2)

    # ══════════════════════════════════════════════════════════
    # 右側：表情指引面板 + 信心分數條（位置適配新版面）
    # ══════════════════════════════════════════════════════════
    draw_guide_panel(frame, target, status, target_conf,
                     debug["hold_remaining"], trainer.hold_duration, is_target_reached,
                     y_start=TOP_H + 10)
    draw_confidence_bars(frame, confs, target=target, y_end=BOT_Y - 8)

    # ══════════════════════════════════════════════════════════
    # 區域 3：底部資訊欄
    # ══════════════════════════════════════════════════════════
    draw_filled_rect_with_alpha(frame, (0, BOT_Y), (w, h), (8, 10, 18), 0.93)
    cv2.line(frame, (0, BOT_Y), (w, BOT_Y), (70, 70, 80), 2)

    # 左側：目標信心進度條
    q_color = trainer._quality_color(target_conf)
    draw_text_outline(frame,
                      f"目標信心：{zh_expr(target)}  {target_conf * 100:.0f}%   門檻：{trainer.hold_threshold * 100:.0f}%",
                      (16, BOT_Y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.64, q_color, 1)
    _progress_bar(frame, 16, BOT_Y + 32, w // 2 - 30, 20, target_conf, q_color)

    # 右側：目前偵測 + 臉部狀態
    draw_text_outline(frame, f"目前偵測：{zh_expr(current_class)}  ({current_conf:.0%})",
                      (w - 420, BOT_Y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, C_LIGHT_GRAY, 1)
    _face_col = C_GREEN if is_aligned else C_RED
    draw_text_outline(frame, "臉部：正常偵測中" if is_aligned else "臉部：未對齊",
                      (w - 420, BOT_Y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, _face_col, 1)

    if trainer.debug_enabled:
        probs = debug.get("probs")
        if probs is not None:
            x0, y0 = 35, TOP_H + 10
            draw_filled_rect_with_alpha(frame, (x0 - 10, y0 - 28), (x0 + 420, y0 + 160), C_BLACK, 0.65)
            cv2.putText(frame, "Debug probabilities", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1)
            for i, exp in enumerate(ALL_EXPRESSIONS):
                cv2.putText(
                    frame,
                    f"{exp}: {float(probs[i]):.3f}",
                    (x0, y0 + 28 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    EXPRESSION_INFO[exp]["color"],
                    1
                )


def draw_finished_summary(frame, trainer):
    h, w, _ = frame.shape

    draw_filled_rect_with_alpha(frame, (0, 0), (w, h), (15, 15, 15), 0.84)

    draw_center_text(frame, "訓練完成！", 90, 1.3, C_YELLOW, 2)

    avg = float(np.mean(trainer.scores)) if trainer.scores else 0.0
    draw_center_text(frame, f"平均品質：{avg * 100:.0f}%", 140, 0.75, C_LIGHT_GRAY, 1)

    x = 230
    y = 220
    row_h = 70

    cv2.putText(frame, "表情", (x, y - 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, C_WHITE, 2)
    cv2.putText(frame, "完成次數", (x + 300, y - 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, C_WHITE, 2)
    cv2.putText(frame, "最佳品質", (x + 560, y - 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, C_WHITE, 2)

    for i, exp in enumerate(ALL_EXPRESSIONS):
        info = EXPRESSION_INFO[exp]
        stat = trainer.stats[exp]
        yy = y + i * row_h

        cv2.rectangle(frame, (x - 20, yy - 38), (x + 820, yy + 18), (35, 35, 35), -1)
        cv2.rectangle(frame, (x - 20, yy - 38), (x - 10, yy + 18), info["color"], -1)

        cv2.putText(frame, info["title"], (x, yy), cv2.FONT_HERSHEY_DUPLEX, 0.65, C_WHITE, 2)
        cv2.putText(frame, str(stat["success"]), (x + 350, yy), cv2.FONT_HERSHEY_DUPLEX, 0.65, C_WHITE, 2)
        cv2.putText(frame, f"{stat['best_conf']:.2f}", (x + 600, yy), cv2.FONT_HERSHEY_DUPLEX, 0.65, C_WHITE, 2)

    cv2.putText(
        frame,
        "品質分數為該表情維持期間之平均目標信心分數。",
        (230, h - 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        C_LIGHT_GRAY,
        1
    )

    cv2.putText(
        frame,
        "按 R 重新開始      |      M 返回主選單      |      Q 離開",
        (60, h - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        C_WHITE,
        1
    )



# ==========================================================
# 臉部運動系統
# ==========================================================
class FacialExercisesSystem:
    def __init__(self):
        self.face_mesh_enabled = False
        self.face_mesh = None

        self.face_mesh_error = ""

        try:
            if mp_face_mesh is None:
                raise RuntimeError(
                    "MediaPipe FaceMesh API not found. "
                    "Please install mediapipe==0.10.14 or a version with face_mesh support."
                )

            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            self.face_mesh_enabled = True
            print("MediaPipe FaceMesh loaded successfully")

        except Exception as e:
            self.face_mesh_enabled = False
            self.face_mesh = None
            self.face_mesh_error = str(e)
            print("MediaPipe FaceMesh disabled:", e)

        self.state = "校正"
        self.is_calibrated = False
        self.reps = 0
        self.max_reps = 10

        # --- 每一 round 開始前 3 秒準備倒數 ---
        self.prep_duration = 3.0
        self.prep_remaining = self.prep_duration
        self.prep_last_update_time = None

        self.jawline_phase = 1

        # --- 10 秒 round scoring 機制 ---
        # 每一回合固定倒數 10 秒。
        # 在 10 秒內，每一幀只要動作達標，就用該幀時間差累加 action_ok_time。
        # 分數 = action_ok_time / 10 秒 * 100。
        self.hold_start_time = None
        self.hold_duration = 10.0
        self.round_duration = 10.0
        self.round_remaining = self.round_duration
        self.round_action_ok_time = 0.0
        self.round_last_update_time = None

        # 保留 hold_remaining 供原本 UI 使用，實際上代表 round_remaining。
        self.hold_remaining = self.round_duration
        self.hold_last_update_time = None

        # --- 組間休息倒數控制 ---
        # neutral 時休息倒數減少；不是 neutral 時不重置，而是慢慢加回。
        self.rest_start_time = None
        self.rest_duration = 5.0
        self.rest_remaining = self.rest_duration
        self.rest_last_update_time = None

        # --- MediaPipe 分數紀錄 ---
        self.current_quality_score = 0.0
        self.last_quality_score = 0.0
        self.average_quality_score = 0.0
        self.quality_history = []
        self.ts_history = []          # 每回合達標累積秒數（Tₛ）記錄

        self.yolo_buffer = deque(maxlen=15)
        self.tilt_buffer = deque(maxlen=10)
        self.y_diff_buffer = deque(maxlen=10)
        self.mouth_buffer = deque(maxlen=10)
        self.pout_buffer = deque(maxlen=10)
        self.brow_h_buffer = deque(maxlen=10)
        self.brow_v_buffer = deque(maxlen=10)

        self.smoothed_class = "none"

        self.debug_vals = {
            "m_height": 0.0,
            "tilt_val": 0.0,
            "pout_val": 0.0,
            "y_diff": 0.0,
            "is_aligned": False,
            "jawline_phase": 1,
            "rest_remaining": 5.0,
            "hold_remaining": 5.0,
            "is_target_reached": False,
            "mp_enabled": False,
            "has_face": False,
            "mp_error": "",
            "position_ok": False,
            "position_msg": "請將臉放入框線中",
            "quality_score": 0.0,
            "last_quality_score": 0.0,
            "average_quality_score": 0.0,
            "round_remaining": 10.0,
            "round_action_ok_time": 0.0,
            "prep_remaining": 3.0,
            "y_diff_std": 1.0,
            "mouth_std": 1.0,
            "mouth_ratio": 0.0,
            "brow_h": 0.0,
            "brow_v": 0.0,
            "brow_h_std": 1.0,
            "brow_v_std": 1.0,
            "quality_history": [],
            "finish_time": None,
            "calib_confirm_time": None
        }
        self.custom_thresh = {
            "Face_Lift": 0.01,
            "Double_Chin": 0.1,
            "Frown_Tighten_H": 0.05,
            "Frown_Tighten_V": 0.03
        }
        self.calib_D0 = {}
        # 個人化校準之原始基準值（margin 套用前），供評估流程即時計算動作幅度使用。
        # 鍵："Face_Lift"、"Double_Chin"、"Frown_Tighten_H"、"Frown_Tighten_V"
        self.calib_baseline = {}
        self.calib_confirm_time = None
        self.calib_confirm_duration = 1.5
        self.finish_time = None

        # 調整機制（論文 4.4.2）：記錄從調整狀態恢復後應回到哪個階段。
        # "準備完成"＝準備倒數結束但尚未達自然基準；"維持中"＝訓練階段中途姿勢異常。
        self.adjust_return_state = None

    def get_dist(self, p1, p2):
        return np.linalg.norm(
            np.array([p1.x, p1.y]) - np.array([p2.x, p2.y])
        )

    def get_smooth_avg(self, buffer):
        return sum(buffer) / len(buffer) if buffer else 0

    def check_face_alignment(self, lm):
        left_eye = lm[33]
        right_eye = lm[263]
        nose = lm[1]

        eye_y_diff = abs(left_eye.y - right_eye.y)
        is_level = eye_y_diff < 0.03

        dist_left = abs(nose.x - left_eye.x)
        dist_right = abs(nose.x - right_eye.x)

        if max(dist_left, dist_right) == 0:
            return False

        yaw_ratio = min(dist_left, dist_right) / max(dist_left, dist_right)
        is_centered = yaw_ratio > 0.75

        return is_level and is_centered

    def calc_mediapipe_quality_score(self, exercise_name, smooth_y_diff, smooth_mouth):
        """
        舊版幾何幅度分數已不作為最終分數。
        目前分數改由 10 秒 round 內「動作達標累積秒數」計算：
            score = action_ok_time / 10 * 100
        此函式保留作為相容用途。
        """
        return float(np.clip(self.current_quality_score, 0.0, 100.0))

    def reset_round_timer(self):
        """重新開始目前這一 round，不增加完成次數。"""
        self.round_remaining = self.round_duration
        self.round_action_ok_time = 0.0
        self.round_last_update_time = None
        self.hold_remaining = self.round_duration
        self.hold_last_update_time = None
        self.current_quality_score = 0.0
        self.debug_vals["quality_score"] = 0.0
        self.debug_vals["round_remaining"] = self.round_remaining
        self.debug_vals["round_action_ok_time"] = self.round_action_ok_time

    def reset_prep_timer(self):
        """每一 round 正式 10 秒計分前的 3 秒準備倒數。"""
        self.prep_remaining = self.prep_duration
        self.prep_last_update_time = None
        self.debug_vals["prep_remaining"] = self.prep_remaining

    def confirm_calibration(self, exercise_name):
        if not self.debug_vals["is_aligned"]:
            print("無法校正：臉部尚未對齊！")
            return

        if exercise_name == "Face_Lift":
            y0 = self.debug_vals["y_diff"]
            self.calib_baseline["Face_Lift"] = y0
            # ×1.1：門檻放寬 10%（y_diff < thresh，門檻越高越容易達標）
            self.custom_thresh["Face_Lift"] = (y0 + 0.005) * 1.05

        elif exercise_name == "Double_Chin":
            D0 = self.debug_vals["m_height"]
            self.calib_D0["Double_Chin"] = max(D0, 1e-6)
            self.calib_baseline["Double_Chin"] = D0
            # D0 × 0.95：向下寬鬆 5%（mouth > thresh，門檻越低越容易達標）
            self.custom_thresh["Double_Chin"] = D0 * 0.95

        elif exercise_name == "Frown_Tighten":
            # 校正時要求使用者「自然放鬆表情」，以放鬆狀態的數值作為個人化基準（非皺眉時的數值）。
            # 門檻直接等於放鬆基準值，只要皺眉使數值低於基準即視為達標。
            H0 = self.debug_vals["brow_h"]
            V0 = self.debug_vals["brow_v"]
            self.calib_baseline["Frown_Tighten_H"] = H0
            self.calib_baseline["Frown_Tighten_V"] = V0
            self.custom_thresh["Frown_Tighten_H"] = H0 * 1.0
            self.custom_thresh["Frown_Tighten_V"] = V0 * 1.0

        self.calib_confirm_time = time.time()
        self.is_calibrated = True
        self.state = "START"
        self.reset_round_timer()
        self.reset_prep_timer()
        self.rest_remaining = self.rest_duration
        self.rest_last_update_time = None

        print(f"已完成校正：{zh_exercise(exercise_name)}")
        if exercise_name in self.custom_thresh:
            print(f"目前門檻值：{self.custom_thresh[exercise_name]:.4f}")
        elif exercise_name == "Frown_Tighten":
            print(
                f"目前門檻值：水平={self.custom_thresh['Frown_Tighten_H']:.4f}"
                f"　垂直={self.custom_thresh['Frown_Tighten_V']:.4f}"
            )

    def reset_exercise_phase(self, exercise_name):
        # 目前已刪除親親天花板 / 下顎線訓練，保留函式供流程重置使用。
        return

    def analyze(self, exercise_name, frame, yolo_results, mp_frame=None, mp_offset=(0, 0), mp_scale=1.0):
        # --- YOLO 表情平滑化 ---
        raw_class = str(yolo_results.get("class", "none")).lower()
        self.yolo_buffer.append(raw_class)

        counts = Counter(self.yolo_buffer)
        most_common_class, count = counts.most_common(1)[0]

        if count >= 8:
            self.smoothed_class = most_common_class

        y_conf = yolo_results.get("conf", 0.0)

        self.debug_vals["calib_confirm_time"] = self.calib_confirm_time
        self.debug_vals["quality_history"] = list(self.quality_history)
        self.debug_vals["finish_time"] = self.finish_time

        self.debug_vals["mp_enabled"] = self.face_mesh_enabled
        self.debug_vals["has_face"] = False
        self.debug_vals["mp_error"] = self.face_mesh_error
        self.debug_vals["is_aligned"] = False

        # MediaPipe 沒有成功載入時，不要假裝成功
        if not self.face_mesh_enabled or self.face_mesh is None:
            return (
                self.state,
                self.reps,
                self.debug_vals,
                self.smoothed_class,
                y_conf
            )

        # MediaPipe 使用原始攝影機畫面做偵測，避免 1280x960 padding 後黑邊影響偵測。
        # 偵測到的 landmark 再依照 resize_with_padding 的 offset/scale 畫回顯示畫面。
        if mp_frame is None:
            mp_frame = frame

        mp_rgb = cv2.cvtColor(mp_frame, cv2.COLOR_BGR2RGB)
        mp_rgb.flags.writeable = False
        results = self.face_mesh.process(mp_rgb)
        mp_rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            draw_face_position_guide(frame, None)
            self.debug_vals["position_ok"] = False
            self.debug_vals["position_msg"] = "請將臉放入框線中"
            self.debug_vals["is_aligned"] = False

            # 完全偵測不到臉（遮蔽／移出畫面）視同姿勢異常，進入調整機制並凍結計時，
            # 避免下一幀恢復偵測時，把中斷期間經過的秒數一次全部計入倒數/達標時間。
            if self.state == "維持中":
                self.state = "調整"
                self.adjust_return_state = "維持中"
                self.round_last_update_time = None
                self.hold_last_update_time = None
            elif self.state == "準備中":
                self.state = "調整"
                self.adjust_return_state = "準備完成"
                self.prep_last_update_time = None
            elif self.state == "調整":
                self.round_last_update_time = None
                self.hold_last_update_time = None
                self.prep_last_update_time = None

            return (
                self.state,
                self.reps,
                self.debug_vals,
                self.smoothed_class,
                y_conf
            )

        

        is_target_reached = False
        self.debug_vals["is_aligned"] = False
        self.debug_vals["jawline_phase"] = self.jawline_phase
        self.debug_vals["rest_remaining"] = self.rest_remaining
        self.debug_vals["hold_remaining"] = self.hold_remaining
        self.debug_vals["round_remaining"] = self.round_remaining
        self.debug_vals["round_action_ok_time"] = self.round_action_ok_time
        self.debug_vals["prep_remaining"] = self.prep_remaining
        self.debug_vals["is_target_reached"] = False

        if results.multi_face_landmarks:
            self.debug_vals["has_face"] = True

            lm = results.multi_face_landmarks[0].landmark
            mp_h, mp_w = mp_frame.shape[:2]
            off_x, off_y = mp_offset

            def to_display_point(pt):
                return (
                    int(pt.x * mp_w * mp_scale + off_x),
                    int(pt.y * mp_h * mp_scale + off_y)
                )

            face_points = [to_display_point(pt) for pt in lm]
            position_ok, position_msg = draw_face_position_guide(frame, face_points)
            self.debug_vals["position_ok"] = position_ok
            self.debug_vals["position_msg"] = position_msg

            # --- 顯示 MediaPipe 偵測點 ---
            # 先畫出全臉 landmark 小點，讓你能確認 MediaPipe 確實有在運作。
            draw_landmark_dots(
                frame, lm, (120, 120, 120),
                src_w=mp_w, src_h=mp_h, scale=mp_scale, offset=(off_x, off_y)
            )

            check_pts = {}

            if exercise_name == "Face_Lift":
                check_pts = {
                    0: (0, 0, 255),
                    61: (255, 0, 0),
                    291: (0, 255, 0)
                }

            elif exercise_name == "Double_Chin":
                check_pts = {
                    13: (0, 255, 255),
                    14: (255, 0, 255)
                }

            elif exercise_name == "Frown_Tighten":
                check_pts = {
                    55: (0, 255, 255),
                    285: (255, 0, 255),
                    159: (0, 165, 255),
                    386: (255, 165, 0)
                }

            for idx, pt_color in check_pts.items():
                pt = lm[idx]
                cx, cy = to_display_point(pt)

                cv2.circle(frame, (cx, cy), 8, pt_color, -1)
                cv2.circle(frame, (cx, cy), 11, C_WHITE, 2)

                cv2.putText(
                    frame,
                    str(idx),
                    (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    C_WHITE,
                    1
                )

            # --- 檢查臉部是否正對鏡頭，並確認臉部位於框線內 ---
            pose_ok = self.check_face_alignment(lm)
            is_aligned = pose_ok and position_ok
            self.debug_vals["is_aligned"] = is_aligned

            # --- 嘴巴張開程度 ---
            raw_mouth = self.get_dist(lm[13], lm[14])
            self.mouth_buffer.append(raw_mouth)
            smooth_mouth = self.get_smooth_avg(self.mouth_buffer)
            self.debug_vals["m_height"] = smooth_mouth

            # Rₘ = Dₜ / D₀ 比例計算（論文 5.4.5）
            if "Double_Chin" in self.calib_D0 and self.calib_D0["Double_Chin"] > 0:
                self.debug_vals["mouth_ratio"] = smooth_mouth / self.calib_D0["Double_Chin"]
            else:
                self.debug_vals["mouth_ratio"] = 0.0

            # --- Face Lift：嘴角與臉部中心相對高度 ---
            diff_l = lm[61].y - lm[0].y
            diff_r = lm[291].y - lm[0].y
            raw_y_diff = max(diff_l, diff_r)
            self.y_diff_buffer.append(raw_y_diff)
            smooth_y_diff = self.get_smooth_avg(self.y_diff_buffer)
            self.debug_vals["y_diff"] = smooth_y_diff

            # --- Jawline 嘟嘴：鼻頭與上唇的 y 距離 ---
            # 上唇往上靠近鼻頭時，這個值會變小
            # --- Jawline 嘟嘴：鼻頭與上唇中心區域的距離 ---
            # 不只使用 lm[13]，改用多個上唇中心點平均，減少單一 landmark 偏移
            # --- Jawline 嘟嘴：鼻頭與上唇上緣中心的距離 ---
            # --- Jawline 嘟嘴：鼻頭與上唇上緣中心的 2D 距離 ---
            nose = lm[1]
            upper_lip = lm[0]

            raw_pout = np.linalg.norm(
                np.array([nose.x, nose.y]) - np.array([upper_lip.x, upper_lip.y])
            )

            self.pout_buffer.append(raw_pout)
            smooth_pout = self.get_smooth_avg(self.pout_buffer)
            self.debug_vals["pout_val"] = smooth_pout

            # --- Frown_Tighten：皺眉肌 AU4 判定 ---
            # 水平特徵：左右眉頭（55, 285）之間的水平距離，皺眉時會縮小。
            raw_brow_h = abs(lm[55].x - lm[285].x)
            self.brow_h_buffer.append(raw_brow_h)
            smooth_brow_h = self.get_smooth_avg(self.brow_h_buffer)
            self.debug_vals["brow_h"] = smooth_brow_h

            # 垂直特徵：左右眉頭相對同側眼睛上緣（159, 386）之垂直距離，
            # 皺眉時眉頭下壓靠近眼睛，此距離會縮小；取左右平均。
            raw_brow_v = (
                abs(lm[55].y - lm[159].y) + abs(lm[285].y - lm[386].y)
            ) / 2.0
            self.brow_v_buffer.append(raw_brow_v)
            smooth_brow_v = self.get_smooth_avg(self.brow_v_buffer)
            self.debug_vals["brow_v"] = smooth_brow_v

            # 10 幀滑動窗口穩定性評估（用於校準引導）
            self.debug_vals["y_diff_std"] = float(np.std(list(self.y_diff_buffer))) if len(self.y_diff_buffer) >= 5 else 1.0
            self.debug_vals["mouth_std"] = float(np.std(list(self.mouth_buffer))) if len(self.mouth_buffer) >= 5 else 1.0
            self.debug_vals["brow_h_std"] = float(np.std(list(self.brow_h_buffer))) if len(self.brow_h_buffer) >= 5 else 1.0
            self.debug_vals["brow_v_std"] = float(np.std(list(self.brow_v_buffer))) if len(self.brow_v_buffer) >= 5 else 1.0

            # ==================================================
            # 動作判定邏輯
            # ==================================================
            # 分數不再由幾何幅度直接換算，而是由 10 秒內動作達標的累積時間計算。
            self.debug_vals["quality_score"] = self.current_quality_score
            self.debug_vals["last_quality_score"] = self.last_quality_score
            self.debug_vals["average_quality_score"] = self.average_quality_score

            if self.is_calibrated and is_aligned:

                if exercise_name == "Face_Lift":
                    # 臉部拉提：動作達標必須偵測為 happy；neutral 不算成功。
                    is_expression_ok = self.smoothed_class == "happy"

                    is_target_reached = (
                        smooth_y_diff < self.custom_thresh["Face_Lift"]
                        and is_expression_ok
                    )

                elif exercise_name == "Double_Chin":
                    # 嘴部開合訓練：Rₘ 超過個人化門檻且 YOLO 辨識為 surprised
                    # 兩項條件須同時成立才判定達標（論文 4.4.5 雙層 AND 判斷）。
                    is_expression_ok = self.smoothed_class == "surprised"

                    is_target_reached = (
                        smooth_mouth > self.custom_thresh["Double_Chin"]
                        and is_expression_ok
                    )

                elif exercise_name == "Frown_Tighten":
                    # 皺眉緊眉：動作達標必須偵測為 angry；
                    # 水平靠攏（brow_h）與垂直下壓（brow_v）須同時達標，
                    # 避免瞇眼或頭部小幅轉動造成的假陽性。
                    is_expression_ok = self.smoothed_class == "angry"

                    is_target_reached = (
                        smooth_brow_h < self.custom_thresh["Frown_Tighten_H"]
                        and smooth_brow_v < self.custom_thresh["Frown_Tighten_V"]
                        and is_expression_ok
                    )

            # ==================================================
            # 狀態機控制：10 秒 scoring round
            # ==================================================
            if not is_aligned and self.state not in ["校正", "RESET", "維持中", "準備中", "調整"]:
                # 尚未開始正式 round 時，若臉部未對齊就停在 START。
                self.state = "START"
                self.round_last_update_time = None
                self.hold_last_update_time = None
                self.reset_prep_timer()

            elif self.state == "START":
                # 每一 round 先進入 3 秒準備倒數，讓使用者有時間調整姿勢。
                if self.is_calibrated and is_aligned:
                    self.state = "準備中"
                    self.reset_prep_timer()
                    self.prep_last_update_time = time.time()

            elif self.state == "準備中":
                now = time.time()

                if self.prep_last_update_time is None:
                    self.prep_last_update_time = now

                dt = now - self.prep_last_update_time
                self.prep_last_update_time = now

                # 準備時間固定往下倒數，中途短暫失準不中斷重來；
                # 論文 4.4.2：僅於「倒數結束時」才判斷使用者是否已回復自然基準。
                self.prep_remaining -= dt
                self.prep_remaining = max(0.0, self.prep_remaining)
                self.debug_vals["prep_remaining"] = self.prep_remaining

                if self.prep_remaining <= 0.0:
                    # 倒數結束：需臉部對齊且表情已回復 neutral，才能進入正式計分，
                    # 避免起始姿勢/表情錯誤導致訓練窗口內 Tₛ 計算失真。
                    is_ready = is_aligned and self.smoothed_class == "neutral"

                    if is_ready:
                        self.state = "維持中"
                        self.reset_round_timer()
                        now = time.time()
                        self.round_last_update_time = now
                        self.hold_last_update_time = now
                        self.hold_start_time = now
                    else:
                        # 尚未回復自然基準：進入調整機制等待，不重跑 3 秒倒數。
                        self.state = "調整"
                        self.adjust_return_state = "準備完成"

            elif self.state == "維持中":
                if not is_aligned:
                    # 訓練階段進行中偵測到姿勢異常（頭部偏移過大／臉部遭遮蔽或移出範圍）：
                    # 論文 4.4.2 調整機制—暫停 10 秒倒數與已累積達標時間 Tₛ，兩者皆凍結，
                    # 待使用者回復基準後自中斷處接續，而非重新計算。
                    self.state = "調整"
                    self.adjust_return_state = "維持中"
                    self.round_last_update_time = None
                    self.hold_last_update_time = None
                    self.debug_vals["is_target_reached"] = False

                else:
                    now = time.time()

                    if self.round_last_update_time is None:
                        self.round_last_update_time = now

                    dt = now - self.round_last_update_time
                    self.round_last_update_time = now

                    # round 倒數固定往下跑；若該幀動作達標，就累加達標時間。
                    self.round_remaining -= dt
                    self.round_remaining = max(0.0, self.round_remaining)

                    if is_target_reached:
                        self.round_action_ok_time += dt
                        self.round_action_ok_time = min(self.round_duration, self.round_action_ok_time)

                    # 分數 = 10 秒內動作達標秒數 / 10 秒 * 100
                    self.current_quality_score = float(np.clip(
                        (self.round_action_ok_time / self.round_duration) * 100.0,
                        0.0,
                        100.0
                    ))

                    self.hold_remaining = self.round_remaining
                    self.debug_vals["hold_remaining"] = self.hold_remaining
                    self.debug_vals["round_remaining"] = self.round_remaining
                    self.debug_vals["round_action_ok_time"] = self.round_action_ok_time
                    self.debug_vals["quality_score"] = self.current_quality_score
                    self.debug_vals["last_quality_score"] = self.last_quality_score
                    self.debug_vals["average_quality_score"] = self.average_quality_score
                    self.debug_vals["is_target_reached"] = is_target_reached

                    # 10 秒 round 結束：記錄分數。
                    # 若 10 秒內完全沒有任何一幀達標，則不增加次數，直接重新開始同一 round。
                    if self.round_remaining <= 0.0:
                        self.last_quality_score = self.current_quality_score
                        self.debug_vals["last_quality_score"] = self.last_quality_score

                        if self.round_action_ok_time <= 0.0:
                            # 10 秒內沒有完成任何有效動作，重新開始目前 round。
                            self.state = "START"
                            self.reset_round_timer()
                            self.reset_prep_timer()

                        else:
                            self.quality_history.append(self.last_quality_score)
                            self.ts_history.append(round(self.round_action_ok_time, 2))
                            self.average_quality_score = float(np.mean(self.quality_history))
                            self.debug_vals["average_quality_score"] = self.average_quality_score

                            self.reps += 1
                            self.reset_round_timer()

                            if self.reps >= self.max_reps:
                                self.state = "FINISHED"
                                self.finish_time = time.time()

                            else:
                                self.state = "RESET"
                                self.rest_remaining = self.rest_duration
                                self.rest_last_update_time = None
                                self.rest_start_time = time.time()
                                self.reset_exercise_phase(exercise_name)

            elif self.state == "調整":
                # 調整機制：暫停對應計時，等待使用者恢復基準後從中斷處接續。
                self.debug_vals["round_remaining"] = self.round_remaining
                self.debug_vals["round_action_ok_time"] = self.round_action_ok_time
                self.debug_vals["prep_remaining"] = self.prep_remaining
                self.debug_vals["hold_remaining"] = self.hold_remaining
                self.debug_vals["is_target_reached"] = False

                if self.adjust_return_state == "維持中":
                    if is_aligned:
                        # 恢復基準：回到訓練階段，接續原有倒數與 Tₛ，不重新計算。
                        self.state = "維持中"
                        now = time.time()
                        self.round_last_update_time = now
                        self.hold_last_update_time = now
                        self.adjust_return_state = None

                elif self.adjust_return_state == "準備完成":
                    is_ready = is_aligned and self.smoothed_class == "neutral"
                    if is_ready:
                        self.state = "維持中"
                        self.reset_round_timer()
                        now = time.time()
                        self.round_last_update_time = now
                        self.hold_last_update_time = now
                        self.hold_start_time = now
                        self.adjust_return_state = None

            elif self.state == "RESET":
                now = time.time()

                if self.rest_last_update_time is None:
                    self.rest_last_update_time = now

                dt = now - self.rest_last_update_time
                self.rest_last_update_time = now

                if self.smoothed_class == "neutral":
                    # 正確休息：倒數減少。
                    self.rest_remaining -= dt
                    self.rest_remaining = max(0.0, self.rest_remaining)
                else:
                    # 不是 neutral：不要整個重跑計時，只把秒數逐漸加回。
                    self.rest_remaining += dt
                    self.rest_remaining = min(self.rest_duration, self.rest_remaining)

                self.debug_vals["rest_remaining"] = self.rest_remaining

                if self.rest_remaining <= 0.0:
                    self.state = "START"
                    self.reset_prep_timer()
                    self.rest_remaining = self.rest_duration
                    self.rest_last_update_time = None
                    self.rest_start_time = None
                    self.reset_exercise_phase(exercise_name)

        return (
            self.state,
            self.reps,
            self.debug_vals,
            self.smoothed_class,
            y_conf
        )


# ==========================================================
# 右側 AR 指引動畫
# ==========================================================
def draw_avatar_animation(
    frame,
    exercise,
    state,
    y_cls="none",
    jawline_phase=1,
    is_target_reached=True
):
    # 校正、休息、完成、調整機制時不顯示指引
    if state in ["FINISHED", "RESET", "校正", "調整"]:
        return

    # HOLDING 且動作正確時，不顯示指引
    # HOLDING 但動作不正確時，繼續顯示指引
    if state == "維持中" and is_target_reached:
        return

    # Jawline 特別處理：
    # 如果 HOLDING 中姿勢錯誤，且表情仍是 sad，顯示第二階段嘟嘴向上
    # 如果表情不是 sad，顯示第一階段抬下巴
    if exercise == "Jawline" and state == "維持中" and not is_target_reached:
        if y_cls == "sad":
            jawline_phase = 2
        else:
            jawline_phase = 1
    h, w, _ = frame.shape

    anim_x, anim_y = w - 190, 130
    anim_w, anim_h = 165, 165

    t_glow = (math.sin(time.time() * 5) + 1) / 2.0
    glow_color = (
        int(30 + 20 * t_glow),
        int(30 + 20 * t_glow),
        int(30 + 20 * t_glow)
    )

    _ax1, _ay1 = max(0, anim_x - 5), max(0, anim_y - 5)
    _ax2, _ay2 = min(w, anim_x + anim_w + 5), min(h, anim_y + anim_h + 5)
    _aroi = frame[_ay1:_ay2, _ax1:_ax2]
    overlay = _aroi.copy()

    cv2.rectangle(
        overlay,
        (anim_x - 5 - _ax1, anim_y - 5 - _ay1),
        (anim_x + anim_w + 5 - _ax1, anim_y + anim_h + 5 - _ay1),
        glow_color,
        -1
    )

    cv2.rectangle(
        overlay,
        (anim_x - _ax1, anim_y - _ay1),
        (anim_x + anim_w - _ax1, anim_y + anim_h - _ay1),
        (20, 20, 20),
        -1
    )

    cv2.addWeighted(overlay, 0.85, _aroi, 0.15, 0, _aroi)

    cv2.putText(
        frame,
        "動作指引",
        (anim_x + 42, anim_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        C_CYAN,
        2
    )

    face_cx = anim_x + int(anim_w / 2)
    face_cy = anim_y + int(anim_h / 2) + 16
    t_anim = (math.sin(time.time() * 3) + 1) / 2.0

    cv2.circle(frame, (face_cx, face_cy), 45, (120, 120, 120), 2)

    is_blinking = math.sin(time.time() * 10) > 0.9
    eye_offset_y = 10

    if is_blinking:
        cv2.line(frame, (face_cx - 20, face_cy - eye_offset_y), (face_cx - 10, face_cy - eye_offset_y), C_WHITE, 2)
        cv2.line(frame, (face_cx + 10, face_cy - eye_offset_y), (face_cx + 20, face_cy - eye_offset_y), C_WHITE, 2)
    else:
        cv2.circle(frame, (face_cx - 15, face_cy - eye_offset_y), 3, C_WHITE, -1)
        cv2.circle(frame, (face_cx + 15, face_cy - eye_offset_y), 3, C_WHITE, -1)

    if exercise == "Face_Lift":
        cv2.putText(frame, "微笑上提", (anim_x + 42, anim_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1)

        cv2.line(frame, (face_cx - 22, face_cy - 22), (face_cx - 8, face_cy - 25), C_WHITE, 2)
        cv2.line(frame, (face_cx + 8, face_cy - 25), (face_cx + 22, face_cy - 22), C_WHITE, 2)

        mouth_y = face_cy + 20
        smile_offset = int(t_anim * 15)

        pts = np.array([
            [face_cx - 20, mouth_y - smile_offset],
            [face_cx, mouth_y + int(smile_offset / 2)],
            [face_cx + 20, mouth_y - smile_offset]
        ], np.int32)

        cv2.polylines(frame, [pts], False, C_GREEN, 3)

    elif exercise == "Jawline":
        if jawline_phase == 1:
            cv2.line(frame, (face_cx - 22, face_cy - 23), (face_cx - 8, face_cy - 23), C_WHITE, 2)
            cv2.line(frame, (face_cx + 8, face_cy - 23), (face_cx + 22, face_cy - 23), C_WHITE, 2)

            guide_label = "抬下巴"
            guide_color = C_CYAN
            jaw_progress = t_anim
            pout_progress = 0.0

        else:
            cv2.line(frame, (face_cx - 22, face_cy - 20), (face_cx - 8, face_cy - 25), C_WHITE, 2)
            cv2.line(frame, (face_cx + 8, face_cy - 25), (face_cx + 22, face_cy - 20), C_WHITE, 2)

            guide_label = "向上嘟嘴"
            guide_color = C_ORANGE
            jaw_progress = 1.0
            pout_progress = t_anim

        cv2.putText(frame, guide_label, (anim_x + 42, anim_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, guide_color, 1)

        jaw_offset = int(jaw_progress * 25)
        pout_offset = int(pout_progress * 12)

        cv2.ellipse(frame, (face_cx, face_cy), (45, 45), 0, 0, 180, (30, 30, 30), 4)

        jaw_pts = np.array([
            [face_cx - 45, face_cy],
            [face_cx, face_cy + 45 - jaw_offset],
            [face_cx + 45, face_cy]
        ], np.int32)

        cv2.polylines(frame, [jaw_pts], False, C_CYAN, 3)
        cv2.circle(frame, (face_cx, face_cy - 5 - int(jaw_offset / 3)), 2, C_WHITE, -1)

        mouth_y = face_cy + 15 - int(jaw_offset / 2) - pout_offset

        if jawline_phase == 1:
            mouth_pts = np.array([
                [face_cx - 14, mouth_y + 5],
                [face_cx, mouth_y - 2],
                [face_cx + 14, mouth_y + 5]
            ], np.int32)

            cv2.polylines(frame, [mouth_pts], False, C_WHITE, 2)

        else:
            mouth_pts = np.array([
                [face_cx - 12, mouth_y + 5],
                [face_cx, mouth_y - 2],
                [face_cx + 12, mouth_y + 5]
            ], np.int32)

            cv2.polylines(frame, [mouth_pts], False, C_ORANGE, 2)
            cv2.ellipse(frame, (face_cx, mouth_y), (8, 5), 0, 0, 180, C_ORANGE, 2)

        cv2.putText(frame, f"階段 {jawline_phase}/2", (anim_x + 50, anim_y + anim_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_WHITE, 1)

    elif exercise == "Double_Chin":
        cv2.putText(frame, "張嘴", (anim_x + 35, anim_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1)

        cv2.line(frame, (face_cx - 20, face_cy - 26), (face_cx - 10, face_cy - 28), C_WHITE, 2)
        cv2.line(frame, (face_cx + 10, face_cy - 28), (face_cx + 20, face_cy - 26), C_WHITE, 2)

        mouth_y = face_cy + 15
        open_offset = int(t_anim * 15)

        cv2.ellipse(frame, (face_cx, mouth_y), (15, open_offset), 0, 180, 360, C_ORANGE, 2)
        cv2.ellipse(frame, (face_cx, mouth_y), (15, open_offset), 0, 0, 180, C_ORANGE, 2)

    elif exercise == "Frown_Tighten":
        cv2.putText(frame, "皺眉", (anim_x + 35, anim_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1)

        # 眉毛隨動畫幅度向下、向中間靠攏
        frown_off = int(t_anim * 10)
        cv2.line(frame, (face_cx - 24, face_cy - 26 + frown_off), (face_cx - 9 + frown_off, face_cy - 16), C_RED, 3)
        cv2.line(frame, (face_cx + 9 - frown_off, face_cy - 16), (face_cx + 24, face_cy - 26 + frown_off), C_RED, 3)

        # 眉間皺褶示意
        cv2.line(frame, (face_cx - 3, face_cy - 20), (face_cx - 3, face_cy - 10), (120, 120, 220), 1)
        cv2.line(frame, (face_cx + 3, face_cy - 20), (face_cx + 3, face_cy - 10), (120, 120, 220), 1)

        cv2.line(frame, (face_cx - 25, face_cy + 22), (face_cx + 25, face_cy + 22), C_RED, 3)


# ==========================================================
# 動作狀態大型提示卡（新增）
# ==========================================================
def draw_action_status_card(frame, state, exercise, y_cls, y_conf, dbg):
    """
    在畫面左側中央顯示大型動作狀態卡片。
    清楚顯示：目前偵測動作、是否達標、具體修正指引。
    """
    h, w, _ = frame.shape

    if not dbg.get("has_face", False):
        return
    if state not in ["校正", "START", "準備中", "維持中", "RESET", "調整"]:
        return

    is_aligned   = dbg.get("is_aligned", False)
    is_reached   = dbg.get("is_target_reached", False)
    action_ok    = dbg.get("round_action_ok_time", 0.0)
    rest_sec     = dbg.get("rest_remaining", 5.0)

    card_x  = 20
    card_y  = 248
    card_w  = min(700, w - 280)
    card_h  = 185

    # 背景色與邊框色依狀態決定
    if state == "維持中" and is_reached:
        bg_color, border_color = (0, 45, 0), C_GREEN
    elif state == "維持中" and not is_reached:
        bg_color, border_color = (45, 0, 0), C_RED
    elif state == "RESET":
        bg_color, border_color = (0, 32, 42), C_CYAN
    elif state == "校正":
        bg_color, border_color = (35, 22, 0), C_ORANGE
    elif state == "調整":
        bg_color, border_color = (40, 35, 0), C_YELLOW
    else:
        bg_color, border_color = (10, 10, 45), C_BLUE

    draw_filled_rect_with_alpha(
        frame,
        (card_x, card_y),
        (card_x + card_w, card_y + card_h),
        bg_color, 0.88
    )
    cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), border_color, 3)

    lx = card_x + 18

    if state == "校正":
        draw_text_outline(frame, "校準中",
                          (lx, card_y + 44), cv2.FONT_HERSHEY_DUPLEX, 0.92, C_ORANGE, 2)
        draw_text_outline(frame, "請放鬆臉部，保持自然表情，面對攝影機",
                          (lx, card_y + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.76, C_WHITE, 2)
        if exercise == "Face_Lift":
            stability_ok = dbg.get("y_diff_std", 1.0) < 0.002
        elif exercise == "Frown_Tighten":
            stability_ok = (
                dbg.get("brow_h_std", 1.0) < 0.002
                and dbg.get("brow_v_std", 1.0) < 0.002
            )
        else:
            stability_ok = dbg.get("mouth_std", 1.0) < 0.002
        if stability_ok:
            draw_text_outline(frame, "數值穩定  ✓   請按  C  完成校準",
                              (lx, card_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.80, C_GREEN, 2)
        else:
            draw_text_outline(frame, "請稍等，等待數值穩定...",
                              (lx, card_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.76, C_YELLOW, 2)

    elif state == "START":
        draw_text_outline(frame, "等待偵測...",
                          (lx, card_y + 55), cv2.FONT_HERSHEY_DUPLEX, 1.0, border_color, 2)
        draw_text_outline(frame, "請正對攝影機，將臉放入框線中",
                          (lx, card_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.78, C_WHITE, 2)

    elif state == "準備中":
        prep_sec = math.ceil(dbg.get("prep_remaining", 3.0))
        draw_text_outline(frame, f"準備開始  ─  {prep_sec} 秒",
                          (lx, card_y + 58), cv2.FONT_HERSHEY_DUPLEX, 1.08, border_color, 2)
        if exercise == "Face_Lift":
            hint = "準備做出  開心微笑  並上提嘴角"
        elif exercise == "Frown_Tighten":
            hint = "準備做出  生氣皺眉  眉頭下壓靠攏"
        else:
            hint = "準備做出  驚訝張嘴  張大嘴巴"
        draw_text_outline(frame, hint,
                          (lx, card_y + 118), cv2.FONT_HERSHEY_SIMPLEX, 0.80, C_WHITE, 2)

    elif state == "維持中":
        expr_text = f"偵測表情：{zh_expr(y_cls)}  ({y_conf:.0%})"
        cv2.putText(frame, expr_text, (lx, card_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, C_LIGHT_GRAY, 1)
        if is_reached:
            draw_text_outline(frame, "動作達標  ✓  分數累積中",
                              (lx, card_y + 96), cv2.FONT_HERSHEY_DUPLEX, 1.08, C_GREEN, 2)
            draw_text_outline(frame, f"已累積：{action_ok:.1f} 秒",
                              (lx, card_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.80, C_GREEN, 2)
        else:
            draw_text_outline(frame, "動作未達標  ✗  請調整",
                              (lx, card_y + 90), cv2.FONT_HERSHEY_DUPLEX, 1.05, C_RED, 2)
            if exercise == "Face_Lift":
                hint = "做出  開心表情  ＋  用力上提嘴角"
            elif exercise == "Frown_Tighten":
                hint = "做出  生氣表情  ＋  用力皺眉靠攏"
            else:
                hint = "做出  驚訝表情  ＋  嘴巴張大"
            draw_text_outline(frame, hint,
                              (lx, card_y + 148), cv2.FONT_HERSHEY_SIMPLEX, 0.82, C_YELLOW, 2)

    elif state == "RESET":
        draw_text_outline(frame, "組間休息  ─  請放鬆臉部",
                          (lx, card_y + 52), cv2.FONT_HERSHEY_DUPLEX, 0.98, C_CYAN, 2)
        if y_cls == "neutral":
            draw_text_outline(frame, "表情正常  ✓  請持續放鬆",
                              (lx, card_y + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.82, C_GREEN, 2)
        else:
            draw_text_outline(frame, f"偵測到：{zh_expr(y_cls)}  ─  請回到  自然表情",
                              (lx, card_y + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.80, C_RED, 2)
        draw_text_outline(frame, f"休息剩餘：{rest_sec:.1f} 秒",
                          (lx, card_y + 158), cv2.FONT_HERSHEY_SIMPLEX, 0.74, C_LIGHT_GRAY, 1)

    elif state == "調整":
        draw_text_outline(frame, "等待恢復基準  ─  計時已暫停",
                          (lx, card_y + 52), cv2.FONT_HERSHEY_DUPLEX, 0.92, C_YELLOW, 2)
        if not is_aligned:
            draw_text_outline(frame, "請正對攝影機，將臉放回框線中",
                              (lx, card_y + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.78, C_WHITE, 2)
        else:
            draw_text_outline(frame, "請放鬆臉部，回到自然表情",
                              (lx, card_y + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.78, C_WHITE, 2)


# ==========================================================
# 動作示意圖
# ==========================================================
def draw_exercises_mini_guide(frame, exercise, state, is_target_reached):
    """右側小型示意圖，說明當前訓練動作的步驟與動態效果。"""
    h, w, _ = frame.shape
    TOP_H = 190

    px, py = w - 224, TOP_H + 220
    pw, ph = 214, 298
    if py + ph > h - 100:
        ph = h - 100 - py
    if ph < 80:
        return

    t_a = (math.sin(time.time() * 2.5) + 1) / 2.0  # 0~1 動畫值

    if state == "維持中" and is_target_reached:
        border_col = C_GREEN
    elif state == "維持中":
        border_col = C_RED
    elif state == "校正":
        border_col = C_ORANGE
    else:
        border_col = (80, 80, 100)

    draw_filled_rect_with_alpha(frame, (px, py), (px + pw, py + ph), (12, 14, 24), 0.90)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), border_col, 2)

    if exercise == "Face_Lift":
        ex_name, ex_sub = "提拉臉頰", "自然微笑 "
    elif exercise == "Double_Chin":
        ex_name, ex_sub = "消雙下巴", "自然張嘴 "
    elif exercise == "Frown_Tighten":
        ex_name, ex_sub = "皺眉緊眉", "自然皺眉 "
    else:
        ex_name, ex_sub = "下顎線條", "抬下巴 + 嘟嘴"

    draw_text_outline(frame, ex_name, (px + 12, py + 28), cv2.FONT_HERSHEY_DUPLEX, 0.70, border_col, 2)
    cv2.putText(frame, ex_sub, (px + 12, py + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_LIGHT_GRAY, 1)
    cv2.line(frame, (px + 8, py + 58), (px + pw - 8, py + 58), (55, 55, 70), 1)

    fcx = px + pw // 2
    fcy = py + 148
    fr  = 52

    # 臉型輪廓
    cv2.circle(frame, (fcx, fcy), fr, (100, 100, 110), 2)

    # 眼睛（共用）
    for dx in [-18, 18]:
        cv2.circle(frame, (fcx + dx, fcy - 14), 4, (200, 200, 200), -1)

    # ── 各動作專屬 ──────────────────────────────────────────
    if exercise == "Face_Lift":
        # 輕鬆眉
        cv2.line(frame, (fcx - 28, fcy - 30), (fcx - 10, fcy - 28), (170, 170, 170), 2)
        cv2.line(frame, (fcx + 10, fcy - 28), (fcx + 28, fcy - 30), (170, 170, 170), 2)

        # 上揚微笑（動畫幅度隨 t_a 變化）
        my = fcy + 22
        smile_off = int(6 + t_a * 12)
        pts = np.array([[fcx - 26, my - smile_off],
                         [fcx, my + smile_off // 2],
                         [fcx + 26, my - smile_off]], np.int32)
        cv2.polylines(frame, [pts], False, C_GREEN, 3)

        # 上提箭頭
        for ax in [fcx - 26, fcx + 26]:
            off = int(t_a * 10)
            cv2.arrowedLine(frame, (ax, my - smile_off + 8 - off),
                            (ax, my - smile_off - 12 - off), C_GREEN, 2, tipLength=0.4)

        # 臉頰圈
        ck_a = int(80 + t_a * 120)
        cv2.circle(frame, (fcx - 32, fcy + 4), 14, (ck_a, ck_a // 2, 0), 1)
        cv2.circle(frame, (fcx + 32, fcy + 4), 14, (ck_a, ck_a // 2, 0), 1)

        steps = ["1. 自然微笑", "2. 嘴角自然", "3. 臉頰感受緊繃"]

    elif exercise == "Double_Chin":
        # 驚訝眉
        cv2.line(frame, (fcx - 28, fcy - 32), (fcx - 8, fcy - 26), (170, 170, 170), 2)
        cv2.line(frame, (fcx + 8, fcy - 26), (fcx + 28, fcy - 32), (170, 170, 170), 2)

        # 張嘴 O（動畫高度）
        oh = int(14 + t_a * 18)
        cv2.ellipse(frame, (fcx, fcy + 22), (16, oh), 0, 0, 360, C_ORANGE, 3)

        # 向下箭頭
        off = int(t_a * 8)
        cv2.arrowedLine(frame, (fcx, fcy + 22 + oh - off),
                        (fcx, fcy + 22 + oh + 18 - off), C_ORANGE, 2, tipLength=0.4)

        # 下巴標示
        cv2.ellipse(frame, (fcx, fcy + fr - 4), (26, 11), 0, 0, 180, (180, 120, 0), 1)
        cv2.putText(frame, "down chin", (fcx - 28, fcy + fr + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 120, 0), 1)

        steps = ["1. 自然張嘴", "2. 嘴巴自然張大", "3. 感受下巴收緊"]

    elif exercise == "Frown_Tighten":
        # 皺眉：眉毛隨動畫幅度向下、向中間靠攏
        frown_off = int(t_a * 8)
        cv2.line(frame, (fcx - 30, fcy - 30 + frown_off), (fcx - 10 + frown_off, fcy - 18), C_RED, 2)
        cv2.line(frame, (fcx + 10 - frown_off, fcy - 18), (fcx + 30, fcy - 30 + frown_off), C_RED, 2)

        # 眉間皺褶
        cv2.line(frame, (fcx - 3, fcy - 24), (fcx - 3, fcy - 12), (120, 120, 220), 1)
        cv2.line(frame, (fcx + 3, fcy - 24), (fcx + 3, fcy - 12), (120, 120, 220), 1)

        # 向內靠攏箭頭
        arrow_off = int(t_a * 8)
        cv2.arrowedLine(frame, (fcx - 30, fcy - 26), (fcx - 18 + arrow_off, fcy - 20), C_RED, 2, tipLength=0.5)
        cv2.arrowedLine(frame, (fcx + 30, fcy - 26), (fcx + 18 - arrow_off, fcy - 20), C_RED, 2, tipLength=0.5)

        # 緊抿的嘴
        cv2.line(frame, (fcx - 24, fcy + 22), (fcx + 24, fcy + 22), C_RED, 3)

        steps = ["1. 自然皺眉", "2. 眉頭向下向中間靠攏", "3. 感受眉間肌肉緊繃"]

    else:  # Jawline
        # 平靜眉
        cv2.line(frame, (fcx - 28, fcy - 30), (fcx - 8, fcy - 26), (170, 170, 170), 2)
        cv2.line(frame, (fcx + 8, fcy - 26), (fcx + 28, fcy - 30), (170, 170, 170), 2)

        # 每 2 秒切換動作相位
        jaw_phase = int(time.time() * 0.5) % 2

        if jaw_phase == 0:
            # 相位 1：抬下巴
            jaw_off = int(t_a * 12)
            jaw_pts = np.array([[fcx - fr + 6, fcy + 10],
                                  [fcx, fcy + fr - jaw_off],
                                  [fcx + fr - 6, fcy + 10]], np.int32)
            cv2.polylines(frame, [jaw_pts], False, C_CYAN, 2)
            cv2.arrowedLine(frame, (fcx, fcy + fr - jaw_off + 6),
                            (fcx, fcy + fr - jaw_off - 14), C_CYAN, 2, tipLength=0.4)
            cv2.line(frame, (fcx - 18, fcy + 20), (fcx + 18, fcy + 20), (170, 170, 170), 2)
            cv2.putText(frame, "chin up", (fcx - 22, fcy + fr + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CYAN, 1)
        else:
            # 相位 2：嘟嘴
            pout_off = int(t_a * 10)
            pout_pts = np.array([[fcx - 18, fcy + 24],
                                   [fcx, fcy + 14 - pout_off],
                                   [fcx + 18, fcy + 24]], np.int32)
            cv2.polylines(frame, [pout_pts], False, C_CYAN, 2)
            cv2.ellipse(frame, (fcx, fcy + 28), (14, 8), 0, 0, 180, C_CYAN, 2)
            cv2.arrowedLine(frame, (fcx, fcy + 14 - pout_off + 6),
                            (fcx, fcy + 14 - pout_off - 14), C_CYAN, 2, tipLength=0.4)
            cv2.putText(frame, "pout lip", (fcx - 22, fcy + fr + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CYAN, 1)

        steps = ["1. 下巴向上抬", "2. 嘴唇向上嘟", "3. 感受下顎緊繃"]

    # 步驟說明
    sep_y = py + 210
    cv2.line(frame, (px + 8, sep_y), (px + pw - 8, sep_y), (55, 55, 70), 1)
    for i, step in enumerate(steps):
        cv2.putText(frame, step, (px + 12, sep_y + 22 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # 底部狀態列
    if state == "維持中":
        s_icon = "達標!" if is_target_reached else "調整中"
        s_col  = C_GREEN if is_target_reached else C_RED
        draw_filled_rect_with_alpha(frame, (px + 8, py + ph - 30), (px + pw - 8, py + ph - 4),
                                    (0, 50, 0) if is_target_reached else (50, 0, 0), 0.80)
        _ts = cv2.getTextSize(s_icon, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)[0]
        draw_text_outline(frame, s_icon, (px + pw // 2 - _ts[0] // 2, py + ph - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.58, s_col, 2)


# ==========================================================
# 訓練畫面 UI
# ==========================================================
def draw_ui(
    frame,
    state,
    reps,
    dbg,
    y_cls,
    y_conf,
    exercise,
    thresh,
    hold_duration,
    hold_start_time,
    rest_start_time,
    rest_duration
):
    h, w, _ = frame.shape

    # ── 版面常數 ─────────────────────────────────────────────
    TOP_H    = 190   # 頂部指令區高度
    BOTTOM_H = 95    # 底部資訊欄高度
    BOT_Y    = h - BOTTOM_H

    # ── 決定狀態色彩與三行文字 ────────────────────────────────
    color       = C_WHITE
    status_text = "準備"
    main_text   = "請跟隨指引"
    sub_text    = ""

    if not dbg["is_aligned"]:
        color       = C_RED
        status_text = "調整位置"
        main_text   = dbg.get("position_msg", "請正對攝影機，將臉放入框線中")
        sub_text    = "請確保臉部出現在畫面中央框線內"
    else:
        if state == "校正":
            color       = C_ORANGE
            status_text = "校準模式"
            if exercise == "Face_Lift":
                _cal_hint = "自然微笑"
            elif exercise == "Frown_Tighten":
                _cal_hint = "自然放鬆表情"
            else:
                _cal_hint = "自然張嘴"
            main_text   = f"請{_cal_hint}並維持"
            sub_text    = "觀察下方數值穩定後按  C  完成校準"

        elif state == "START":
            color       = C_BLUE
            status_text = "準備開始"
            if exercise == "Face_Lift":
                main_text = "即將開始！請準備自然微笑，並上提嘴角"
            elif exercise == "Frown_Tighten":
                main_text = "即將開始！請準備皺眉，並將眉頭向下靠攏"
            else:
                main_text = "即將開始！請準備張嘴，並張大嘴巴"
            sub_text = "系統偵測到您的臉部，倒數結束後請立刻做出動作"

        elif state == "準備中":
            color       = C_BLUE
            status_text = "準備中"
            prep_sec = math.ceil(float(dbg.get("prep_remaining", 3.0)))

            if exercise == "Face_Lift":
                main_text = f"倒數  {prep_sec}  秒後計分開始 ─ 做好  自然微笑 +  自然上提嘴角"
            elif exercise == "Double_Chin":
                main_text = f"倒數  {prep_sec}  秒後計分開始 ─ 做好  張嘴 +  自然張大"
            elif exercise == "Frown_Tighten":
                main_text = f"倒數  {prep_sec}  秒後計分開始 ─ 做好  皺眉 +  眉頭下壓靠攏"

        elif state == "維持中":
            if dbg.get("is_target_reached", False):
                color       = C_GREEN
                status_text = "動作達標"
                main_text   = "很棒！請維持這個動作，分數持續累積中"
                if exercise == "Face_Lift":
                    sub_text = "保持  自然微笑 + 嘴角 自然上提  直到計時結束"
                elif exercise == "Frown_Tighten":
                    sub_text = "保持  皺眉 + 眉頭下壓靠攏  直到計時結束"
                else:
                    sub_text = "保持  張嘴 + 嘴巴 自然張大  直到計時結束"
            else:
                color       = C_RED
                status_text = "請調整動作"
                if exercise == "Face_Lift":
                    main_text = "請自然微笑，並以 自然上提嘴角"
                    sub_text  = "嘴角需比放鬆時更高，臉頰肌肉以 自然上提"
                elif exercise == "Frown_Tighten":
                    main_text = "請皺眉，並將眉頭向下靠攏"
                    sub_text  = "眉頭需比放鬆時更靠近、更貼近眼睛"
                else:
                    main_text = "請張嘴，並以 自然張大嘴巴"
                    sub_text  = "嘴巴需比放鬆時開得更大，露出上下排牙齒"

        elif state == "RESET":
            color = C_CYAN
            status_text = "組間休息"
            _rest_remain = dbg.get("rest_remaining", 5.0)
            if y_cls != "neutral":
                color     = C_ORANGE
                main_text = "請放鬆臉部，回到  自然放鬆  的表情"
            else:
                main_text = "表情放鬆！請休息，等待下一組開始"
            sub_text = f"休息剩餘：{_rest_remain:.0f} 秒，下一組即將開始"

        elif state == "調整":
            color       = C_YELLOW
            status_text = "等待恢復基準"
            main_text   = "偵測到姿勢或表情異常，計時已暫停"
            sub_text    = "請回到自然表情並正對攝影機，系統將自動接續訓練"

        elif state == "FINISHED":
            color       = C_BLUE
            status_text = "訓練完成"
            main_text   = "恭喜完成所有回合！"
            sub_text    = "按  M  返回主選單選擇其他動作"

    # ══════════════════════════════════════════════════════════
    # 區域 1：頂部指令區（深色背景，清楚呈現當前任務）
    # ══════════════════════════════════════════════════════════
    draw_filled_rect_with_alpha(frame, (0, 0), (w, TOP_H), (8, 10, 18), 0.93)

    # 第一行：狀態 pill（動態寬度）+ 動作名稱 + 回合數
    _pill_tw = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0][0]
    _px1, _py1 = 16, 10
    _px2, _py2 = 16 + _pill_tw + 30, 68
    cv2.rectangle(frame, (_px1, _py1), (_px2, _py2), color, -1)
    draw_text_outline(frame, status_text, (_px1 + 13, _py2 - 11),
                      cv2.FONT_HERSHEY_DUPLEX, 1.0, (8, 10, 18), 2)
    draw_text_outline(frame, dbg.get("exercise_display", zh_exercise(exercise)), (_px2 + 18, 60),
                      cv2.FONT_HERSHEY_DUPLEX, 1.08, C_YELLOW, 2)
    _reps_text = f"回合  {reps} / {dbg.get('max_reps', 10)}"
    _reps_tw   = cv2.getTextSize(_reps_text, cv2.FONT_HERSHEY_DUPLEX, 0.96, 2)[0][0]
    draw_text_outline(frame, _reps_text, (w - _reps_tw - 18, 60),
                      cv2.FONT_HERSHEY_DUPLEX, 0.96, C_WHITE, 2)

    # 第一行底部細分隔線
    cv2.line(frame, (16, 74), (w - 16, 74), (55, 55, 68), 1)

    # 第二行：主要動作指引（最重要，大字，顏色随狀態）
    _main_color = (
        C_GREEN  if (state == "維持中" and dbg.get("is_target_reached", False)) else
        C_RED    if (state == "維持中" and not dbg.get("is_target_reached", True)) else
        C_RED    if not dbg["is_aligned"] else
        C_ORANGE if state == "校正" else
        C_ORANGE if (state == "RESET" and color == C_ORANGE) else
        C_CYAN   if state == "RESET" else
        C_YELLOW if state == "調整" else
        C_WHITE
    )
    draw_text_outline(frame, main_text, (18, 128),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.94, _main_color, 2)

    # 第三行：補充說明（灰色小字）
    if sub_text:
        draw_text_outline(frame, sub_text, (18, 166),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.66, C_LIGHT_GRAY, 1)

    # 快捷鍵提示（右側，不占主要視覺空間）
    draw_text_outline(frame, "C：校正   M：主選單   Q：離開",
                      (w - 328, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (180, 180, 100), 1)

    # 頂部指令區下方色彩邊線（反映狀態）
    cv2.line(frame, (0, TOP_H), (w, TOP_H), color, 3)

    # 全畫面狀態邊框
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)

    # ══════════════════════════════════════════════════════════
    # 浮動計時圓圈（右側，不干擾臉部偵測區）
    # ══════════════════════════════════════════════════════════
    _TCX = w - 96
    _TCY = TOP_H + 108
    _RC  = 74

    _gx1, _gy1 = max(0, _TCX - _RC - 16), max(0, _TCY - _RC - 16)
    _gx2, _gy2 = min(frame.shape[1], _TCX + _RC + 16), min(frame.shape[0], _TCY + _RC + 16)
    _groi = frame[_gy1:_gy2, _gx1:_gx2]
    _ov = _groi.copy()
    cv2.circle(_ov, (_TCX - _gx1, _TCY - _gy1), _RC + 16, (6, 8, 16), -1)
    cv2.addWeighted(_ov, 0.80, _groi, 0.20, 0, _groi)
    cv2.circle(frame, (_TCX, _TCY), _RC, (55, 55, 55), 6)

    _progress = 0.0
    _text_p   = ""
    if dbg["is_aligned"]:
        if state == "準備中":
            _rem = dbg.get("prep_remaining", 3.0)
            _progress = max(0.0, 1.0 - _rem / 3.0)
            _text_p = f"{math.ceil(_rem)}s"
        elif state == "維持中":
            _rem = dbg.get("round_remaining", hold_duration)
            _progress = max(0.0, 1.0 - _rem / hold_duration)
            _text_p = f"{math.ceil(_rem)}s"
        elif state == "RESET":
            _rem = dbg.get("rest_remaining", rest_duration)
            _progress = max(0.0, 1.0 - _rem / rest_duration)
            _text_p = f"{int(_rem)}s"

    if state == "維持中":
        _gr = int(_RC + 12 * math.sin(time.time() * 8))
        _gc = (80, 255, 80) if dbg.get("is_target_reached", False) else (60, 60, 255)
        cv2.circle(frame, (_TCX, _TCY), _gr, _gc, 2)

    if _progress > 0:
        cv2.ellipse(frame, (_TCX, _TCY), (_RC, _RC), 0, -90, -90 + int(_progress * 360), color, 7)

    if _text_p:
        _ts2 = cv2.getTextSize(_text_p, cv2.FONT_HERSHEY_DUPLEX, 1.15, 2)[0]
        draw_text_outline(frame, _text_p,
                          (_TCX - _ts2[0] // 2, _TCY + _ts2[1] // 2 + 4),
                          cv2.FONT_HERSHEY_DUPLEX, 1.15, color, 2)

    # 動作示意圖（計時圓圈下方，僅限訓練中顯示）
    if state not in ["FINISHED"]:
        draw_exercises_mini_guide(frame, exercise, state, dbg.get("is_target_reached", False))

    # ══════════════════════════════════════════════════════════
    # 區域 3：底部資訊欄（達標進度 + 分數 + FaceMesh）
    # ══════════════════════════════════════════════════════════
    draw_filled_rect_with_alpha(frame, (0, BOT_Y), (w, h), (8, 10, 18), 0.93)
    cv2.line(frame, (0, BOT_Y), (w, BOT_Y), (70, 70, 80), 2)

    _score_now  = float(dbg.get("quality_score", 0.0))
    _score_last = float(dbg.get("last_quality_score", 0.0))
    _score_avg  = float(dbg.get("average_quality_score", 0.0))
    _ok_sec     = float(dbg.get("round_action_ok_time", 0.0))
    _sc_color   = C_GREEN if _score_now >= 60 else C_YELLOW

    _blx = 16
    _bly = BOT_Y + 14
    _blw = w // 2 - 40
    _blh = 20
    _blf = min(1.0, _ok_sec / hold_duration) if hold_duration > 0 else 0.0
    draw_text_outline(frame, "達標進度",
                      (_blx, _bly + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_LIGHT_GRAY, 1)
    _bfx = _blx + 90
    cv2.rectangle(frame, (_bfx, _bly), (_bfx + _blw, _bly + _blh), (40, 40, 40), -1)
    cv2.rectangle(frame, (_bfx, _bly), (_bfx + int(_blf * _blw), _bly + _blh), _sc_color, -1)
    cv2.rectangle(frame, (_bfx, _bly), (_bfx + _blw, _bly + _blh), (100, 100, 100), 1)
    draw_text_outline(frame, f"{_ok_sec:.1f} / {hold_duration:.0f} 秒",
                      (_bfx + _blw + 8, _bly + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.62, _sc_color, 1)
    draw_text_outline(
        frame,
        f"本次：{_score_now:.0f} 分    上次：{_score_last:.0f} 分    平均：{_score_avg:.0f} 分",
        (_blx, BOT_Y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.66, _sc_color, 1
    )

    _mp_ok  = dbg.get("has_face", False) and dbg.get("mp_enabled", False)
    _mp_txt = "FaceMesh  已偵測" if _mp_ok else "FaceMesh  未偵測"
    _mp_col = C_GREEN if _mp_ok else C_RED
    draw_text_outline(frame, _mp_txt,
                      (w - 420, BOT_Y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, _mp_col, 1)
    draw_text_outline(frame, f"偵測表情：{zh_expr(y_cls)}  ({y_conf:.0%})",
                      (w - 420, BOT_Y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.60, C_LIGHT_GRAY, 1)
    draw_text_outline(frame, "C：校正   M：主選單   Q：離開",
                      (w - 420, BOT_Y + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 100), 1)

    # ══════════════════════════════════════════════════════════
    # 校正模式：特徵數值面板（疊加在底部欄上方）
    # ══════════════════════════════════════════════════════════
    if state == "校正" and dbg.get("has_face", False) and dbg.get("is_aligned", False):
        _cal_h = 118
        _cal_y = BOT_Y - _cal_h
        draw_filled_rect_with_alpha(frame, (0, _cal_y), (w, BOT_Y), (16, 26, 42), 0.93)
        cv2.rectangle(frame, (0, _cal_y), (w, BOT_Y), C_ORANGE, 2)

        _extra_feature_txt = ""

        if exercise == "Face_Lift":
            _fval  = dbg.get("y_diff", 0.0)
            _fstd  = dbg.get("y_diff_std", 1.0)
            _fname = "嘴角高度差"
            _fmax  = 0.15
            _cal_hint = "自然微笑"
        elif exercise == "Frown_Tighten":
            _fval  = dbg.get("brow_h", 0.0)
            _fstd  = max(dbg.get("brow_h_std", 1.0), dbg.get("brow_v_std", 1.0))
            _fname = "眉頭水平距離"
            _fmax  = 0.12
            _cal_hint = "自然放鬆表情"
            _extra_feature_txt = (
                f"　｜　眉頭-眼睛垂直距離：{dbg.get('brow_v', 0.0):.4f}"
                f"　波動 σ：{dbg.get('brow_v_std', 1.0):.5f}"
            )
        else:
            _fval  = dbg.get("m_height", 0.0)
            _fstd  = dbg.get("mouth_std", 1.0)
            _fname = "嘴巴張開程度"
            _fmax  = 0.10
            _cal_hint = "自然張嘴"

        _stable   = _fstd < 0.002
        _stab_col = C_GREEN if _stable else C_YELLOW
        _stab_txt = f"數值穩定  ✓   請按  C  完成校準（{_cal_hint}即可）" if _stable else f"請{_cal_hint}做出動作並靜止，等待數值穩定..."

        draw_text_outline(
            frame, f"即時特徵 ─ {_fname}：{_fval:.4f}　波動 σ：{_fstd:.5f}{_extra_feature_txt}",
            (16, _cal_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_CYAN, 1
        )
        _cbx, _cby = 16, _cal_y + 46
        _cbw, _cbh = w - 220, 20
        _cnf = min(1.0, _fval / _fmax) if _fmax > 0 else 0.0
        cv2.rectangle(frame, (_cbx, _cby), (_cbx + _cbw, _cby + _cbh), (40, 40, 40), -1)
        cv2.rectangle(frame, (_cbx, _cby), (_cbx + int(_cnf * _cbw), _cby + _cbh), _stab_col, -1)
        cv2.rectangle(frame, (_cbx, _cby), (_cbx + _cbw, _cby + _cbh), (100, 100, 100), 1)
        draw_text_outline(frame, f"{_fval:.4f}",
                          (_cbx + _cbw + 8, _cby + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.60, C_WHITE, 1)
        draw_text_outline(frame, _stab_txt,
                          (16, _cal_y + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.82, _stab_col, 2)

    # ══════════════════════════════════════════════════════════
    # 校準完成閃訊
    # ══════════════════════════════════════════════════════════
    _cct = dbg.get("calib_confirm_time", None)
    if _cct is not None and (time.time() - _cct) < 1.5:
        _mw, _mh = 580, 76
        _mx = (w - _mw) // 2
        _my = h // 2 - 38
        draw_filled_rect_with_alpha(frame, (_mx, _my), (_mx + _mw, _my + _mh), (0, 55, 0), 0.95)
        cv2.rectangle(frame, (_mx, _my), (_mx + _mw, _my + _mh), C_GREEN, 2)
        draw_text_outline(frame, "校準完成，個人基準值已記錄",
                          (_mx + 32, _my + 50), cv2.FONT_HERSHEY_DUPLEX, 0.92, C_GREEN, 2)

    # ══════════════════════════════════════════════════════════
    # FINISHED：訓練結果摘要彈窗
    # ══════════════════════════════════════════════════════════
    if state == "FINISHED":
        _qh  = dbg.get("quality_history", [])
        _pw  = 600
        _ph  = max(210, 128 + len(_qh) * 34 + 80)
        _ppx = (w - _pw) // 2
        _ppy = max(TOP_H + 20, (h - _ph) // 2)
        draw_filled_rect_with_alpha(frame, (_ppx, _ppy),
                                    (_ppx + _pw, _ppy + _ph), (10, 12, 22), 0.96)
        cv2.rectangle(frame, (_ppx, _ppy), (_ppx + _pw, _ppy + _ph), C_BLUE, 3)
        draw_text_outline(frame, "訓練結果摘要",
                          (_ppx + 24, _ppy + 48), cv2.FONT_HERSHEY_DUPLEX, 1.0, C_YELLOW, 2)
        cv2.line(frame, (_ppx + 16, _ppy + 60),
                 (_ppx + _pw - 16, _ppy + 60), (60, 60, 90), 1)
        _avg = float(np.mean(_qh)) if _qh else 0.0
        for _i, _sc in enumerate(_qh):
            _ry  = _ppy + 88 + _i * 34
            _bfp = int(min(_sc / 100.0, 1.0) * 250)
            _bc  = C_GREEN if _sc >= 60 else C_YELLOW
            cv2.rectangle(frame, (_ppx + 190, _ry - 18),
                          (_ppx + 190 + 250, _ry + 4), (40, 40, 40), -1)
            cv2.rectangle(frame, (_ppx + 190, _ry - 18),
                          (_ppx + 190 + _bfp, _ry + 4), _bc, -1)
            draw_text_outline(frame, f"第 {_i + 1} 回合：{_sc:.0f} 分",
                              (_ppx + 24, _ry), cv2.FONT_HERSHEY_SIMPLEX, 0.68, C_WHITE, 1)
        _ay = _ppy + 88 + len(_qh) * 34 + 28
        draw_text_outline(frame, f"整體平均分數：{_avg:.0f} 分",
                          (_ppx + 24, _ay), cv2.FONT_HERSHEY_SIMPLEX, 0.82, C_YELLOW, 2)
        draw_text_outline(frame, "按  M  返回主選單",
                          (_ppx + 24, _ppy + _ph - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.64, C_LIGHT_GRAY, 1)

# ==========================================================
# ==========================================================
# 評估資料記錄器
# ==========================================================
class DataLogger:
    """
    自動將每回合分數、Tₛ、校準基準值存成 CSV，供評估研究使用。
    使用方式：
        logger = DataLogger(participant_id="01", session="pre_test")
        logger.log_calibration("Face_Lift", c0=0.032, threshold=0.037)
        logger.log_round("Face_Lift", round_num=1, score=80.0, ts=8.0)
        logger.save()   # 訓練結束時呼叫
    """
    SESSION_LABELS = {
        "pre_test":  "前測",
        "training":  "訓練",
        "post_test": "後測",
    }

    def __init__(self, participant_id="01", session="training", save_dir=None):
        self.participant_id = str(participant_id).zfill(2)
        self.session        = session
        self.save_dir       = Path(save_dir) if save_dir else (SCRIPT_DIR / "evaluation_data")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._rounds        = []          # [{exercise, round, score, ts_achieved}]
        self._calibrations  = []          # [{exercise, baseline, threshold}]
        self._expr_rounds   = []          # [{expression, round_idx, score}]
        self._amplitudes    = []          # [{exercise, round, amp_displacement, amp_mean_sustained, raw_extremum, duration_s, n_frames}]
        self.last_saved_path = None
        # 同一個 session 內多個階段共用同一個 logger，save() 會被呼叫多次；
        # 記錄每份清單上次成功寫入時的長度，長度沒變就不用重寫該檔案，
        # 減少不必要的重複寫入（也降低撞到檔案短暫鎖定的機率）。
        self._last_saved_len = {"rounds": 0, "expr": 0, "calib": 0, "amp": 0}

    # ── 記錄介面 ──────────────────────────────────────────────
    def log_calibration(self, exercise, baseline, threshold):
        self._calibrations.append({
            "participant": self.participant_id,
            "session":     self.session,
            "exercise":    exercise,
            "baseline":    round(float(baseline), 6),
            "threshold":   round(float(threshold), 6),
        })

    def log_round(self, exercise, round_num, score, ts_achieved):
        self._rounds.append({
            "participant": self.participant_id,
            "session":     self.session,
            "exercise":    exercise,
            "round":       int(round_num),
            "score":       round(float(score), 2),
            "Ts":          round(float(ts_achieved), 2),
        })

    def log_expression_round(self, expression, round_idx, score):
        self._expr_rounds.append({
            "participant": self.participant_id,
            "session":     self.session,
            "expression":  expression,
            "round_idx":   int(round_idx),
            "score":       round(float(score), 4),
        })

    def log_amplitude(self, exercise, round_num, amp_displacement, amp_mean_sustained,
                       raw_extremum, duration_s, n_frames):
        """記錄「維持中」狀態即時量測之動作幅度（非事後由影片重建）。
        對應論文回測腳本 round_amplitudes() 之同名欄位，唯此處數值來自
        即時狀態機已知之真實回合起訖幀，非從影片以啟發式規則猜測回合邊界。
        """
        self._amplitudes.append({
            "participant":         self.participant_id,
            "session":             self.session,
            "exercise":            exercise,
            "round":               int(round_num),
            "amp_displacement":    round(float(amp_displacement), 6),
            "amp_mean_sustained":  round(float(amp_mean_sustained), 6),
            "raw_extremum":        round(float(raw_extremum), 6),
            "duration_s":          round(float(duration_s), 2),
            "n_frames":            int(n_frames),
        })

    # ── 儲存 ──────────────────────────────────────────────────
    @staticmethod
    def _write_csv_retry(path, fieldnames, rows, attempts=5, delay_s=0.3):
        """寫入 CSV，遇到 PermissionError 時重試。

        同一個 session（如 training）內有多個階段（表情訓練、臉部拉提、下顎線條）
        共用同一個 DataLogger 實例，每個階段結束都會重新呼叫 save()、重寫全部
        已累積的 CSV（包含前一階段已寫過的檔案）。Windows 上防毒軟體／OneDrive
        常會在檔案剛寫入的瞬間短暫鎖住它，導致緊接著的第二次寫入撞到
        PermissionError；這與資料內容或邏輯無關，重試即可解決。
        """
        import time as _time
        last_err = None
        for attempt in range(attempts):
            try:
                with open(path, "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader(); w.writerows(rows)
                return
            except PermissionError as e:
                last_err = e
                if attempt < attempts - 1:
                    print(f"[警告] 寫入 {path.name} 遇到權限錯誤，{delay_s:.1f}秒後重試"
                          f"（第 {attempt + 1}/{attempts} 次）...")
                    _time.sleep(delay_s)
        raise last_err

    def save(self):
        saved = []
        prefix = f"P{self.participant_id}_{self.session}"

        if self._rounds and len(self._rounds) != self._last_saved_len["rounds"]:
            p = self.save_dir / f"{prefix}_exercises.csv"
            self._write_csv_retry(p, ["participant", "session", "exercise", "round", "score", "Ts"], self._rounds)
            self._last_saved_len["rounds"] = len(self._rounds)
            saved.append(str(p))

        if self._expr_rounds and len(self._expr_rounds) != self._last_saved_len["expr"]:
            p = self.save_dir / f"{prefix}_expression.csv"
            self._write_csv_retry(p, ["participant", "session", "expression", "round_idx", "score"], self._expr_rounds)
            self._last_saved_len["expr"] = len(self._expr_rounds)
            saved.append(str(p))

        if self._calibrations and len(self._calibrations) != self._last_saved_len["calib"]:
            p = self.save_dir / f"{prefix}_calibration.csv"
            self._write_csv_retry(p, ["participant", "session", "exercise", "baseline", "threshold"], self._calibrations)
            self._last_saved_len["calib"] = len(self._calibrations)
            saved.append(str(p))

        if self._amplitudes and len(self._amplitudes) != self._last_saved_len["amp"]:
            p = self.save_dir / f"{prefix}_amplitude.csv"
            self._write_csv_retry(p, [
                "participant", "session", "exercise", "round",
                "amp_displacement", "amp_mean_sustained", "raw_extremum",
                "duration_s", "n_frames"], self._amplitudes)
            self._last_saved_len["amp"] = len(self._amplitudes)
            saved.append(str(p))

        if saved:
            self.last_saved_path = saved[0]
            print("\n========== 資料已儲存 ==========")
            for s in saved:
                print(" ", s)
            print("=================================\n")
        return saved


# 音效管理（winsound 背景執行緒播放，不阻塞主迴圈）
# ==========================================================
class SoundManager:
    """
    音效管理器。
    單次音效（準備、開始、休息、完成）：狀態轉換時觸發一次。
    持續音效（達標節拍、錯誤警示）：每幀呼叫，內部以時間間隔控制播放頻率。
    所有聲音在 daemon 背景執行緒播放，不阻塞主迴圈。
    """
    def __init__(self):
        self._last_target_time = 0.0
        self._last_error_time  = 0.0
        self._target_interval  = 0.82   # 達標節拍間隔（秒）
        self._error_interval   = 1.6    # 錯誤警示間隔（秒）

    def _play(self, pattern):
        def _worker():
            for freq, dur in pattern:
                try:
                    winsound.Beep(int(freq), int(dur))
                except Exception:
                    pass
        threading.Thread(target=_worker, daemon=True).start()

    # ── 準備倒數：最後一秒高音、其餘中音 ──────────────────────
    def play_prep_tick(self, count):
        if count == 1:
            self._play([(1047, 200)])           # 高音長鳴
        else:
            self._play([(660, 110)])             # 中音短促

    # ── 訓練計分開始：C-E-G-C 上行四音 ────────────────────────
    def play_training_start(self):
        self._last_target_time = time.time() + 0.55  # 避免開始音與達標音重疊
        self._play([(523, 80), (659, 80), (784, 80), (1047, 320)])

    # ── 動作達標持續節拍（每 0.82 秒一次雙音節）───────────────
    def play_target_continuous(self):
        now = time.time()
        if now - self._last_target_time >= self._target_interval:
            self._last_target_time = now
            self._play([(880, 55), (1047, 90)])  # 輕快上行雙音

    # ── 動作錯誤持續警示（每 1.6 秒一次低沉雙音）──────────────
    def play_error_continuous(self):
        now = time.time()
        if now - self._last_error_time >= self._error_interval:
            self._last_error_time = now
            self._play([(330, 130), (262, 200)])  # 沉重下行雙音

    # ── 組間休息：G-E-C 下行三音，放鬆感 ─────────────────────
    def play_rest(self):
        self._play([(784, 110), (659, 110), (523, 230)])

    # ── 訓練全部完成：C-E-G-C-E 五音上行慶賀 ──────────────────
    def play_finished(self):
        self._play([(523, 100), (659, 100), (784, 100), (1047, 100), (1319, 430)])


# ==========================================================
# 主程式
# ==========================================================
def run(
    weights=ROOT / "weights" / "v7best.onnx",
    source="0",
    imgsz=(224, 224),
    conf_thres=0.01,
    iou_thres=0.45,
    device="cpu",
    view_img=False,
    participant="01",
    session="training",
    reps=10,
    fps_log=None,
    auto_exercise=None,
    auto_duration=None,
):
    print("========== PROGRAM START ==========")
    print("ONNX CPU 模式")
    print("權重檔 =", weights)
    print("來源 =", source)
    print("顯示尺寸 =", f"{DISPLAY_W}x{DISPLAY_H}")

    if not Path(weights).exists():
        print("錯誤：找不到權重檔：", weights)
        return

    if isinstance(imgsz, (list, tuple)):
        if len(imgsz) == 1:
            model_w = int(imgsz[0])
            model_h = int(imgsz[0])
        else:
            model_w = int(imgsz[0])
            model_h = int(imgsz[1])
    else:
        model_w = int(imgsz)
        model_h = int(imgsz)

    print("模型內容輸入 =", f"{model_w}x{model_h}")

    model = ONNXDetector(
        weights,
        class_names=ALL_EXPRESSIONS,
        img_size=(model_w, model_h)
    )

    cap_source = int(source) if str(source).isdigit() else str(source)
    cap = cv2.VideoCapture(cap_source, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(cap_source)

    if not cap.isOpened():
        print("錯誤：無法開啟來源：", source)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)

    current_screen = "MENU"
    current_exercise = None
    exercises_sys = None
    expression_trainer = None
    expression_alignment_checker = None

    sound_mgr = SoundManager()
    data_logger = DataLogger(participant_id=participant, session=session)
    _exercises_saved   = set()   # 避免重複儲存（FINISHED 狀態可能維持多幀）
    _expr_saved   = False
    prev_exercises_state = None
    prev_target_reached = False
    prev_prep_ceil = None

    window_name = "臉部運動訓練系統"

    if view_img:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, WINDOW_W, WINDOW_H)

    # ══════════════════════════════════════════════════════════
    # 效能測速用鉤子（供論文 3.4 節「實際顯示吞吐量」量測，預設關閉，
    # 不影響一般互動使用）：
    #   fps_log      ：指定 CSV 路徑後，逐秒記錄視窗吞吐量（含 GUI 繪製與
    #                   cv2.waitKeyEx 事件迴圈，是完整正式系統的真實 FPS）。
    #   auto_exercise：略過主選單，開機直接進入指定運動項目，並在偵測到
    #                   臉部對齊後自動完成校準（等同自動按下 C 鍵），完成
    #                   一輪後自動重新開始，讓系統可在無人操作下持續運作。
    #   auto_duration：經過此秒數後自動結束（等同自動按下 Q 鍵），供無人
    #                   值守的背景測速使用。
    # ══════════════════════════════════════════════════════════
    _max_reps = reps  # reps 這個區域變數稍後會在迴圈內被 exercises_sys.analyze() 的回傳值覆寫，先保留原始上限
    _run_start_time = time.perf_counter()
    fps_csv_file = None
    fps_csv_writer = None
    fps_run_start = None
    fps_window_start = None
    fps_window_frames = 0
    fps_total_frames = 0
    if fps_log:
        fps_csv_file = open(fps_log, "w", newline="", encoding="utf-8")
        fps_csv_writer = csv.writer(fps_csv_file)
        fps_csv_writer.writerow(["elapsed_s", "frame_idx", "window_fps"])
        fps_run_start = time.perf_counter()
        fps_window_start = fps_run_start

    auto_aligned_streak = 0
    if auto_exercise:
        current_exercise = auto_exercise
        exercises_sys = FacialExercisesSystem()
        exercises_sys.max_reps = _max_reps
        current_screen = "exercises_TRAINING"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("錯誤：無法讀取影像畫面")
            break
        frame = cv2.flip(frame, 1)  # 鏡像顯示，符合使用者面對攝影機的直覺

        im0, pad_x, pad_y, disp_scale = resize_with_padding(
            frame,
            DISPLAY_W,
            DISPLAY_H,
            return_info=True
        )

        # ==================================================
        # 主選單
        # ==================================================
        if current_screen == "MENU":
            draw_main_menu(im0)

            if view_img:
                cv2.imshow(window_name, resize_for_window(im0))
                key = cv2.waitKeyEx(1)

                if key_pressed(key, "q") or key == 27:
                    break

                elif key_pressed(key, "1"):
                    expression_trainer = ExpressionTrainingSystem(rounds=2)
                    expression_alignment_checker = FaceAlignmentChecker()
                    current_screen = "EXPRESSION_TRAINING"

                elif key_pressed(key, "2"):
                    current_exercise = "Face_Lift"
                    exercises_sys = FacialExercisesSystem()
                    exercises_sys.max_reps = reps
                    current_screen = "exercises_TRAINING"

                elif key_pressed(key, "3"):
                    current_exercise = "Double_Chin"
                    exercises_sys = FacialExercisesSystem()
                    exercises_sys.max_reps = reps
                    current_screen = "exercises_TRAINING"

                elif key_pressed(key, "4"):
                    current_exercise = "Frown_Tighten"
                    exercises_sys = FacialExercisesSystem()
                    exercises_sys.max_reps = reps
                    current_screen = "exercises_TRAINING"

            continue

        # ==================================================
        # ONNX CPU 表情辨識
        # ==================================================
        onnx_result = model.infer(im0, conf_thres=conf_thres)
        yolo_results = {
            "class": normalize_class_name(onnx_result.get("class", "none")),
            "conf": float(onnx_result.get("conf", 0.0)),
            "all_conf": {
                normalize_class_name(k): float(v)
                for k, v in onnx_result.get("all_conf", {}).items()
            },
            "probs": onnx_result.get("probs", None)
        }

        # ==================================================
        # 1. 臉部表情訓練
        # ==================================================
        if current_screen == "EXPRESSION_TRAINING":
            if expression_trainer is None:
                expression_trainer = ExpressionTrainingSystem(rounds=2)
            if expression_alignment_checker is None:
                expression_alignment_checker = FaceAlignmentChecker()

            is_aligned, align_msg = expression_alignment_checker.check(im0)

            status, feedback, color, debug = expression_trainer.analyze(
                yolo_results,
                is_aligned
            )

            confs = yolo_results.get("all_conf", {})

            if expression_trainer.state == "DONE":
                draw_finished_summary(im0, expression_trainer)
                # 表情訓練全數完成：記錄各回合分數（只執行一次）
                if not _expr_saved:
                    for idx, (expr, sc) in enumerate(
                            zip(expression_trainer.sequence, expression_trainer.scores), start=1):
                        data_logger.log_expression_round(expr, idx, sc)
                    data_logger.save()
                    _expr_saved = True
            else:
                draw_training_ui(
                    im0,
                    expression_trainer,
                    status,
                    feedback,
                    color,
                    debug,
                    is_aligned,
                    align_msg,
                    confs
                )

        # ==================================================
        # 2. 臉部拉提訓練 / 3. 嘴部開合訓練
        # ==================================================
        elif current_screen == "exercises_TRAINING":
            if exercises_sys is None:
                exercises_sys = FacialExercisesSystem()
                if current_exercise is None:
                    current_exercise = "Double_Chin"

            state, reps, dbg, y_cls, y_conf = exercises_sys.analyze(
                current_exercise,
                im0,
                yolo_results,
                mp_frame=frame,
                mp_offset=(pad_x, pad_y),
                mp_scale=disp_scale
            )

            # --- 音效觸發 ---
            is_target_reached = dbg.get("is_target_reached", False)

            # 狀態轉換：單次音效 + 資料儲存
            if state != prev_exercises_state:
                if state == "準備中":
                    prev_prep_ceil = None
                elif state == "維持中":
                    sound_mgr.play_training_start()
                elif state == "RESET":
                    sound_mgr.play_rest()
                elif state == "FINISHED":
                    sound_mgr.play_finished()
                    # 訓練結束：將所有回合資料寫入 CSV（每個 exercise 只存一次）
                    if current_exercise not in _exercises_saved:
                        for i, (sc, ts) in enumerate(zip(exercises_sys.quality_history,
                                                          exercises_sys.ts_history), start=1):
                            data_logger.log_round(current_exercise, i, sc, ts)
                        data_logger.save()
                        _exercises_saved.add(current_exercise)
                prev_exercises_state = state

            # 準備倒數：每整數秒觸發一次
            if state == "準備中":
                prep_ceil = math.ceil(dbg.get("prep_remaining", 3.0))
                if prep_ceil != prev_prep_ceil and 1 <= prep_ceil <= 3:
                    sound_mgr.play_prep_tick(prep_ceil)
                    prev_prep_ceil = prep_ceil

            # 訓練中：動作未達標時播放警示音
            if state == "維持中":
                if not is_target_reached:
                    sound_mgr.play_error_continuous()    # 錯誤警示

            prev_target_reached = is_target_reached

            draw_ui(
                im0,
                state,
                reps,
                dbg,
                y_cls,
                y_conf,
                current_exercise,
                exercises_sys.custom_thresh,
                exercises_sys.hold_duration,
                exercises_sys.hold_start_time,
                exercises_sys.rest_start_time,
                exercises_sys.rest_duration
            )

            # --- 自動測速模式：等同自動按 C 完成校準、FINISHED 後自動重來 ---
            if auto_exercise:
                if exercises_sys.state == "校正":
                    if dbg.get("is_aligned", False):
                        auto_aligned_streak += 1
                    else:
                        auto_aligned_streak = 0
                    if auto_aligned_streak >= 30:  # 約1秒(30幀)持續對齊後自動校準
                        exercises_sys.confirm_calibration(current_exercise)
                        auto_aligned_streak = 0
                elif state == "FINISHED":
                    exercises_sys = FacialExercisesSystem()
                    exercises_sys.max_reps = _max_reps
                    prev_exercises_state = None
                    prev_target_reached = False
                    prev_prep_ceil = None
                    auto_aligned_streak = 0

        if view_img:
            cv2.imshow(window_name, resize_for_window(im0))
            key = cv2.waitKeyEx(1)

            # --- FPS 測速記錄（含GUI繪製與事件迴圈，逐秒記錄一次視窗吞吐量）---
            if fps_log:
                fps_total_frames += 1
                fps_window_frames += 1
                _now = time.perf_counter()
                if _now - fps_window_start >= 1.0:
                    _window_fps = fps_window_frames / (_now - fps_window_start)
                    fps_csv_writer.writerow([
                        round(_now - fps_run_start, 2), fps_total_frames, round(_window_fps, 2)
                    ])
                    fps_csv_file.flush()
                    fps_window_start = _now
                    fps_window_frames = 0

            if auto_duration and (time.perf_counter() - _run_start_time) >= auto_duration:
                break

            if key_pressed(key, "q") or key == 27:
                break

            if current_screen == "EXPRESSION_TRAINING":
                if key_pressed(key, "m"):
                    current_screen = "MENU"
                    expression_trainer = None
                    expression_alignment_checker = None

                elif key_pressed(key, "r"):
                    if expression_trainer is not None:
                        expression_trainer.reset()

                elif key_pressed(key, "d"):
                    if expression_trainer is not None:
                        expression_trainer.debug_enabled = not expression_trainer.debug_enabled

                elif key == 32:
                    if expression_trainer is not None:
                        expression_trainer.start()

            elif current_screen == "exercises_TRAINING":
                if key_pressed(key, "m"):
                    current_screen = "MENU"
                    current_exercise = None
                    exercises_sys = None
                    prev_exercises_state = None
                    prev_target_reached = False
                    prev_prep_ceil = None

                elif key_pressed(key, "c") and exercises_sys is not None and exercises_sys.state == "校正":
                    exercises_sys.confirm_calibration(current_exercise)
                    # 記錄校準基準值
                    if current_exercise == "Face_Lift":
                        baseline = exercises_sys.debug_vals.get("y_diff", 0.0)
                        thresh   = exercises_sys.custom_thresh.get("Face_Lift", 0.0)
                        data_logger.log_calibration("Face_Lift", baseline, thresh)
                    elif current_exercise == "Double_Chin":
                        baseline = exercises_sys.calib_D0.get("Double_Chin", 0.0)
                        thresh   = exercises_sys.custom_thresh.get("Double_Chin", 0.0)
                        data_logger.log_calibration("Double_Chin", baseline, thresh)
                    elif current_exercise == "Frown_Tighten":
                        baseline_h = exercises_sys.debug_vals.get("brow_h", 0.0)
                        thresh_h   = exercises_sys.custom_thresh.get("Frown_Tighten_H", 0.0)
                        data_logger.log_calibration("Frown_Tighten_H", baseline_h, thresh_h)
                        baseline_v = exercises_sys.debug_vals.get("brow_v", 0.0)
                        thresh_v   = exercises_sys.custom_thresh.get("Frown_Tighten_V", 0.0)
                        data_logger.log_calibration("Frown_Tighten_V", baseline_v, thresh_v)

    cap.release()
    cv2.destroyAllWindows()
    if fps_csv_file:
        fps_csv_file.close()
        print(f"FPS 記錄已存至：{fps_log}")


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        default=str(SCRIPT_DIR / "weights" / "v7best.onnx")
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0"
    )

    parser.add_argument(
        "--imgsz",
        "--img",
        nargs="+",
        type=int,
        default=[224, 224]
    )

    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.45
    )

    parser.add_argument(
         "--device",
        default="cpu",
        help="使用 CPU"
    )

    parser.add_argument(
        "--view-img",
        action="store_true",
        default=True
    )

    parser.add_argument(
        "--participant",
        type=str,
        default="01",
        help="受試者編號，例如 01、02…15"
    )

    parser.add_argument(
        "--session",
        type=str,
        default="training",
        choices=["pre_test", "training", "post_test"],
        help="訓練階段：pre_test（前測）/ training（正式訓練）/ post_test（後測）"
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=10,
        help="每項運動最多回合數（前後測建議設 3）"
    )

    parser.add_argument(
        "--fps-log",
        type=str,
        default=None,
        help="供效能測速用：指定CSV路徑後，逐秒記錄含GUI繪製之真實視窗吞吐量FPS"
    )

    parser.add_argument(
        "--auto-exercise",
        type=str,
        default=None,
        choices=["Face_Lift", "Double_Chin", "Frown_Tighten"],
        help="供效能測速用：略過主選單直接進入指定運動項目，偵測到臉部對齊後自動校準、完成後自動重來"
    )

    parser.add_argument(
        "--auto-duration",
        type=float,
        default=None,
        help="供效能測速用：經過此秒數後自動結束程式，供無人值守之背景測速"
    )

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1

    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
