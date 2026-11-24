# -*- coding: utf-8 -*-
import os, csv
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
MODEL = os.environ.get("SEG_ONNX", f"{PKG}/model/seg_task_ssl_pb9-3-off.onnx")
LABELS = os.environ.get("SEG_LABELS", f"{PKG}/model/seg_labels.csv")

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

INFER_H = 1024
CONF = 0.5

PRESETS = {
    3: dict(pick="near", min_area=1500, min_w=180, min_ar=3.0, close=(221, 15),
            col=(0, 255, 0)),        # stop line, green
    2: dict(pick="big",  min_area=4000, min_w=200, min_ar=1.5, close=(201, 201),
            col=(0, 150, 255)),      # crosswalk, orange
}
FALLBACK = dict(pick="near", min_area=1500, min_w=180, min_ar=3.0, close=(221, 15),
                col=(255, 200, 0))

_rows = list(csv.DictReader(open(LABELS)))
NAME = {int(r["id"]): r["name"] for r in _rows}
ABBR = {r["abbr"].upper(): int(r["id"]) for r in _rows}
ABBR_OF = {int(r["id"]): r["abbr"].upper() for r in _rows}
PALETTE = np.zeros((len(_rows), 3), np.uint8)
for _r in _rows:
    PALETTE[int(_r["id"])] = (int(_r["b"]), int(_r["g"]), int(_r["r"]))   # BGR


def ids(spec):
    """"SL,CW" or "3,2" -> [3, 2]; "all" -> every class except background."""
    if spec.strip().lower() == "all":
        return [i for i in sorted(NAME) if i != 0]
    return [ABBR[t.strip().upper()] if t.strip().upper() in ABBR else int(t)
            for t in spec.split(",")]


def cfg(c, **ov):
    """Post-processing parameters for one class; non-None entries in ov override the
    preset. Returns a dict that also carries the morphology kernel as 'ker'."""
    d = dict(PRESETS.get(c, FALLBACK))
    for k in ("pick", "min_area", "min_w", "min_ar"):
        if ov.get(k) is not None:
            d[k] = ov[k]
    if ov.get("close_w") is not None or ov.get("close_h") is not None:
        d["close"] = (ov.get("close_w") or d["close"][0], ov.get("close_h") or d["close"][1])
    d["ker"] = cv2.getStructuringElement(cv2.MORPH_RECT, d["close"])
    return d


def describe(c, d):
    return (f"class {c}={NAME[c]}  pick={d['pick']}  min-area>={d['min_area']} "
            f"min-width>={d['min_w']} min-aspect>={d['min_ar']} "
            f"close-kernel={d['close'][0]}x{d['close'][1]}")


class Seg:
    """ONNX inference wrapper. Returns (predicted class map, probability map), both at
    the same resolution as the input frame."""

    def __init__(self, head="fine", infer_h=INFER_H, model=MODEL):
        if not os.path.exists(model):
            raise SystemExit(
                f"Model file not found: {model}\n"
                f"The weights are not shipped with this repository (they are not owned "
                f"by this project).\nRequest seg_task_ssl_pb9-3-off.onnx from the author "
                f"and place it in model/,\nor point at it with SEG_ONNX=/path/to/model.onnx")
        import onnxruntime as ort
        avail = ort.get_available_providers()
        self.s = ort.InferenceSession(
            model, providers=[p for p in ("CUDAExecutionProvider",
                                          "CPUExecutionProvider") if p in avail])
        on = [o.name for o in self.s.get_outputs()]
        self.pi, self.qi = on.index(f"{head}_preds"), on.index(f"{head}_probs")
        self.h, self.model = infer_h, model

    @property
    def providers(self):
        return self.s.get_providers()

    def __call__(self, fr):
        H, W = fr.shape[:2]
        h = self.h
        w = max(32, int(round(W * h / H / 32)) * 32)     # width must be a multiple of 32
        x = cv2.cvtColor(cv2.resize(fr, (w, h)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        o = self.s.run(None, {"input": ((x - MEAN) / STD).transpose(2, 0, 1)[None]})
        return (cv2.resize(o[self.pi][0].astype(np.uint8), (W, H),
                           interpolation=cv2.INTER_NEAREST),
                cv2.resize(o[self.qi][0], (W, H), interpolation=cv2.INTER_LINEAR))


def components(pred, prob, c, d, conf=CONF):
    """Mask -> closing -> connected components -> area / width / aspect-ratio filtering.

    Returns ([(x, y, w, h, area, label), ...], label image), already sorted by d['pick']
    ('big' = largest area first, 'near' = lowest bottom edge first). Scanning and
    rendering share this function; otherwise a scan would select clips whose detections
    are all filtered out at render time (measured on 20260818_072500).
    """
    m = cv2.morphologyEx(((pred == c) & (prob >= conf)).astype(np.uint8),
                         cv2.MORPH_CLOSE, d["ker"])
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = [(int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3]), int(st[i, 4]), i)
           for i in range(1, n)
           if st[i, 4] >= d["min_area"] and st[i, 2] >= d["min_w"]
           and st[i, 3] > 0 and st[i, 2] / st[i, 3] >= d["min_ar"]]
    out.sort(key=(lambda b: -b[4]) if d["pick"] == "big" else (lambda b: -(b[1] + b[3])))
    return out, lab


SUPPRESS = {3: dict(ref=2, min_gap=50)}


def suppress(comps, ref_mask, min_gap):
    """Drop components whose near-edge gap to ref_mask is under min_gap (compared within
    the same x range)."""
    out = []
    for c in comps:
        x, y, w, h = c[0], c[1], c[2], c[3]
        rows = np.nonzero(ref_mask[:, x:x + w].any(1))[0]
        if len(rows) and (y + h) - int(rows.max()) < min_gap:
            continue
        out.append(c)
    return out
