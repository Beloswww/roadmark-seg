# -*- coding: utf-8 -*-
"""Run the seg_task_ssl ONNX (road-marking segmentation) and turn the selected classes
into detection boxes.

The model and the per-class parameters live in segcfg.py.

Usage:
  python infer_seg_onnx.py video <video> <F0> <F1> <out.mp4> [--cls SL,CW] [--clean]
  python infer_seg_onnx.py shots <video> <frame,frame,...> --out <dir> [--tag prefix]
  python infer_seg_onnx.py probe <video> <frame,frame,...>
  python infer_seg_onnx.py image <image...> [--out dir] [--cls SL,CW|all] [--mask] [--clean]
"""
import sys, os, json, argparse, collections
import numpy as np, cv2
import segcfg as C

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["video", "probe", "shots", "image"])
ap.add_argument("video", help="path to the video; in image mode, path to the image")
ap.add_argument("rest", nargs="*", help="mode-dependent: frame range, frame list, or more images")
ap.add_argument("--cls", default="SL", help="class ids or abbreviations, comma separated; 'all' for every class. Each class uses its own parameters")
ap.add_argument("--head", default="fine", choices=["fine", "coarse"],
                help="which segmentation head to read")
ap.add_argument("--infer", type=int, default=C.INFER_H, help="inference short side")
ap.add_argument("--conf", type=float, default=C.CONF, help="confidence threshold")
ap.add_argument("--win", type=int, default=6,
                help="video mode: temporal confirmation window, in frames")
ap.add_argument("--hit", type=int, default=3,
                help="video mode: hits required within --win before a box is emitted")
ap.add_argument("--fps", type=float, default=15.0, help="frame rate of the output video")
ap.add_argument("--clean", action="store_true", help="draw only the boxes and class names (no mask overlay, no confidence)")
ap.add_argument("--box", default="bbox", choices=["bbox", "rbox", "both"],
                help="which box form to draw (all three are always written to the json)")
ap.add_argument("--all", action="store_true", help="keep every detection per class, not just one")
ap.add_argument("--no-suppress", action="store_true", help="disable cross-class suppression (see segcfg.SUPPRESS)")
ap.add_argument("--mask", action="store_true", help="image mode: overlay the mask of every class, ignoring the box thresholds")
ap.add_argument("--sup-gap", type=int, default=None, help="minimum gap in px used for suppression, overriding the default")
ap.add_argument("--out", default=None, help="output directory (shots / image modes)")
ap.add_argument("--tag", default="", help="filename prefix for written images")
# If given, these override the preset for EVERY class; otherwise each class keeps its own
_OV = {"pick": "'near' (lowest bottom edge) or 'big' (largest area)",
       "min-area": "minimum component area in px", "min-w": "minimum width in px",
       "min-ar": "minimum width/height aspect ratio",
       "close-w": "closing kernel width", "close-h": "closing kernel height"}
for f, t in (("pick", str), ("min-area", int), ("min-w", int), ("min-ar", float),
             ("close-w", int), ("close-h", int)):
    ap.add_argument(f"--{f}", type=t, default=None, help=f"override for ALL classes: {_OV[f]}")
a = ap.parse_args()

CLS = C.ids(a.cls)
CFG = {c: C.cfg(c, pick=a.pick, min_area=a.min_area, min_w=a.min_w, min_ar=a.min_ar,
                close_w=a.close_w, close_h=a.close_h) for c in CLS}
for c in CLS:
    print(C.describe(c, CFG[c]) + f"  box-colour-BGR={CFG[c]['col']}")

SUP = {} if a.no_suppress else {c: dict(C.SUPPRESS[c]) for c in CLS if c in C.SUPPRESS}
for c in SUP:
    if a.sup_gap is not None:
        SUP[c]["min_gap"] = a.sup_gap
    print(f"suppress {C.NAME[c]}: drop any whose near-edge gap to "
          f"{C.NAME[SUP[c]['ref']]} is < {SUP[c]['min_gap']}px")

seg = C.Seg(head=a.head, infer_h=a.infer)
print(f"model {os.path.basename(seg.model)}  provider {seg.providers}", flush=True)
if "CUDAExecutionProvider" not in seg.providers:
    print("  !! GPU not in use, running on CPU -- roughly 37x slower "
          "(2.4 s/frame vs 64 ms/frame).\n"
          "     If onnxruntime printed a CUDA error above, see the "
          "'GPU not working' section of the README.", flush=True)


def boxes(cid, pred, prob):
    comp, lab = C.components(pred, prob, cid, CFG[cid], a.conf)
    if cid in SUP:                      # suppression must run before the one-per-class cut
        ref = (pred == SUP[cid]["ref"]) & (prob >= a.conf)
        comp = C.suppress(comp, ref, SUP[cid]["min_gap"])
    out = []
    for x, y, w, h, ar, i in comp:
        mm = (lab == i).astype(np.uint8)
        c = max(cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                key=cv2.contourArea)
        out.append({"cls": int(cid), "name": C.NAME[cid], "bbox": [x, y, w, h],
                    "rbox": cv2.boxPoints(cv2.minAreaRect(c)).astype(int).tolist(),
                    "poly": cv2.approxPolyDP(c, 4, True).reshape(-1, 2).tolist(),
                    "conf": float(prob[mm == 1].mean()), "area": ar})
    return out if a.all else out[:1]


def draw(fr, pred, prob, per, conf):
    """per: {cid: [box, ...]}; conf: {cid: bool}. With --clean, draw only boxes and
    class names -- no mask overlay, no confidence value."""
    vis = fr.copy()
    if not a.clean:
        m = np.isin(pred, [c for c in CLS if per.get(c)]) & (prob >= a.conf)
        if m.any():
            vis[m] = (0.55 * C.PALETTE[pred[m]] + 0.45 * vis[m]).astype(np.uint8)
    for cid in CLS:
        if not conf.get(cid):
            continue
        col = CFG[cid]["col"]
        for b in per[cid]:
            x, y, w, h = b["bbox"]
            if a.box in ("bbox", "both"):
                cv2.rectangle(vis, (x, y), (x + w, y + h), col, 3)
            if a.box in ("rbox", "both"):
                cv2.polylines(vis, [np.array(b["rbox"])], True, col, 3)
            lbl = b["name"] if a.clean else f"{b['name']} {b['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(lbl, 0, 0.9, 2)
            ty = y - 8 if y - th - 14 >= 0 else y + h + th + 10
            cv2.rectangle(vis, (x, ty - th - 8), (x + tw + 12, ty + 6), col, -1)
            cv2.putText(vis, lbl, (x + 6, ty), 0, 0.9, (0, 0, 0), 2)
    return vis


def classes_in(pred, prob, minpx=300):
    """Print the pixel count, mean confidence and vertical centroid of every class
    present in the frame -- the same listing that 'probe' produces."""
    uid, cnt = np.unique(pred, return_counts=True)
    for i, c in sorted(zip(uid, cnt), key=lambda t: -t[1]):
        if i == 0 or c < minpx:
            continue
        ys = np.nonzero(pred == i)[0]
        print(f"   {i:>3} {C.NAME[int(i)]:<22}{c:>8}px  conf {prob[pred==i].mean():.2f}  "
              f"centroid-y {100*ys.mean()/pred.shape[0]:.0f}%")


if a.mode == "image":
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    for p in [a.video] + list(a.rest):
        fr = cv2.imread(p)
        if fr is None:
            print(f"--- {p}: cannot read (not an image, or wrong path)"); continue
        pred, prob = seg(fr)
        per = {c: boxes(c, pred, prob) for c in CLS}
        vis = draw(fr, pred, prob, per, {c: bool(per[c]) for c in CLS})
        if a.mask:                      # overlay every class mask, ignoring the box thresholds
            m = (pred != 0) & (prob >= a.conf)
            if m.any():
                vis[m] = (0.55 * C.PALETTE[pred[m]] + 0.45 * vis[m]).astype(np.uint8)
        d = a.out or os.path.dirname(os.path.abspath(p))
        fn = f"{d}/{a.tag}{os.path.splitext(os.path.basename(p))[0]}_seg.jpg"
        cv2.imwrite(fn, vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
        json.dump({str(c): per[c] for c in CLS}, open(fn[:-4] + ".json", "w"))
        print(f"--- {os.path.basename(p)}  {fr.shape[1]}x{fr.shape[0]} -> {fn}")
        print("    boxes: " + "  ".join(f"{C.NAME[c]}:{len(per[c])}" for c in CLS))
        classes_in(pred, prob)
    sys.exit()

if a.mode == "probe":
    cap = cv2.VideoCapture(a.video)
    for fi in [int(v) for v in a.rest[0].split(",")]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            print(f"f{fi}: cannot read frame"); continue
        pred, prob = seg(fr)
        print(f"--- f{fi} ---")
        classes_in(pred, prob)
    sys.exit()

if a.mode == "shots":
    os.makedirs(a.out, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    for fi in [int(v) for v in a.rest[0].split(",")]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            print(f"  f{fi}: cannot read frame"); continue
        pred, prob = seg(fr)
        per = {c: boxes(c, pred, prob) for c in CLS}
        fn = f"{a.out}/{a.tag}f{fi:05d}.jpg"
        cv2.imwrite(fn, draw(fr, pred, prob, per, {c: bool(per[c]) for c in CLS}),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        json.dump(per, open(fn[:-4] + ".json", "w"))
        print(f"  {os.path.basename(fn)}  " +
              "  ".join(f"{C.NAME[c]}:{len(per[c])}" for c in CLS))
    cap.release()
    sys.exit()

F0, F1, OUT = int(a.rest[0]), int(a.rest[1]), a.rest[2]
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
cap = cv2.VideoCapture(a.video)
W, H = int(cap.get(3)), int(cap.get(4))
cap.set(cv2.CAP_PROP_POS_FRAMES, F0)
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
hist = {c: collections.deque(maxlen=a.win) for c in CLS}   # temporal confirmation, per class
rec, nhit = {}, {c: 0 for c in CLS}
for fi in range(F0, F1):
    ok, fr = cap.read()
    if not ok:
        break
    pred, prob = seg(fr)
    per = {c: boxes(c, pred, prob) for c in CLS}
    conf = {}
    for c in CLS:
        hist[c].append(bool(per[c]))
        conf[c] = sum(hist[c]) >= a.hit
        nhit[c] += conf[c]
    rec[fi] = {str(c): per[c] for c in CLS if conf[c]}
    vw.write(draw(fr, pred, prob, per, conf))
    if (fi - F0) % 150 == 0:
        print(f"  f{fi}  " + "  ".join(f"{C.NAME[c]}:{len(per[c])}" for c in CLS), flush=True)
vw.release(); cap.release()
json.dump(rec, open(OUT.rsplit(".", 1)[0] + "_boxes.json", "w"))
tot = F1 - F0
print(f"done {OUT}   {tot} frames;  " +
      ";  ".join(f"{C.NAME[c]} {nhit[c]} frames ({100*nhit[c]/max(tot,1):.1f}%)"
                 for c in CLS))
