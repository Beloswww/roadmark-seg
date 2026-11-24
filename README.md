# roadmark-seg

道路標線**語意分割**推論工具。吃一支行車影片或一張圖片，可以輸出兩種結果：

- **語意分割**（`--render seg`）—— 25 類的逐像素遮罩，不做任何框的後處理。
  可以疊在原畫面上、也可以只出純遮罩，另可存逐像素的 class id 供後續運算使用。
- **偵測框**（`--render box`，預設）—— 把遮罩經過形態學與幾何濾除後轉成偵測框，
  輸出畫好框的影片／圖片和逐幀的框座標 JSON。

> 見 [`model/README.md`](model/README.md)。

---

## repo

1. **前處理** —— 模型圖裡沒有正規化層，外部套 ImageNet mean/std；短邊縮到 1024、寬補成 32 的倍數
2. **遮罩 → 框** —— 二值化 → 閉運算 → 連通元件 → 面積／寬度／長寬比濾除 → 時序確認，**逐類別各用各的參數**
3. **類別間抑制** —— 擋掉把斑馬線最近側那條當成停止線的誤判
4. **命令列介面** —— `video` / `shots` / `probe` / `image` 四種模式

---

## 安裝

```bash
git clone https://github.com/<你的帳號>/roadmark-seg.git
cd roadmark-seg

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 取得模型後放進 model/
cp /path/to/seg_task_ssl_pb9-3-off.onnx model/
```

需要 Python 3.9+。GPU 版需要 CUDA 13 + cuDNN 9（`pip install onnxruntime-gpu` 會一併帶上）。

---

## 執行

```bash
# 1) 先看某一幀模型認出哪些類別(不產圖, 最快)
./run.sh probe video.avi 2300

# 2) 語意分割: 遮罩疊在原畫面上, 附類別色票
./run.sh shots video.avi 2300 --render seg --legend --out ./out

# 3) 語意分割: 只要純遮罩(黑底, 存成 png), 外加逐像素 class id
./run.sh shots video.avi 2300 --render seg --seg-style mask --save-raw --out ./out

# 4) 整段影片的語意分割
./run.sh video video.avi 2280 2380 seg.mp4 --render seg --legend

# 5) 偵測框(預設)
./run.sh video video.avi 2280 2380 out.mp4 --cls SL,CW --clean

# 6) 一般圖片(非影片), 兩種模式都適用
./run.sh image a.jpg b.png --render seg --legend --out ./imgout
```

`probe` 的輸出長這樣：

```
--- f2300 ---
    20 motor waiting zone       39196px  conf 0.85  centroid-y 67%
     2 crosswalk                 8363px  conf 0.85  centroid-y 48%
     3 stop line                 6494px  conf 0.87  centroid-y 52%
    18 slow                      5330px  conf 0.79  centroid-y 98%
```

## 輸出什麼

### `--render seg`（語意分割）

完全不經過框的後處理 —— 每一個高於 `--conf` 的像素都直接上色，這才是模型真正的輸出。

| | 產出 |
|---|---|
| `--seg-style overlay`（預設） | 調色盤以 0.55 疊在原畫面上，存 `.jpg` |
| `--seg-style mask` | 純遮罩、黑底、不含原畫面，存 **`.png`**（純色塊存 jpg 會有壓縮雜訊） |
| `--legend` | 左上角畫出該幀出現的類別色票。25 色遮罩沒有色票根本看不懂 |
| `--save-raw` | 另存 `<名稱>_cls.png`（8-bit，像素值＝class id）與 `<名稱>_conf.png`（0–255 對應信心 0.0–1.0）。`video` 模式會輸出成 `<輸出檔名>_raw/` 底下的 png 序列 |

`_cls.png` 是給後續程式用的，不是給人看的（整張幾乎全黑，因為 class id 都是小數字）：

```python
import cv2, numpy as np
cls = cv2.imread("f02300_cls.png", cv2.IMREAD_UNCHANGED)   # (H, W) uint8
print(np.unique(cls))          # [ 0  2  3  4  5  8  9 11 12 14 18 20 21]
print((cls == 3).sum())        # 6494  <- stop line 的像素數
```

### `--render box`（偵測框，預設）

`video` 模式產出兩個檔：`out.mp4`（畫好框）和 `out_boxes.json`（逐幀框座標）。
每個框含三種形式 —— `bbox`（正交外接框）、`rbox`（最小面積旋轉框）、`poly`（輪廓多邊形）——
因為 175° 魚眼下標線明顯彎曲，正交外接框會過鬆。

---

## 參數

| 參數 | 預設 | 說明 |
|---|---|---|
| `--render` | `box` | `box` 偵測框／`seg` 語意分割／`both` 框疊在分割上 |
| `--seg-style` | `overlay` | `overlay` 疊在原畫面／`mask` 純遮罩黑底 |
| `--legend` | 關 | 畫類別色票 |
| `--save-raw` | 關 | 另存逐像素 class id 與信心圖 |
| `--cls` | 見右 | 類別。吃縮寫（`SL,CW`）、id（`3,2`）或 `all`。**預設值隨模式而定**：`--render seg` 是 `all`，其餘是 `SL` |
| `--clean` | 關 | 只畫框與類別名，不疊半透明遮罩 |
| `--box` | `bbox` | `bbox` / `rbox` / `both` |
| `--all` | 關 | 每類不只留一個框（預設停止線取最前方、斑馬線取面積最大） |
| `--conf` | `0.5` | 信心門檻 |
| `--win` / `--hit` | `6` / `3` | 時序確認：最近 N 幀中出現 M 次才輸出，防閃爍（僅 `video` 模式） |
| `--infer` | `1024` | 推論短邊。訓練解析度未知，1024 是實測 recall 與速度的折衷 |
| `--head` | `fine` | `fine` / `coarse` 兩個輸出頭 |
| `--mask` | 關 | `image` 模式：畫出所有類別的遮罩，不受框門檻影響 |
| `--no-suppress` | 關 | 關閉類別間抑制 |
| `--pick` `--min-area` `--min-w` `--min-ar` `--close-w` `--close-h` | 見下 | 給定則覆寫**所有**類別的後處理參數 |

### 逐類別後處理參數

兩類形態差異極大，實測不可共用（在 `src/segcfg.py` 的 `PRESETS`）：

| 參數 | stop line (3) | crosswalk (2) | 為何不同 |
|---|---|---|---|
| 取哪一塊 | 最前方（底緣最低） | 面積最大 | 斑馬線近距離遮罩會裂成 3 塊，取「最近」只會框到左下一小片 |
| 長寬比下限 | 3.0 | 1.5 | 近距離斑馬線實測 573×288，長寬比僅 2.0 |
| 閉運算核 | 221×15 | 201×201 | 停止線是單一細長線段；斑馬線要縱橫都接合 |
| 面積下限 | 1500 | 4000 | |
| 寬度下限 | 180 | 200 | |

**這些門檻只影響 `--render box`／`both`，`--render seg` 完全不受影響。**
換成一般相機、不同解析度或不同視角時，這些門檻多半要重調，常見症狀是「遮罩有、框是 0」。
先用 `./run.sh image a.jpg --render seg --legend` 確認模型有沒有分割出東西，再調門檻。

### 類別

`model/seg_labels.csv`，25 類（id 0 為 background）：

| id | 縮寫 | 名稱 | | id | 縮寫 | 名稱 |
|---|---|---|---|---|---|---|
| 1 | BJ | box junction | | 13 | RA | right arrow |
| 2 | CW | crosswalk | | 14 | LSA | left straight arrow |
| 3 | SL | stop line | | 15 | RSA | right straight arrow |
| 4 | SSW | solid single white | | 16 | CL | channelizing line |
| 5 | SSY | solid single yellow | | 17 | MP | motor prohibited |
| 6 | SSR | solid single red | | 18 | SLOW | slow |
| 7 | SDW | solid double white | | 19 | MPL | motor priority lane |
| 8 | SDY | solid double yellow | | 20 | MWZ | motor waiting zone |
| 9 | DSW | dashed single white | | 21 | LTB | left turn box |
| 10 | DSY | dashed single yellow | | 22 | MI | motor icon |
| 11 | LA | left arrow | | 23 | BI | bike icon |
| 12 | SA | straight arrow | | 24 | PL | parking lot |

---

## 已知限制

都來自模型本身，不是後處理造成的：

1. **拿不到各類別的原始機率圖。** softmax + argmax 已內建在 ONNX 圖裡，輸出只有
   (預測類別, 該類機率)。因此**無法對單一類別獨立調低門檻，也畫不出 PR 曲線**。
   要解決得請模型作者重新匯出 argmax 之前的 logits。

2. **正規化參數是推測的。** 圖裡沒有正規化層，這裡套 ImageNet mean/std
   (0.485, 0.456, 0.406 / 0.229, 0.224, 0.225)，**未經模型作者確認**；訓練解析度同樣未知。

3. **近距離會把停止線翻成斑馬線。** 車子接近時 cls3 的像素數驟降為 0
   （實測 f2820→f2830：43,619 → 617 → 0 px，同時 cls2 由 16,714 升到 172,286），
   所以停止線的框會在壓線前約 0.3 秒消失。掃過 head(fine/coarse) × 推論解析度(768/1024/1280)
   × 信心門檻(0.3/0.5) 共 12 組，結果完全相同 —— 模型就是不輸出。

4. **停止線 (3) 與機車待轉區 (20) 有混淆。** 56 幀取樣中兩者並存佔 12%，其中 6/7 的 SL
   位在 MWZ 上方，且 MWZ 的像素數是 SL 的 7.6 倍。依台灣路口配置停止線應該更靠近車身，
   與此相反。本工具一律只用 id 3。

---

## GPU 跑不起來

`run.sh` 會自動把 `site-packages/nvidia/*/lib` 掛進 `LD_LIBRARY_PATH`
（pip 裝的 CUDA lib 不在系統路徑裡）。啟動時會印出實際用的 provider：

```
model seg_task_ssl_pb9-3-off.onnx  provider ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

看到 `CUDAExecutionProvider` 排第一就對了。若只有 `CPUExecutionProvider`，程式會明確警告
—— **CPU 約慢 37 倍**（2.4 s/幀 vs 64 ms/幀），而且不會報錯，只是變很慢。

常見原因是 CUDA lib 散在多個環境裡（例如錯誤訊息寫 `libcurand.so.10: cannot open shared
object file`）。找到那個 `.so` 所在的目錄，用環境變數補上：

```bash
SEG_EXTRA_LIB=/path/to/nvidia/curand/lib ./run.sh probe video.avi 100
```

其他可用的環境變數：

| 變數 | 用途 |
|---|---|
| `PYTHON` | 指定直譯器（預設 `python3`） |
| `SEG_ONNX` | 模型路徑（預設 `model/seg_task_ssl_pb9-3-off.onnx`） |
| `SEG_LABELS` | 類別表路徑 |
| `SEG_EXTRA_LIB` | 額外的 CUDA library 目錄，冒號分隔 |

---

## 檔案

```
roadmark-seg/
├── run.sh                    執行入口(處理 GPU library path)
├── requirements.txt
├── model/
│   ├── README.md             模型要放哪裡
│   └── seg_labels.csv        25 類對照表
└── src/
    ├── infer_seg_onnx.py     四種模式 + 繪圖
    └── segcfg.py             路徑、逐類別參數、ONNX 前後處理
```

