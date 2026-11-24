# 模型權重放這裡

`seg_task_ssl_pb9-3-off.onnx`（約 58 MB），放到這個資料夾底下：

```
model/seg_task_ssl_pb9-3-off.onnx
```

或是放在別處，用環境變數指定：

```bash
SEG_ONNX=/path/to/model.onnx ./run.sh probe video.mp4 100
```

## 模型規格

| 項目 | 內容 |
|---|---|
| 架構 | SegFormer (MiT-B1, embed 64/128/320/512, decoder 256d) + 師生半監督 |
| 輸出頭 | `coarse_seg_head` / `fine_seg_head`，各 25 類；本專案用 `fine` |
| 輸入 | `input`，NCHW float32，四個維度全動態 |
| 輸出 | `coarse_preds` / `coarse_probs` / `fine_preds` / `fine_probs` |
| 匯出 | pytorch 2.7.0、opset 17、1336 節點 |

`seg_labels.csv` 是類別對照表（id / 名稱 / 縮寫 / 語意色）。**ONNX 檔本身不帶任何
metadata**（實測 `metadata_props` 為空），所以類別名稱、正規化參數、訓練解析度都得從外部提供。
