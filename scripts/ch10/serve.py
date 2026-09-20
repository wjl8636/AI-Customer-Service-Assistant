# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch10 推理服务:ONNX + FastAPI 独立进程 :8110,轻运行时(onnxruntime + tokenizers,不背 torch)。
起停:make classifier-up / classifier-down(仓库惯例 nohup + pid 文件)。"""
import json
import pathlib

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer

from app.core.taxonomy import TOPIC_NAMES
from scripts.ch10.inference_lib import apply_threshold

DIR = pathlib.Path("data/ch10/onnx")

app = FastAPI(title="mewhelp ch10 topic classifier")
_sess = ort.InferenceSession(str(DIR / "model.onnx"), providers=["CPUExecutionProvider"])
_tok = Tokenizer.from_file(str(DIR / "tokenizer.json"))
_tok.enable_truncation(max_length=128)
_tok.enable_padding(pad_id=0, pad_token="[PAD]")
_threshold: float = json.loads((DIR / "threshold.json").read_text())["threshold"]


class ClassifyIn(BaseModel):
    texts: list[str]


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/classify")
def classify(body: ClassifyIn):
    if not body.texts:
        return {"results": []}
    encs = _tok.encode_batch(body.texts)
    feed = {
        "input_ids": np.array([e.ids for e in encs], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64),
        "token_type_ids": np.array([e.type_ids for e in encs], dtype=np.int64),
    }
    (logits,) = _sess.run(["logits"], feed)
    probs = 1 / (1 + np.exp(-logits))
    preds = apply_threshold(probs, _threshold)   # 与 evaluate 共用同一套阈值+兜底,行为不分叉
    results = []
    for row, pred in zip(probs, preds):
        labels = [TOPIC_NAMES[i] for i in range(len(pred)) if pred[i]]
        results.append({"labels": labels,
                        "scores": {TOPIC_NAMES[i]: round(float(p), 4)
                                   for i, p in enumerate(row)}})
    return {"results": results}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8110)
