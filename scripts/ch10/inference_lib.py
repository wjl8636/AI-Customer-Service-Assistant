# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch10 推理纯函数:阈值应用与空标签兜底。evaluate 与 serve 共用,行为不许分叉。
轻依赖(仅 numpy),serve 的 ONNX 轻运行时可安全 import。"""
import numpy as np


def apply_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    """sigmoid 概率 → 0/1 多标签矩阵:过线即命中;某行全不过线取最高分兜底,不产出空标签。"""
    preds = (probs >= threshold).astype(int)
    for r in range(len(preds)):
        if preds[r].sum() == 0:
            preds[r][probs[r].argmax()] = 1
    return preds
