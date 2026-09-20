# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch10 阈值应用与兜底纯函数(evaluate/serve 共用,行为不许分叉)。"""
import numpy as np

from scripts.ch10.inference_lib import apply_threshold


def test_over_threshold_multi_hit():
    probs = np.array([[0.9, 0.85, 0.1]])
    assert apply_threshold(probs, 0.45).tolist() == [[1, 1, 0]]


def test_all_below_falls_back_to_argmax():
    probs = np.array([[0.2, 0.05, 0.31]])
    assert apply_threshold(probs, 0.45).tolist() == [[0, 0, 1]]


def test_boundary_value_hits():
    probs = np.array([[0.45, 0.449, 0.0]])
    assert apply_threshold(probs, 0.45).tolist() == [[1, 0, 0]]


def test_batch_rows_independent():
    probs = np.array([[0.9, 0.1], [0.1, 0.2]])
    out = apply_threshold(probs, 0.45)
    assert out.tolist() == [[1, 0], [0, 1]]   # 第二行走兜底
