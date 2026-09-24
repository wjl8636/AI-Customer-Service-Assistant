"""建库材料清单:哪些文件进知识库、各自算什么内容类型。

一处定义、三处共用——离线建库 CLI(scripts/build_kb.py)、切块预览(scripts/show_kb.py)、
录入页(/kb)。三边各写一份就会出现「预览看到三份、实际入库四份」这种对不上的账。
"""
import pathlib

KB_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "kb"

# 文件 → content_type
SOURCE_TYPES: dict[str, str] = {
    "product-faq.md": "faq",
    "returns-policy.md": "policy",
    "after-sales-manual.md": "manual",
    "product-specs.md": "spec",     # 带具体型号的商品规格,供 BM25 精确词命中
    "member-benefits.md": "policy",  # 会员等级与积分,规则型内容归 policy
    "billing-shipping.md": "policy", # 开票与配送,同上
}

# 录入页允许选的内容类型:政策/手册这类没有天然问题,questions 落章节标题
CONTENT_TYPES: tuple[str, ...] = ("faq", "policy", "manual", "spec")
CONTENT_TYPE_DESC: dict[str, str] = {
    "faq": "商品 FAQ:questions 填真实问法",
    "policy": "政策条款:questions 填章节标题、category 填上级路径",
    "manual": "售后手册:同政策,按标题层级切",
    "spec": "商品规格:含具体型号,精确词召回靠它",
}
