"""ch10 权威归并术语表:全系统唯一一份,17 类主题类目。
数据处理、预标、训练、推理、评测、前端全部 import 这里,不许各自抄一份。
类目名与边界说明采用课程 README 归并术语表原文;元组顺序即 label id,训练/推理共用。
severity 是容错档位:归错会带偏补知识优先级的类目从严。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TopicClass:
    name: str
    boundary: str                 # 一句边界说明:什么算这一类,近邻类目靠它划开
    examples: tuple[str, ...]     # 用户的说法示例
    severity: str                 # 容错档位:严 / 中 / 宽


TOPIC_CLASSES: tuple[TopicClass, ...] = (
    TopicClass("退换货", "退货、换货、退款怎么办;修归保修维修,退归这里",
               ("退货", "退款", "退钱", "想退了", "七天无理由还能退不"), "严"),
    TopicClass("物流", "货走到哪了、什么时候送到;运费的钱事归运费",
               ("快递", "发货", "到哪了", "怎么还不动", "海外直邮"), "严"),
    TopicClass("尺码", "大小、码数合不合适",
               ("猫窝买大了", "猫别墅尺寸", "项圈偏码", "适合几斤的猫"), "严"),
    TopicClass("发票", "开票、抬头、报销凭证",
               ("开发票", "发票抬头开错了", "能开增值税发票吗", "合并开票"), "严"),
    TopicClass("质量问题", "商品本身的毛病",
               ("开胶", "破了个洞", "有瑕疵", "猫砂盆电机坏了"), "严"),
    TopicClass("运费", "运费谁出、运费险理赔;管的是钱,货走到哪了归物流",
               ("包邮吗", "退货运费谁承担", "运费险怎么赔"), "中"),
    TopicClass("优惠活动", "券和活动怎么用、能不能叠",
               ("优惠券", "满减", "活动价", "能叠加用吗", "双十一有活动吗"), "中"),
    TopicClass("价保", "买完降价了补不补差价",
               ("刚买就降价了", "能补差价吗", "保价期多久"), "中"),
    TopicClass("支付", "付款环节出的问题",
               ("付不了款", "花呗分期", "扣了两次钱", "货到付款", "数字人民币"), "中"),
    TopicClass("订单修改", "下单之后改信息、取消订单",
               ("改地址", "改电话号码", "订单还能取消吗"), "中"),
    TopicClass("库存补货", "有没有货、什么时候补",
               ("有货吗", "断货了", "什么时候补货", "有现货吗"), "中"),
    TopicClass("商品信息", "材质、功能、用法",
               ("什么材质", "怎么洗", "冻干怎么保存", "废砂盒多久倒", "猫粮怎么选"), "中"),
    TopicClass("保修维修", "保修期限、维修换新;修归这里,退归退换货",
               ("保修多久", "坏了能修吗", "能换新吗"), "中"),
    TopicClass("账号", "登录、绑定、账号安全",
               ("登录不上", "忘了密码", "换绑手机号", "注销账号"), "宽"),
    TopicClass("会员积分", "会员权益、积分怎么用",
               ("积分怎么用", "会员几级", "积分能抵钱吗"), "宽"),
    TopicClass("评价", "评价、晒单的规则",
               ("评价怎么改", "追评在哪写", "晒单有奖励吗"), "宽"),
    TopicClass("其他", "上面都对不上的,先兜底",
               ("闲聊", "转人工", "客服几点上班"), "宽"),
)

TOPIC_NAMES: tuple[str, ...] = tuple(c.name for c in TOPIC_CLASSES)
LABEL2ID: dict[str, int] = {name: i for i, name in enumerate(TOPIC_NAMES)}
ID2LABEL: dict[int, str] = {i: name for i, name in enumerate(TOPIC_NAMES)}
NUM_CLASSES = len(TOPIC_CLASSES)
SEVERITY: dict[str, str] = {c.name: c.severity for c in TOPIC_CLASSES}


def terminology_table() -> str:
    """预标/造数 prompt 用的术语表文本:类目:边界说明(示例)。"""
    return "\n".join(
        f"- {c.name}:{c.boundary}(示例:{'、'.join(c.examples)})" for c in TOPIC_CLASSES
    )
