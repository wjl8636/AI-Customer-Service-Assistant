"""读图小注:让模型看着这一轮的数说一句结论,和产物一起落盘。

页面上的「读图」原来是写死的句子。数换了、结论还是那句话,数据一变就容易不对
(比如单均最高那路换成了不走多步工具的意图,写死的「多步调用」那半句就成了错的)。

改成模型生成,但生成时机放在**产物落盘那一刻**,不放在页面渲染时:
- 看板不该依赖上游活着,依赖不齐的时候恰恰最该打开它;
- 同一份产物每次打开都得是同一句话,刷新一次换一种说法不像结论,像瞎猜;
- 重跑的时候数和注一起换,不会出现「新数配旧注」。

模型会胡说,所以两道闸:
1. prompt 只喂这一轮的数,并且明说只准引用给到的数字;
2. 落盘前机械校验——注里出现的每个数字都得在给它的数据里对得上(含四舍五入与
   百分比、余数这几种写法),对不上就整条丢掉,页面退回自己那句兜底话。
   宁可显示旧口径的句子,也不能把编出来的数印在页面上。
"""
import asyncio
import json
import logging
import re

logger = logging.getLogger(__name__)

MAX_CHARS = 130          # 一句到两句的量;再长就不是「读图」而是又一段正文
TIMEOUT = 120.0          # 单条注的等待上限。带思维链的模型一条要跑一分钟出头,
                         # 而这是产物落盘时的一次性开销,不是页面打开时的等待,给宽点值得

# 数字要认,标识符里的数字不能认:p25 是分位名、Recall@10 是指标名、bge-m3 是模型名,
# 它们里面的 25 / 10 / 3 都不是模型下的结论。所以前面挨着字母、@、下划线、连字符或小数点的不算。
_NUM = re.compile(r"(?<![A-Za-z@_.\-\d])\d+(?:,\d{3})*(?:\.\d+)?(?![A-Za-z_])")

# 每类图给模型交代两件事:这张图画的是什么、读者拿它做什么决定
KINDS: dict[str, tuple[str, str]] = {
    "rag_mrr": (
        "四种检索策略(纯向量 / 纯 BM25 / 混合 / 混合+重排)在四类问题桶(政策类 / 型号类 / "
        "口语类 / 跨文档类)以及总体上的 MRR,越高说明正确证据排得越靠前。",
        "读者要判断哪一路检索该上线,以及每一路的短板在哪个桶。",
    ),
    "rag_recall": (
        "同样四策略 × 五桶的 Recall@5,看这题需要的证据在前五条里凑齐了几成;跨文档类一问要两三块不同小节的知识。",
        "读者要判断哪一路会漏证据,漏在哪个桶。",
    ),
    "rag_coverage": (
        "证据覆盖度:召回回来的十条证据里,标准答案的要点有几个在。",
        "读者要判断召回的证据够不够答题,不只是「有没有召回到」。",
    ),
    "rag_answer_coverage": (
        "端到端答案覆盖度:同一套生成提示词,只换检索策略,最终答案覆盖了标准要点的比例。",
        "读者要看检索差会不会一路传导到答案缺要点。",
    ),
    "cost_by_intent": (
        "按意图分堆的 token 账:每条意图的请求数、总 token、单均 token、占总量的比例。"
        "单均高说明一次用户提问背后有多次模型调用(多步工具链),与问题数量无关。",
        "读者要决定先给哪条意图瘦 prompt 或换小模型。",
    ),
    "eval_trend": (
        "评估流水线最近两轮的四个指标(Recall@5、MRR、Faithfulness、拒答率),"
        "以及本轮相对上一轮的涨跌。",
        "读者要判断飞轮写回知识库之后有没有把质量拉下来,该不该去翻最近通过的审核。",
    ),
    "confidence_calibration": (
        "证据置信度阈值扫描:每个候选阈值下,库里有答案的题通过率与库外该拒的题放行率,"
        "以及选定的那条线。可答被误拦的那些会走兜底进问题池,是数据飞轮的燃料。",
        "读者要理解这条线为什么定在这儿,以及往左右挪要付什么代价。",
    ),
}

_RULES = (
    "你在给一个技术看板写「读图」小注,读者是正在学这套系统的开发者。要求:\n"
    "1. 只能引用我给你的数据里出现的数字,一个都不许自己算、不许估、不许编;\n"
    f"2. 全文不超过 {MAX_CHARS} 字,一到两句话,最后落在「所以该看哪儿 / 该做什么」上;\n"
    "3. 中文口语,像同事指着屏幕说话。不用分号,不用破折号,整段最多一个句号;\n"
    "4. 不要复述图上所有数字,挑最说明问题的一两个;\n"
    "5. 策略名、题型名一律用我给的中文标签(比如「口语类」),不要出现 C_colloquial "
    "这种英文字段名;\n"
    "6. 句中停顿用半角逗号「,」,不要用全角「，」,句末用「。」;\n"
    "7. 四位以上的数写千分位(7,942),跟页面表格里的写法一致;\n"
    "8. 直接输出这句话本身,不要加引号、标题、markdown 或任何解释。"
)


def _payload_numbers(payload) -> set[str]:
    """数据里所有数字的可接受写法:原样、常见小数位、百分比、以及比例的余数。

    余数(1-x)也算合法:读图里「代价是 15% 的可答被误拦」正是 1 - 通过率 0.85,
    这是同一个数的另一面,不是新造的数。
    """
    out: set[str] = set()

    def add(x: float) -> None:
        for s in (f"{x:g}", f"{x:.0f}", f"{x:.1f}", f"{x:.2f}", f"{x:.3f}"):
            out.add(s)

    def walk(node) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            add(float(node))
            add(abs(float(node)) * 100)
            if 0.0 <= float(node) <= 1.0:
                add((1.0 - float(node)) * 100)
                add(1.0 - float(node))
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str):
            for m in _NUM.finditer(node):
                try:
                    add(float(m.group(0).replace(",", "")))
                except ValueError:
                    pass

    walk(payload)
    return {s.rstrip("0").rstrip(".") if "." in s else s for s in out}


def verify(text: str, payload) -> bool:
    """注里的每个数字都得在数据里找得到,否则整条不要。"""
    allowed = _payload_numbers(payload)
    for m in _NUM.finditer(text):
        raw = m.group(0).replace(",", "")
        norm = raw.rstrip("0").rstrip(".") if "." in raw else raw
        if norm not in allowed:
            logger.warning("读图小注引用了数据里没有的数 %s,丢弃该注", raw)
            return False
    return True


def tidy(text: str) -> str:
    """排版归一:标点、数字前后的空格、句末,都按页面上那几句兜底话的口径来。

    光靠 prompt 约束模型不稳(全角逗号、句末多个感叹号、数字紧贴汉字都出现过),
    这些又都是机械可判的,所以在代码里定死,注和兜底句放在同一行看不出两种笔法。
    """
    text = " ".join(text.split()).strip().strip("「」\"'")
    text = text.replace("，", ",").replace("；", ",").replace(";", ",")
    text = re.sub(r",\s+", ",", text)
    text = re.sub(r"(?<=[一-鿿])(?=[\dA-Za-z])", " ", text)     # 汉字后紧跟数字或英文
    text = re.sub(r"(?<=[\dA-Za-z%])(?=[一-鿿])", " ", text)    # 数字或英文后紧跟汉字
    # 英文单词后面直接粘个数(总token8821)也拆开:粘着的数会被下面的校验漏掉。
    # 只在三个以上字母之后拆,免得把 p25、bge-m3 这类名字里的数字也拆走
    text = re.sub(r"(?<=[A-Za-z]{3})(?=\d)", " ", text)
    text = text.rstrip("!!??.、,·… ")
    return text if text.endswith(("。", ")", ")")) else text + "。"


async def generate(kind: str, payload, model=None) -> str | None:
    """生成一条读图小注;拿不到、太长、数字对不上,一律回 None 让页面用兜底句。"""
    what, decision = KINDS[kind]
    if model is None:
        from app.core.llm import get_chat_model
        model = get_chat_model()
    prompt = (f"{_RULES}\n\n这张图画的是:{what}\n读者要做的判断:{decision}\n\n"
              f"数据(JSON):\n{json.dumps(payload, ensure_ascii=False)}")
    try:
        resp = await asyncio.wait_for(model.ainvoke(prompt), timeout=TIMEOUT)
    except Exception as e:
        logger.warning("读图小注 %s 生成失败(%s: %s),页面用兜底句", kind, type(e).__name__, e)
        return None
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    text = tidy(text)
    if len(text) <= 1 or len(text) > MAX_CHARS:
        logger.warning("读图小注 %s 长度 %d 不合格,丢弃", kind, len(text))
        return None
    return text if verify(text, payload) else None


async def generate_all(jobs: dict[str, object], model=None) -> dict[str, str]:
    """一批注并发生成;失败的那条直接不进结果,由页面回落兜底句。"""
    kinds = list(jobs)
    notes = await asyncio.gather(*(generate(k, jobs[k], model) for k in kinds))
    return {k: n for k, n in zip(kinds, notes) if n}
