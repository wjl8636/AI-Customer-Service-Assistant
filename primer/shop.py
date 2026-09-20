"""几个例子共用的东西：连模型的客户端，和两个查假数据的函数。"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(base_url=os.environ["CHAT_BASE_URL"], api_key=os.environ["CHAT_API_KEY"])
MODEL = os.environ["CHAT_MODEL"]

ORDERS = {
    "SO20260901": {"item": "云朵跑鞋 白色 42码", "status": "已发货", "tracking_no": "SF1234567890"},
}
LOGISTICS = {
    "SF1234567890": {"carrier": "顺丰速运", "status": "运输中，已到杭州转运中心", "eta": "预计 9 月 18 日送达"},
}


def query_order(order_id):
    return ORDERS.get(order_id, {"error": f"没有找到订单 {order_id}"})


def query_logistics(tracking_no):
    return LOGISTICS.get(tracking_no, {"error": f"没有找到快递单 {tracking_no}"})
