from pydantic import BaseModel


class ApproveRequest(BaseModel):
    approved_answer: str
