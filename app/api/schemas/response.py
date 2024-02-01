from pydantic import BaseModel
from typing import List, Dict, Union, Any


class ASRResult(BaseModel):
    result: List
    took: float
    message: str