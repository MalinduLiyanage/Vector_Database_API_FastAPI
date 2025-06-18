from pydantic import BaseModel
from typing import List

class ANNSearchRequest(BaseModel):
    query_vector: List[float]
    top_k: int
    collection_name: str
    field_name: str
