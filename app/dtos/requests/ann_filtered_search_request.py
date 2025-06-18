from pydantic import BaseModel
from typing import List

class ANNFilteredSearchRequest(BaseModel):
    query_vector: List[float]
    top_k: int
    collection_name: str
    field_name: str
    filter: str
    output_fields: List[str]
