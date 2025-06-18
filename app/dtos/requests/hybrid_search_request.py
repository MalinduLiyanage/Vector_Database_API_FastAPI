from pydantic import BaseModel
from typing import List, Optional

class HybridSearchRequest(BaseModel):
    query_vectors: List[List[float]]
    weights: List[float]
    collection_name: str
    vector_field: str
    per_query_limit: int
    combined_limit: int
    filter: Optional[str] = None
    output_fields: Optional[List[str]] = None
    metric_type: str
    consistency_level: str
