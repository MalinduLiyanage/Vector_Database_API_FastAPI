from fastapi import APIRouter, Depends

from app.dtos.requests.ann_filtered_search_request import ANNFilteredSearchRequest
from app.dtos.requests.ann_search_request import ANNSearchRequest
from app.dtos.requests.hybrid_search_request import HybridSearchRequest
from app.dtos.responses.base_response import BaseResponse
from app.services.vector_database_query_service.vector_database_query_service import MilvusService

router = APIRouter(prefix="/Vector")

@router.post("/annsearch", response_model=BaseResponse)
async def ann_search(
    request: ANNSearchRequest,
    service: MilvusService = Depends()
):
    return await service.ann_search(request)

@router.post("/annfilteredsearch", response_model=BaseResponse)
async def ann_filtered_search(
    request: ANNFilteredSearchRequest,
    service: MilvusService = Depends()
):
    return await service.ann_filtered_search(request)

@router.post("/hybridsearch", response_model=BaseResponse)
async def ann_filtered_search(
    request: HybridSearchRequest,
    service: MilvusService = Depends()
):
    return await service.hybrid_search(request)
