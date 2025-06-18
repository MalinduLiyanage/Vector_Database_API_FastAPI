from typing import List, Dict, Any
from fastapi import Depends
from starlette.status import HTTP_200_OK, HTTP_422_UNPROCESSABLE_ENTITY
from app.configs.app_settings import AppSettings, get_app_settings
from app.dtos.requests.ann_search_request import ANNSearchRequest
from app.dtos.requests.ann_filtered_search_request import ANNFilteredSearchRequest
from app.dtos.requests.hybrid_search_request import HybridSearchRequest
from app.dtos.responses.base_response import BaseResponse
from pymilvus import Collection, connections, AnnSearchRequest, WeightedRanker


class MilvusService:
    def __init__(self, settings: AppSettings = Depends(get_app_settings)):
        self.settings = settings
        self._client_initialized = False

    def _init_client(self):
        if not self._client_initialized:
            connections.connect(
                alias = "default",
                host = self.settings.milvus_host,
                port = self.settings.milvus_port,
                user = self.settings.milvus_user,
                password = self.settings.milvus_password,
                db_name = self.settings.milvus_database,
            )
            self._client_initialized = True

    async def ann_search(self, request: ANNSearchRequest) -> BaseResponse[List[Dict[str, Any]]]:
        try:
            self._init_client()
            collection = Collection(name=request.collection_name)
            collection.load()

            vectors = [request.query_vector]

            results = collection.search(
                data=vectors,
                anns_field=request.field_name,
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=request.top_k,
                output_fields=[]
            )

            output = []
            for hit in results[0]:
                output.append({
                    "record_id": hit.id,
                    "score": hit.distance
                })

            collection.release()

            return BaseResponse(
                status_code=HTTP_200_OK,
                message="ANN search completed successfully.",
                data=output
            )

        except Exception as e:
            return BaseResponse(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                message=f"ANN search failed: {str(e)}",
                data=None
            )

    async def ann_filtered_search(self, request: ANNFilteredSearchRequest) -> BaseResponse[List[Dict[str, Any]]]:
        try:
            self._init_client()
            collection = Collection(name=request.collection_name)
            collection.load()

            vectors = [request.query_vector]

            results = collection.search(
                data=vectors,
                anns_field=request.field_name,
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=request.top_k,
                expr=request.filter,
                output_fields=request.output_fields
            )

            output = []
            for hit in results[0]:
                record = {
                    "record_id": hit.id,
                }
                if hit.entity:
                    record.update(hit.entity)
                output.append(record)

            collection.release()

            return BaseResponse(
                status_code=HTTP_200_OK,
                message="Filtered ANN search completed successfully.",
                data=output
            )

        except Exception as e:
            return BaseResponse(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                message=f"Filtered ANN search failed: {str(e)}",
                data=None
            )

    async def hybrid_search(self, request: HybridSearchRequest) -> BaseResponse[List[Dict[str, Any]]]:
        try:
            self._init_client()

            collection = Collection(name=request.collection_name)
            collection.load()

            if len(request.query_vectors) != len(request.weights):
                return BaseResponse(
                    status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                    message="Number of query vectors and weights must match.",
                    data=None
                )

            search_requests = []
            for vector in request.query_vectors:
                search_requests.append(
                    AnnSearchRequest(
                        data=[vector],
                        anns_field=request.vector_field,
                        param={"metric_type": request.metric_type},
                        limit=request.per_query_limit,
                        expr=request.filter
                    )
                )

            result = collection.hybrid_search(
                reqs=search_requests,
                rerank=WeightedRanker(*request.weights),
                limit=request.combined_limit,
                output_fields=request.output_fields or [],
                consistency_level=request.consistency_level
            )

            output = []
            for hits in result:
                for hit in hits:
                    record = {
                        "record_id": hit.id,
                        "score": hit.distance
                    }
                    if hit.entity:
                        record.update(hit.entity)
                    output.append(record)

            collection.release()

            return BaseResponse(
                status_code=HTTP_200_OK,
                message="Hybrid ANN search completed successfully.",
                data=output
            )

        except Exception as e:
            return BaseResponse(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                message=f"Hybrid ANN search failed: {str(e)}",
                data=None
            )