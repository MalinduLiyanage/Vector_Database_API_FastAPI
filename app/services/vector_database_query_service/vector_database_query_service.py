# app/services/milvus_service.py

from typing import List, Dict, Any
from fastapi import Depends
from starlette.status import HTTP_200_OK, HTTP_500_INTERNAL_SERVER_ERROR
from app.configs.app_settings import AppSettings, get_app_settings
from app.dtos.requests.ann_search_request import ANNSearchRequest
from app.dtos.responses.base_response import BaseResponse
import logging

# Replace this with your actual Milvus client (e.g., pymilvus)
from pymilvus import Collection, connections, utility

logger = logging.getLogger("milvus_service")


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
            logger.info("ANN search completed successfully.")

            return BaseResponse(
                status_code=HTTP_200_OK,
                message="ANN search completed successfully.",
                data=output
            )

        except Exception as e:
            logger.error(f"ANN search failed: {e}")
            return BaseResponse(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                message=f"ANN search failed: {str(e)}",
                data=None
            )
