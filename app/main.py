from fastapi import FastAPI
from app.configs.app_settings import get_app_settings
from app.api.v1.endpoints import vector_db_endpoints

settings = get_app_settings()

print("Loaded App Settings:")
print("Host:", settings.app_host)
print("Port:", settings.app_port)
print("Milvus Host:", settings.milvus_host)
print("Milvus Port:", settings.milvus_port)
print("Milvus User:", settings.milvus_user)
print("Milvus Password:", settings.milvus_password)
print("Milvus Database:", settings.milvus_database)

app = FastAPI()

app.include_router(vector_db_endpoints.router, prefix="/api")
