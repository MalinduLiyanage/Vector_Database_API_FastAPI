from fastapi import FastAPI
from app.configs.app_settings import get_app_settings

settings = get_app_settings()

print("Loaded App Settings:")
print("Host:", settings.app_host)
print("Port:", settings.app_port)
print("Milvus URI:", settings.milvus_uri)
print("Milvus User:", settings.milvus_user)
print("Milvus Password:", settings.milvus_password)
print("Milvus Database:", settings.milvus_database)

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
