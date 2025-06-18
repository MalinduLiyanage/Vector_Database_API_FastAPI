from pydantic_settings import  BaseSettings
from app.utils.env_utils import get_env_mode

class AppSettings(BaseSettings):
    # default values
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    milvus_uri: str = ""
    milvus_user: str = ""
    milvus_password: str = ""
    milvus_database: str = ""

    class Config:
        extra = "ignore"
        env_file = ".env" # default env file. can be override dynamically

        # read env configurations from specified environment file
        if get_env_mode() is not None:
            print("get_env_mode")
            env_file = f".env.{get_env_mode()}"
            print("ENV MODE", env_file)

def get_app_settings()->AppSettings:
    return AppSettings()