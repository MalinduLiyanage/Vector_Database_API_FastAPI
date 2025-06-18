from typing import Union

def get_env_mode() -> Union[str, None]:
    import os
    return os.getenv("FAST_ENVIRONMENT")