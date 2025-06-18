from typing import Optional, TypeVar, Generic
from pydantic.generics import GenericModel

T = TypeVar('T')
class BaseResponse(GenericModel, Generic[T]):
    status_code: int = 0
    message: Optional[str] = None
    data: Optional[T] = None