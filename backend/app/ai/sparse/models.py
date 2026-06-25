from pydantic import BaseModel


class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]
