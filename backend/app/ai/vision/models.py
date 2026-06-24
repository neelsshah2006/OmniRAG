from pydantic import BaseModel


class VisionResponse(BaseModel):
    description: str
    model: str
