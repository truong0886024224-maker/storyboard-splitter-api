from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class PanelBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    w: int
    h: int


class SceneItem(BaseModel):
    scene: int
    fileName: str
    mimeType: str
    width: int
    height: int
    ratio: str
    url: str
    base64: Optional[str]
    box: Dict[str, Any]


class SplitResponse(BaseModel):
    total: int
    width: int
    height: int
    ratio: str
    mode: str
    scenes: List[SceneItem]
