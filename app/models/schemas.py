from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum

class ModelType(str, Enum):
    YOLOv8n = "yolov8n-seg.pt"
    YOLOv8s = "yolov8s-seg.pt"
    YOLOv8m = "yolov8m-seg.pt"
    YOLOv8l = "yolov8l-seg.pt"
    YOLOv8x = "yolov8x-seg.pt"

class DetectionRequest(BaseModel):
    model_type: ModelType = Field(default=ModelType.YOLOv8x, description="نوع مدل")
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="آستانه اطمینان")
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0, description="آستانه IOU")
    visualize: bool = Field(default=True, description="بصری‌سازی نتایج")
    return_image: bool = Field(default=True, description="بازگشت تصویر پردازش شده")

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    area: float

class Detection(BaseModel):
    id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

class DetectionResponse(BaseModel):
    success: bool
    message: str
    model_used: str
    processing_time: float
    total_detections: int
    detections: List[Detection]
    class_distribution: Dict[str, int]
    image_size: Optional[Tuple[int, int]] = None
    visualized_image: Optional[str] = None  # base64
    summary_plot: Optional[str] = None  # base64
    timestamp: datetime = Field(default_factory=datetime.now)

class BatchDetectionRequest(BaseModel):
    urls: List[str] = Field(..., description="لیست URL تصاویر")
    model_type: ModelType = Field(default=ModelType.YOLOv8x)
    confidence_threshold: float = Field(default=0.25)

class BatchDetectionResult(BaseModel):
    url: str
    success: bool
    detections: Optional[List[Detection]] = None
    error: Optional[str] = None
    processing_time: float

class BatchDetectionResponse(BaseModel):
    total_images: int
    successful: int
    failed: int
    total_processing_time: float
    average_processing_time: float
    results: List[BatchDetectionResult]

class ModelInfo(BaseModel):
    name: str
    type: str
    classes: int
    parameters: Optional[str] = None
    size_mb: float
    description: str
    performance: Dict[str, float]