from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional
import uuid
from pathlib import Path
import shutil

from ..services.detector import detector
from ..services.image_processor import image_processor
from ..utils.visualization import visualizer
from ..models.schemas import (
    DetectionRequest, DetectionResponse, Detection,
    BatchDetectionRequest, BatchDetectionResponse,
    ModelInfo, ModelType
)
from ..core.config import settings

router = APIRouter(prefix=settings.API_V1_PREFIX)


@router.get("/health", tags=["Health"])
async def health_check():
    """بررسی سلامت API"""
    return {
        "status": "healthy",
        "service": "Vision Detection API",
        "version": settings.APP_VERSION,
        "model_loaded": detector.model is not None
    }


@router.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def get_available_models():
    """دریافت لیست مدل‌های موجود"""
    models_info = [
        ModelInfo(
            name="YOLOv8 Nano",
            type="Segmentation",
            classes=80,
            size_mb=6.2,
            description="سبک‌ترین مدل - مناسب برای دستگاه‌های محدود",
            performance={"speed": 95, "accuracy": 68.5}
        ),
        ModelInfo(
            name="YOLOv8 Small",
            type="Segmentation",
            classes=80,
            size_mb=21.5,
            description="تعادل بین سرعت و دقت",
            performance={"speed": 85, "accuracy": 72.3}
        ),
        ModelInfo(
            name="YOLOv8 Medium",
            type="Segmentation",
            classes=80,
            size_mb=49.7,
            description="مناسب برای کاربردهای عمومی",
            performance={"speed": 70, "accuracy": 76.2}
        ),
        ModelInfo(
            name="YOLOv8 Large",
            type="Segmentation",
            classes=80,
            size_mb=87.7,
            description="دقت بالا برای کاربردهای حساس",
            performance={"speed": 55, "accuracy": 78.9}
        ),
        ModelInfo(
            name="YOLOv8 XLarge",
            type="Segmentation",
            classes=80,
            size_mb=130.4,
            description="قوی‌ترین مدل - بیشترین دقت",
            performance={"speed": 40, "accuracy": 80.3}
        )
    ]
    return models_info


@router.post("/detect/upload", response_model=DetectionResponse, tags=["Detection"])
async def detect_from_upload(
        file: UploadFile = File(...),
        config: DetectionRequest = Depends(),
        background_tasks: BackgroundTasks = None
):
    """
    تشخیص اشیاء از آپلود فایل

    - **file**: تصویر ورودی
    - **config**: تنظیمات تشخیص
    """
    try:
        # بررسی نوع فایل
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "فایل باید تصویر باشد")

        # خواندن فایل
        contents = await file.read()

        # بارگذاری تصویر
        image = image_processor.load_image(contents)

        # اعتبارسنجی سایز
        image_processor.validate_image_size(image)

        # تشخیص اشیاء
        results = detector.detect(
            image=image,
            conf_threshold=config.confidence_threshold,
            iou_threshold=config.iou_threshold
        )

        # بصری‌سازی اگر درخواست شده
        visualized_image = None
        summary_plot = None

        if config.visualize and results["detections"]:
            # رسم تشخیص‌ها
            viz_image = visualizer.draw_detections(
                image=image,
                detections=results["detections"],
                masks=results.get("masks", [])
            )

            if config.return_image:
                visualized_image = visualizer.image_to_base64(viz_image)

            # ایجاد نمودار خلاصه
            summary_plot = visualizer.create_summary_plot(results["detections"])

            # ذخیره در پس‌زمینه
            if background_tasks:
                filename = f"{uuid.uuid4()}.jpg"
                save_path = Path(settings.RESULTS_DIR) / filename
                image_processor.save_image(viz_image, save_path)

        # تبدیل به مدل پاسخ
        detections = [
            Detection(
                id=d["id"],
                class_id=d["class_id"],
                class_name=d["class_name"],
                confidence=d["confidence"],
                bbox=d["bbox"]
            )
            for d in results["detections"]
        ]

        response = DetectionResponse(
            success=True,
            message="تشخیص با موفقیت انجام شد",
            model_used=config.model_type.value,
            processing_time=results["processing_time"],
            total_detections=results["total_detections"],
            detections=detections,
            class_distribution=results["class_distribution"],
            image_size=results["image_size"],
            visualized_image=visualized_image,
            summary_plot=summary_plot
        )

        return response

    except Exception as e:
        raise HTTPException(500, f"خطا در پردازش: {str(e)}")


@router.post("/detect/url", tags=["Detection"])
async def detect_from_url(
        url: str,
        config: DetectionRequest = Depends()
):
    pass


@router.post("/detect/batch", response_model=BatchDetectionResponse, tags=["Batch"])
async def batch_detection(
        request: BatchDetectionRequest,
        background_tasks: BackgroundTasks
):
    pass


@router.get("/results/{filename}", tags=["Results"])
async def get_result_image(filename: str):
    file_path = Path(settings.RESULTS_DIR) / filename
    if not file_path.exists():
        raise HTTPException(404, "فایل یافت نشد")
    return FileResponse(file_path)


@router.post("/model/switch", tags=["Models"])
async def switch_model(model_type: ModelType):
    try:
        global detector
        detector = detector.__class__(model_type.value)
        return {"message": f"مدل به {model_type.value} تغییر یافت"}
    except Exception as e:
        raise HTTPException(500, f"خطا در تغییر مدل: {e}")