import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from loguru import logger
from ultralytics import YOLO
from ..core.config import settings


class ObjectDetector:

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.DEFAULT_MODEL
        self.model = None
        self.class_names = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading model from yolo {self.model_name}...")
            start_time = time.time()

            model_path = Path(settings.MODEL_DIR) / self.model_name
            if model_path.exists():
                self.model = YOLO(str(model_path))
            else:
                self.model = YOLO(self.model_name)

            self.class_names = self.model.names
            load_time = time.time() - start_time
            logger.info(f"model {self.model_name} in {load_time:.2f} loaded")
            logger.info(f"classes using{len(self.class_names)}")

        except Exception as e:
            logger.error(f"error loading model{e}")
            raise

    def detect(
            self,
            image: np.ndarray,
            conf_threshold: Optional[float] = None,
            iou_threshold: Optional[float] = None,
            save: bool = False,
            save_path: Optional[str] = None
    ) -> Dict[str, Any]:

        conf = conf_threshold or settings.CONFIDENCE_THRESHOLD
        iou = iou_threshold or settings.IOU_THRESHOLD

        try:
            results = self.model(
                image,
                conf=conf,
                iou=iou,
                save=save,
                save_dir=save_path
            )

            result = results[0]
            detection_data = self._process_results(result)

            detection_data.update({
                "model": self.model_name,
                "confidence_threshold": conf,
                "iou_threshold": iou,
                "image_size": image.shape[:2],
                "processing_time": detection_data.get("processing_time", 0)
            })

            return detection_data

        except Exception as e:
            logger.error(f"{e}")
            raise

    def _process_results(self, result) -> Dict[str, Any]:
        """پردازش نتایج مدل"""
        start_time = time.time()

        detections = []
        masks_data = []

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()

            for i, (mask, box) in enumerate(zip(masks, boxes)):
                x1, y1, x2, y2, conf, cls = box[:6]

                detection = {
                    "id": i + 1,
                    "class_id": int(cls),
                    "class_name": self.class_names[int(cls)],
                    "confidence": float(conf),
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1),
                        "area": float((x2 - x1) * (y2 - y1))
                    },
                    "mask": mask.tolist() if hasattr(mask, 'tolist') else None
                }
                detections.append(detection)

                # استخراج نقاط ماسک
                mask_points = self._extract_mask_points(mask)
                masks_data.append({
                    "detection_id": i + 1,
                    "points": mask_points
                })

        processing_time = time.time() - start_time

        return {
            "detections": detections,
            "masks": masks_data,
            "total_detections": len(detections),
            "processing_time": processing_time,
            "class_distribution": self._get_class_distribution(detections)
        }

    def _extract_mask_points(self, mask: np.ndarray) -> List[List[float]]:
        mask_resized = cv2.resize(mask, (mask.shape[1], mask.shape[0]))
        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            mask_binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            epsilon = 0.001 * cv2.arcLength(contours[0], True)
            approx = cv2.approxPolyDP(contours[0], epsilon, True)
            return [point[0].tolist() for point in approx]

        return []

    def _get_class_distribution(self, detections: List[Dict]) -> Dict[str, int]:
        distribution = {}
        for det in detections:
            class_name = det["class_name"]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution


# Singleton instance
detector = ObjectDetector()