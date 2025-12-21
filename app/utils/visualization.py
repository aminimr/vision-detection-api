import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64


class Visualizer:
    """کلاس برای بصری‌سازی نتایج تشخیص"""

    def __init__(self):
        # پالت رنگی برای کلاس‌های مختلف
        self.colors = self._generate_colors(100)

    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """تولید رنگ‌های تصادفی"""
        return [tuple(np.random.randint(0, 256, size=3).tolist()) for _ in range(n)]

    def draw_detections(
            self,
            image: np.ndarray,
            detections: List[Dict],
            masks: List[Dict] = None,
            show_labels: bool = True,
            show_conf: bool = True,
            mask_alpha: float = 0.3
    ) -> np.ndarray:
        """
        رسم باکس‌ها و ماسک‌ها روی تصویر

        Returns:
            تصویر حاوی بصری‌سازی
        """
        result = image.copy()

        for det in detections:
            bbox = det["bbox"]
            class_id = det["class_id"]
            color = self.colors[class_id % len(self.colors)]

            # رسم باکس
            x1, y1, x2, y2 = map(int, [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # رسم ماسک (اگر وجود دارد)
            if masks and det["id"] <= len(masks):
                mask_info = next((m for m in masks if m["detection_id"] == det["id"]), None)
                if mask_info and mask_info["points"]:
                    points = np.array(mask_info["points"], dtype=np.int32)
                    mask_overlay = result.copy()
                    cv2.fillPoly(mask_overlay, [points], color)
                    result = cv2.addWeighted(result, 1 - mask_alpha, mask_overlay, mask_alpha, 0)

            # نوشتن برچسب
            if show_labels:
                label = det["class_name"]
                if show_conf:
                    label += f" {det['confidence']:.2f}"

                # محاسبه سایز متن
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )

                # پس‌زمینه برای متن
                cv2.rectangle(
                    result,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    color,
                    -1
                )

                # نوشتن متن
                cv2.putText(
                    result,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

        return result

    def create_summary_plot(
            self,
            detections: List[Dict],
            save_path: Optional[str] = None
    ) -> str:
        """ایجاد نمودار خلاصه نتایج"""
        if not detections:
            return ""

        # توزیع کلاس‌ها
        class_names = [d["class_name"] for d in detections]
        unique_classes, counts = np.unique(class_names, return_counts=True)

        # ایجاد نمودار
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # نمودار میله‌ای
        bars = axes[0].bar(unique_classes, counts)
        axes[0].set_title("توزیع اشیاء تشخیص داده شده")
        axes[0].set_xlabel("کلاس")
        axes[0].set_ylabel("تعداد")
        axes[0].tick_params(axis='x', rotation=45)

        # رنگ‌آمیزی میله‌ها
        for i, bar in enumerate(bars):
            bar.set_color(np.array(self.colors[i % len(self.colors)]) / 255)

        # نمودار دایره‌ای
        axes[1].pie(counts, labels=unique_classes, autopct='%1.1f%%')
        axes[1].set_title("درصد توزیع")

        plt.tight_layout()

        # ذخیره یا بازگشت به صورت base64
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            # تبدیل به base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return f"data:image/png;base64,{img_str}"

    def image_to_base64(self, image: np.ndarray) -> str:
        """تبدیل تصویر به base64"""
        _, buffer = cv2.imencode('.jpg', image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"


# Singleton instance
visualizer = Visualizer()