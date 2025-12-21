from pydantic_settings import BaseSettings
from pydantic import Field
import os


class Settings(BaseSettings):
    APP_NAME: str = "Vision Detection API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")

    DEFAULT_MODEL: str = Field(default="yolov8x-seg.pt", env="DEFAULT_MODEL")
    CONFIDENCE_THRESHOLD: float = Field(default=0.25, env="CONFIDENCE_THRESHOLD")
    IOU_THRESHOLD: float = Field(default=0.45, env="IOU_THRESHOLD")

    UPLOAD_DIR: str = Field(default="static/uploads", env="UPLOAD_DIR")
    RESULTS_DIR: str = Field(default="static/results", env="RESULTS_DIR")
    MODEL_DIR: str = Field(default="models", env="MODEL_DIR")

    API_V1_PREFIX: str = "/api/v1"
    HOST: str = Field(default="127.0.0.1", env="HOST")
    PORT: int = Field(default=8000, env="PORT")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RESULTS_DIR, exist_ok=True)
os.makedirs(settings.MODEL_DIR, exist_ok=True)