"""Central configuration via pydantic-settings.

All values can be overridden by environment variables or a .env file.
Variable names are the field names uppercased (e.g. LOG_LEVEL, WEBCAM_INDEX).
"""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from env variables (prefix ``EIDON_``) or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="EIDON_",  # all env vars: EIDON_LOG_LEVEL, EIDON_API_KEY, etc.
        case_sensitive=False,
        extra="ignore",
    )

    # --- Paths ---------------------------------------------------------------
    model_dir: Path = Field(
        default=Path("models"), description="Directory containing ONNX model files"
    )
    gallery_path: Path = Field(
        default=Path("data/gallery/gallery.npz"),
        description="Path to the enrolled-identity gallery archive",
    )

    # --- Model pack ----------------------------------------------------------
    det_model_pack: str = Field(default="buffalo_s", description="InsightFace model pack name")
    det_model_name: str = Field(
        default="det_500m.onnx", description="RetinaFace detector ONNX filename"
    )
    rec_model_name: str = Field(
        default="w600k_mbf.onnx", description="ArcFace recognition ONNX filename"
    )

    # --- Recognition ---------------------------------------------------------
    similarity_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity to accept an identity match",
    )
    det_size_w: int = Field(default=640, gt=0, description="Detector input width")
    det_size_h: int = Field(default=640, gt=0, description="Detector input height")

    # --- Webcam --------------------------------------------------------------
    webcam_index: int = Field(default=0, ge=0, description="OpenCV VideoCapture device index")
    inference_skip_frames: int = Field(
        default=2,
        ge=1,
        description="Run inference every N frames; interpolate bounding boxes in between",
    )

    # --- Performance ---------------------------------------------------------
    max_workers: int = Field(default=4, gt=0, description="Maximum parallel preprocessing workers")
    batch_size: int = Field(default=8, gt=0, description="Embedding inference batch size")

    # --- API -----------------------------------------------------------------
    api_key: str | None = Field(
        default=None,
        repr=False,
        description="X-API-Key header value for protected endpoints. Auth disabled when None.",
    )

    # --- Logging -------------------------------------------------------------
    log_level: str = Field(default="INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR")

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got '{v}'")
        return upper

    # --- Derived properties --------------------------------------------------
    @property
    def det_size(self) -> tuple[int, int]:
        """Detector input resolution as (width, height)."""
        return (self.det_size_w, self.det_size_h)

    @property
    def det_model_path(self) -> Path:
        """Absolute path to the RetinaFace ONNX model file."""
        return self.model_dir / self.det_model_pack / self.det_model_name

    @property
    def rec_model_path(self) -> Path:
        """Absolute path to the ArcFace recognition ONNX model file."""
        return self.model_dir / self.det_model_pack / self.rec_model_name


# Module-level singleton — import this everywhere instead of constructing Settings()
settings = Settings()
