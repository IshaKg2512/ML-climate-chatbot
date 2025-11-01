from __future__ import annotations

import os
from dataclasses import dataclass

from pathlib import Path
from dotenv import load_dotenv, find_dotenv


# Robust .env loading:
# 1) Try CWD (when running via Streamlit)
load_dotenv(find_dotenv(filename=".env", usecwd=True))
# 2) Try a .env next to this file as a fallback
env_nearby = Path(__file__).with_name('.env')
if env_nearby.exists():
    load_dotenv(env_nearby)


@dataclass(frozen=True)
class AppConfig:
    data_gov_in_api_key: str | None
    openai_api_key: str | None
    anthropic_api_key: str | None
    # Curated resource IDs (override via env if needed)
    crop_production_resource_id: str | None
    district_crop_production_resource_id: str | None
    rainfall_resource_id: str | None


def get_config() -> AppConfig:
    return AppConfig(
        data_gov_in_api_key=os.getenv("DATA_GOV_IN_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        crop_production_resource_id=os.getenv(
            "CROP_PRODUCTION_RESOURCE_ID",
            "9ef84268-d588-465a-a308-a864a43d0070",
        ),
        district_crop_production_resource_id=os.getenv(
            "DISTRICT_CROP_PRODUCTION_RESOURCE_ID",
            "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69",
        ),
        rainfall_resource_id=os.getenv("RAINFALL_RESOURCE_ID"),
    )


