from pydantic import BaseModel
from typing import List, Optional

class PredictionRequest(BaseModel):
    country_code: str
    target_series: Optional[str] = "GDP (current US$)"
    exclude_series: Optional[List[str]] = ["GNI, Atlas method (current US$)"]
    years_count: Optional[int] = 5

class UploadResponse(BaseModel):
    file_id: str
    country_code: str
    message: str

class CountryData(BaseModel):
    country_code: str
    filename: str

class PredictionData(BaseModel):
    year: int
    value: Optional[float] = None
    median: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None

class PredictionResponse(BaseModel):
    historical: List[PredictionData]
    forecast: List[PredictionData]
    country: str
    target: str