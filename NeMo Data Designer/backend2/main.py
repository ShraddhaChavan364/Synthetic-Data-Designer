
# backend/main.py

from dotenv import load_dotenv
from pathlib import Path

# Load .env from parent folder (exactly as requested)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

import os
from typing import List, Dict, Any, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nemo_microservices.data_designer.essentials import NeMoDataDesignerClient
from utils import (
    deep_copy_template,
    validate_columns_config,
    validate_subcat_mapping,
    build_config,
)

# --- FastAPI setup ---
app = FastAPI(title="NeMo Data Designer Backend", version="1.0")

# CORS for local dev: Streamlit frontend -> FastAPI backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Client initialization (same base_url & header style) ---
API_KEY = os.getenv("NEMO_DD_API_KEY")
client = NeMoDataDesignerClient(
    base_url="https://ai.api.nvidia.com/v1/nemo/dd",
    default_headers={"Authorization": f"Bearer {API_KEY}"} if API_KEY else {},
)

# --- Models ---
class ColumnConfig(BaseModel):
    name: str
    type: str
    values: Optional[Any] = None
    weights: Optional[Any] = None
    category: Optional[str] = None
    low: Optional[float] = None
    high: Optional[float] = None
    convert_to: Optional[str] = None
    age_range: Optional[List[int]] = None
    system_prompt: Optional[str] = None
    prompt: Optional[str] = None
    hidden: Optional[bool] = False

class PreviewRequest(BaseModel):
    columns: List[ColumnConfig]
    model_alias: str
    num_records: int

class ValidateRequest(BaseModel):
    columns: List[ColumnConfig]

# --- Health ---
@app.get("/health")
def health():
    return {
        "status": "ok",
        "api_key_present": bool(API_KEY),
    }

# --- Templates (same deep copy semantics) ---
@app.get("/templates/{use_case_key}")
def get_template(use_case_key: str):
    if use_case_key not in {"ecommerce", "healthcare"}:
        raise HTTPException(status_code=400, detail="Invalid use_case_key")
    return deep_copy_template(use_case_key)

# --- Validation (unchanged rules) ---
@app.post("/validate")
def validate(req: ValidateRequest):
    cfg_dicts = [c.dict() for c in req.columns]
    valid, errs = validate_columns_config(cfg_dicts)
    subcat_errs = validate_subcat_mapping(cfg_dicts)
    return {"valid": valid and not subcat_errs, "errors": errs + subcat_errs}

# --- RAW Preview (no sanitization, no derivations) ---
# Frontend will apply sanitizers & derivations exactly as in original file.
@app.post("/preview_raw")
def preview_raw(req: PreviewRequest):
    if not API_KEY:
        raise HTTPException(status_code=400, detail="Missing NEMO_DD_API_KEY")

    cfg_dicts = [c.dict() for c in req.columns]
    valid, errs = validate_columns_config(cfg_dicts)
    mapping_errors = validate_subcat_mapping(cfg_dicts)
    if not valid or mapping_errors:
        raise HTTPException(status_code=400, detail=errs + mapping_errors)

    cb = build_config(cfg_dicts, req.model_alias)
    preview = client.preview(cb, num_records=req.num_records)
    df = preview.dataset
    # Return raw records; frontend will do the exact same processing as original
    return {"rows": len(df), "records": df.to_dict(orient="records")}
