
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
from nemo_microservices.data_designer.essentials import (
    CategorySamplerParams,
    DataDesignerConfigBuilder,
    LLMTextColumnConfig,
    PersonSamplerParams,
    SamplerColumnConfig,
    SamplerType,
    SubcategorySamplerParams,
    UniformSamplerParams,
)

# -----------------------------
# Templates (IDENTICAL)
# -----------------------------
def ecommerce_template() -> List[Dict[str, Any]]:
    return [
        {"name": "channel", "type": "category",
         "values": ["Web", "Mobile App", "Marketplace"], "weights": [5, 3, 2]},
        {"name": "country_region", "type": "category",
         "values": ["INDIA", "USA", "Russia"], "weights": [4, 3, 3]},
        {"name": "product_category", "type": "category",
         "values": ["Electronics", "Fashion", "Home"], "weights": [5, 4, 3]},
        {"name": "product_subcategory", "type": "subcategory", "category": "product_category",
         "values": {
             "Electronics": ["Smartphones", "Laptops", "Headphones"],
             "Fashion": ["Clothing", "Footwear", "Accessories"],
             "Home": ["Furniture", "Decor", "Kitchen"]
         }},
        {"name": "brand", "type": "category",
         "values": ["Generic", "NovaTech", "UrbanWear", "HomeCraft"], "weights": [4, 3, 3, 2]},
        {"name": "price", "type": "uniform", "low": 10, "high": 2000, "convert_to": "float"},
        {"name": "discount_percent", "type": "category",
         "values": [0, 5, 10, 15, 20, 30], "weights": [5, 3, 3, 2, 2, 1]},
        {"name": "quantity", "type": "uniform", "low": 1, "high": 3, "convert_to": "int"},
        {"name": "payment_method", "type": "category",
         "values": ["Card", "UPI", "COD", "Wallet"], "weights": [4, 3, 2, 1]},
        {"name": "customer", "type": "person", "age_range": [18, 70], "hidden": True},
        {"name": "rating", "type": "category",
         "values": [1, 2, 3, 4, 5], "weights": [1, 2, 3, 4, 5]},
        {"name": "delivery_speed", "type": "category",
         "values": ["Same Day", "2–3 Days", "Standard (5–7 Days)"], "weights": [1, 4, 5]},
        {
            "name": "product_title",
            "type": "llm-text",
            "system_prompt": (
                "You are an expert e-commerce copywriter. Return ONLY a realistic, concise product title. "
                "Avoid promotional claims and PII. Use brand + key attribute(s) + subcategory. Keep under 12 words."
            ),
            "prompt": (
                "Category: {{ product_category }} → {{ product_subcategory }}\n"
                "Brand: {{ brand }}\n"
                "Price: {{ price }}\n"
                "Discount: {{ discount_percent }}%\n"
                "Channel: {{ channel }}\n"
                "Region: {{ country_region }}.\n"
                "Constraints: Title only; include relevant attributes; neutral tone."
            ),
        },
        {
            "name": "review_text",
            "type": "llm-text",
            "system_prompt": (
                "Write a realistic, helpful product review. Avoid meta text, PII, and offensive language. "
                "Be specific about use case, pros/cons, and delivery experience. 60–120 words."
            ),
            "prompt": (
                "Product: {{ product_title }}\n"
                "Subcategory: {{ product_subcategory }}\n"
                "Brand: {{ brand }}\n"
                "Channel: {{ channel }}\n"
                "Delivery: {{ delivery_speed }}\n"
                "Rating: {{ rating }}.\n"
                "Include 1–2 positives, 1 area to improve, packaging/delivery note, recommend or not. One paragraph."
            ),
        },
    ]


def healthcare_template() -> List[Dict[str, Any]]:
    return [
        {"name": "facility_region", "type": "category",
         "values": ["AMER", "EMEA", "APAC"], "weights": [4, 3, 3]},
        {"name": "department", "type": "category",
         "values": ["Cardiology", "Endocrinology", "General Practice"], "weights": [3, 3, 4]},
        {"name": "patient", "type": "person", "age_range": [0, 95], "hidden": True},
        {"name": "sex_assigned_at_birth", "type": "category",
         "values": ["Female", "Male"], "weights": [5, 5]},
        {"name": "patient_age", "type": "uniform", "low": 1, "high": 95, "convert_to": "int"},
        {"name": "insurance_provider", "type": "category",
         "values": ["Private", "Government", "Self-pay"], "weights": [5, 3, 2]},
        {"name": "blood_group", "type": "category",
         "values": ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"], "weights": [2,1,2,1,3,2,1,1]},
        {"name": "emergency_contact_relation", "type": "category",
         "values": ["Spouse", "Parent", "Sibling", "Friend"], "weights": [3,3,2,2]},
        {"name": "primary_condition", "type": "category",
         "values": ["Hypertension", "Diabetes", "Asthma"], "weights": [4, 3, 3]},
        {"name": "chief_complaint", "type": "llm-text",
         "system_prompt": (
             "You are a clinical intake nurse. Return ONLY a concise, realistic chief complaint. No PII."
         ),
         "prompt": (
             "Age: {{ patient_age }}; Sex at birth: {{ sex_assigned_at_birth }}; "
             "Department: {{ department }}; Condition: {{ primary_condition }}.\n"
             "Return 1–2 short sentences describing the main symptom/reason for visit."
         )},
        {"name": "clinical_notes", "type": "llm-text",
         "system_prompt": (
             "Write realistic SOAP-style clinical notes. Avoid PII and keep consistent with provided context."
         ),
         "prompt": (
             "Encounter in {{ department }} for {{ primary_condition }}.\n"
             "Patient age: {{ patient_age }}; Sex at birth: {{ sex_assigned_at_birth }}.\n"
             "Include: Subjective (chief complaint), Objective (key exam findings), Assessment, Plan.\n"
             "150–200 words; concise and realistic."
         )},
    ]


# -----------------------------
# Deep copy
# -----------------------------
def deep_copy_template(use_case_key: str) -> List[Dict[str, Any]]:
    tpl = ecommerce_template() if use_case_key == "ecommerce" else healthcare_template()
    return json.loads(json.dumps(tpl))  # deep copy


# -----------------------------
# Validation
# -----------------------------
def validate_columns_config(cfg: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    errors = []
    names = set()
    for col in cfg:
        n = col.get("name")
        t = col.get("type")
        if not n or not t:
            errors.append("Missing name or type.")
            continue
        if n in names:
            errors.append(f"Duplicate column name: {n}")
        names.add(n)
        if t == "category":
            if not col.get("values"):
                errors.append(f"{n} missing values.")
        elif t == "subcategory":
            if not col.get("category"):
                errors.append(f"{n} missing parent category reference.")
            if not isinstance(col.get("values"), dict) or not col.get("values"):
                errors.append(f"{n} requires a non-empty mapping dictionary.")
        elif t == "uniform":
            if col.get("low") is None or col.get("high") is None:
                errors.append(f"{n} missing low/high.")
        elif t == "person":
            if not col.get("age_range") or len(col.get("age_range")) != 2:
                errors.append(f"{n} missing age_range [min, max].")
        elif t == "llm-text":
            if not col.get("system_prompt") or not col.get("prompt"):
                errors.append(f"{n} missing LLM prompts.")
        else:
            errors.append(f"Unsupported type: {t}")
    return (len(errors) == 0), errors


def validate_subcat_mapping(cfg: List[Dict[str, Any]]) -> List[str]:
    errors = []
    cats = {c["name"]: c for c in cfg if c["type"] == "category"}
    for sc in [c for c in cfg if c["type"] == "subcategory"]:
        parent = sc.get("category")
        mapping = sc.get("values") or {}
        parent_vals = cats.get(parent, {}).get("values", [])
        missing = [pv for pv in parent_vals if pv not in mapping]
        if missing:
            errors.append(f"Subcategory '{sc['name']}' missing mappings for: {missing}")
    return errors


# -----------------------------
# Build config
# -----------------------------
def build_config(columns: List[Dict[str, Any]], model_alias_local: str) -> DataDesignerConfigBuilder:
    cb = DataDesignerConfigBuilder()
    for col in columns:
        t = col["type"]
        if t == "category":
            cb.add_column(SamplerColumnConfig(
                name=col["name"], sampler_type=SamplerType.CATEGORY,
                params=CategorySamplerParams(values=col["values"], weights=col.get("weights"))
            ))
        elif t == "subcategory":
            cb.add_column(SamplerColumnConfig(
                name=col["name"], sampler_type=SamplerType.SUBCATEGORY,
                params=SubcategorySamplerParams(category=col["category"], values=col["values"])
            ))
        elif t == "uniform":
            cb.add_column(SamplerColumnConfig(
                name=col["name"], sampler_type=SamplerType.UNIFORM,
                params=UniformSamplerParams(low=col["low"], high=col["high"]),
                convert_to=col.get("convert_to", "float"),
            ))
        elif t == "person":
            cb.add_column(SamplerColumnConfig(
                name=col["name"], sampler_type=SamplerType.PERSON,
                params=PersonSamplerParams(age_range=col["age_range"]),
            ))
        elif t == "llm-text":
            cb.add_column(LLMTextColumnConfig(
                name=col["name"],
                system_prompt=col["system_prompt"],
                prompt=col["prompt"],
                model_alias=model_alias_local,
            ))
        else:
            continue
    return cb
