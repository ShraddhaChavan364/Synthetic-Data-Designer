
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import streamlit as st
import requests

API_BASE = os.getenv("NEMO_BACKEND_URL", "http://localhost:8000")

# -----------------------------
# App setup
# -----------------------------
st.set_page_config(page_title="NeMo Data Designer – Generator", layout="wide")
st.title("NeMo Data Designer — Synthetic Data Generator")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")
use_case_label = st.sidebar.selectbox("Use Case", ["E-commerce", "Healthcare"])
use_case = "ecommerce" if use_case_label.startswith("E") else "healthcare"

model_aliases = [
    "nemotron-nano-v2",
    "nemotron-super",
    "mistral-small",
    "gpt-oss-20b",
    "gpt-oss-120b",
    "llama-4-scout-17b",
]
model_alias = st.sidebar.selectbox("Model Alias", model_aliases, index=0)
num_preview = st.sidebar.slider("Preview rows", 5, 100, 20)
num_generate = st.sidebar.slider("Generate rows", 50, 10000, 500)
save_dir = st.sidebar.text_input("Save directory", "outputs")
Path(save_dir).mkdir(parents=True, exist_ok=True)

# -----------------------------
# Templates
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

# -----------------------------
# Healthcare template
# -----------------------------
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
# State Management
# -----------------------------
def deep_copy_template(use_case_key: str) -> List[Dict[str, Any]]:
    tpl = ecommerce_template() if use_case_key == "ecommerce" else healthcare_template()
    return json.loads(json.dumps(tpl))

if "prev_use_case" not in st.session_state:
    st.session_state.prev_use_case = use_case
if use_case != st.session_state.prev_use_case:
    st.session_state.columns_config = deep_copy_template(use_case)
    st.session_state.staging_df = pd.DataFrame()
    st.session_state.generated_df = pd.DataFrame()
    st.session_state.prev_use_case = use_case
    st.rerun()

if "columns_config" not in st.session_state:
    st.session_state.columns_config = deep_copy_template(use_case)
if "staging_df" not in st.session_state:
    st.session_state.staging_df = pd.DataFrame()
if "generated_df" not in st.session_state:
    st.session_state.generated_df = pd.DataFrame()

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
# Sanitizers & Derivations
# -----------------------------
def sanitize_llm_bold(df: pd.DataFrame, cfg: List[Dict[str, Any]]) -> pd.DataFrame:
    llm_cols = [c.get('name') for c in cfg if c.get('type') == 'llm-text']
    for col in llm_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda t: t.replace('**', '') if isinstance(t, str) else t)
    return df

def sanitize_unicode(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    replacements = {
        '\u2013': '-', '\u2014': '-', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2022': '-', '\u2023': '-',
        '\u00a0': ' ', '\u200b': '', '\u2010': '-', '\u2011': '-'
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text

def sanitize_llm_text(df: pd.DataFrame, cfg: List[Dict[str, Any]]) -> pd.DataFrame:
    llm_cols = [c.get('name') for c in cfg if c.get('type') == 'llm-text']
    for col in llm_cols:
        if col in df.columns:
            df[col] = df[col].apply(sanitize_unicode)
            df[col] = df[col].fillna("").astype(str).str.strip()
            df[col] = df[col].apply(lambda t: t if t else "(no content)")
    return df

def add_derived_columns_and_drop_hidden(df: pd.DataFrame, cfg: List[Dict[str, Any]]) -> pd.DataFrame:
    out = df.copy()
    if "customer" in out.columns:
        s = out["customer"]
        out["customer_first_name"] = s.apply(lambda v: v.get("first_name", "") if isinstance(v, dict) else "")
        out["customer_last_name"] = s.apply(lambda v: v.get("last_name", "") if isinstance(v, dict) else "")
        out["customer_email"] = s.apply(lambda v: v.get("email_address", "") if isinstance(v, dict) else "")
        out.drop(columns=["customer"], inplace=True, errors="ignore")
    if "patient" in out.columns:
        s = out["patient"]
        out["patient_first_name"] = s.apply(lambda v: v.get("first_name", "") if isinstance(v, dict) else "")
        out["patient_last_name"] = s.apply(lambda v: v.get("last_name", "") if isinstance(v, dict) else "")
        out["patient_email"] = s.apply(lambda v: v.get("email_address", "") if isinstance(v, dict) else "")
        out["patient_dob"] = s.apply(lambda v: v.get("birth_date", "") if isinstance(v, dict) else "")
        out.drop(columns=["patient"], inplace=True, errors="ignore")
    if "country_region" in out.columns:
        region_to_currency = {"INDIA": "INR", "USA": "USD", "Russia": "RUB"}
        out["currency"] = out["country_region"].map(region_to_currency).fillna("USD")
    for c in ["price", "quantity", "discount_percent"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if {"price", "quantity", "discount_percent"}.issubset(out.columns):
        out["final_price"] = (out["price"] * out["quantity"] * (1 - (out["discount_percent"] / 100.0))).round(2)
        out["final_price"] = out["final_price"].fillna(0)
    return out

def save_all_formats(df, base_dir, base_name):
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(base_dir) / f"{base_name}.csv"
    json_path = Path(base_dir) / f"{base_name}.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records")
    return {"csv": str(csv_path), "json": str(json_path)}

# -----------------------------
# Backend helper
# -----------------------------
def backend_preview_raw(cfg: List[Dict[str, Any]], model_alias_local: str, n: int) -> pd.DataFrame:
    resp = requests.post(
        f"{API_BASE}/preview_raw",
        json={"columns": cfg, "model_alias": model_alias_local, "num_records": n},
        timeout=None,
    )
    data = resp.json()
    return pd.DataFrame(data.get("records", []))

# -----------------------------
# Tabs
# -----------------------------
tab_design, tab_preview, tab_generate, tab_analytics = st.tabs(["Design", "Preview", "Generate", "Analytics"])

# -----------------------------
# Design Tab
# -----------------------------
with tab_design:
    st.subheader("Design Your Schema")
    st.caption(f"Current Use Case: **{use_case_label}** \n Columns: {len(st.session_state.columns_config)}")
    cols_cfg = st.session_state.columns_config
    visible_cols = [c for c in cols_cfg if not c.get("hidden")]
    if not visible_cols:
        st.warning("Schema is empty. Reset to template or add columns.")
    else:
        for idx, col in enumerate(visible_cols):
            with st.expander(f"{col['name']} ({col['type']})", expanded=False):
                t = col["type"]
                if t == "category":
                    vals = st.text_area("Values (comma-separated)",
                                        value=", ".join(map(str, col.get("values", []))),
                                        key=f"vals_{idx}")
                    wts_list = col.get("weights", [])
                    wts = st.text_input("Weights (comma-separated, optional)",
                                        value=", ".join(map(str, wts_list)) if wts_list else "",
                                        key=f"wts_{idx}")
                    if st.button("Save", key=f"save_cat_{idx}"):
                        col["values"] = [v.strip() for v in vals.split(",") if v.strip()]
                        col["weights"] = [int(w.strip()) for w in wts.split(",") if w.strip()] if wts.strip() else None
                        st.success("Saved.")
                elif t == "subcategory":
                    cat_cols = [c["name"] for c in cols_cfg if c.get("type") == "category"]
                    category_default = col.get("category", cat_cols[0] if cat_cols else "")
                    default_index = cat_cols.index(category_default) if category_default in cat_cols else 0
                    category = st.selectbox("Parent category column", cat_cols, index=default_index,
                                             key=f"subcat_parent_{idx}")
                    raw_map = st.text_area("Subcategory mapping JSON",
                                           value=json.dumps(col.get("values", {}), indent=2),
                                           key=f"subcat_vals_{idx}")
                    if st.button("Save", key=f"save_subcat_{idx}"):
                        col["category"] = category
                        col["values"] = json.loads(raw_map or "{}")
                        st.success("Saved.")
                elif t == "uniform":
                    c1, c2 = st.columns(2)
                    with c1:
                        low = st.number_input("Low", value=float(col.get("low", 0)), key=f"unif_low_{idx}")
                    with c2:
                        high = st.number_input("High", value=float(col.get("high", 1)), key=f"unif_high_{idx}")
                    convert_to = st.selectbox("Convert to", ["float", "int"],
                                              index=0 if col.get("convert_to", "float") == "float" else 1,
                                              key=f"unif_conv_{idx}")
                    if st.button("Save", key=f"save_unif_{idx}"):
                        col["low"] = low
                        col["high"] = high
                        col["convert_to"] = convert_to
                        st.success("Saved.")
                elif t == "llm-text":
                    sys_p = st.text_area("System prompt", value=col.get("system_prompt", ""), key=f"llm_sys_{idx}")
                    usr_p = st.text_area("User prompt", value=col.get("prompt", ""), key=f"llm_usr_{idx}")
                    if st.button("Save", key=f"save_llm_{idx}"):
                        col["system_prompt"] = sys_p
                        col["prompt"] = usr_p
                        st.success("Saved.")

                if st.button("Remove column", key=f"remove_{idx}"):
                    orig_index = cols_cfg.index(col)
                    st.session_state.columns_config.pop(orig_index)
                    st.success("Removed.")
                    st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Reset to Template"):
                st.session_state.columns_config = deep_copy_template(use_case)
                st.session_state.staging_df = pd.DataFrame()
                st.session_state.generated_df = pd.DataFrame()
                st.rerun()
        with c2:
            if st.button("Reload schema from template"):
                st.session_state.columns_config = deep_copy_template(use_case)
                st.success("Reloaded schema from template.")

# -----------------------------
# Preview Tab
# -----------------------------
with tab_preview:
    st.subheader("Preview Data")
    if st.button("Run Preview"):
        valid, errs = validate_columns_config(st.session_state.columns_config)
        mapping_errors = validate_subcat_mapping(st.session_state.columns_config)
        if not valid or mapping_errors:
            st.error("Fix schema issues:")
            for e in errs:
                st.write(f"• {e}")
            for e in mapping_errors:
                st.write(f"• {e}")
        else:
            with st.spinner("Generating preview..."):
                try:
                    preview_df = backend_preview_raw(st.session_state.columns_config, model_alias, num_preview)
                    preview_df = sanitize_llm_bold(preview_df, st.session_state.columns_config)
                    preview_df = sanitize_llm_text(preview_df, st.session_state.columns_config)
                    preview_df = add_derived_columns_and_drop_hidden(preview_df, st.session_state.columns_config)
                    st.session_state.staging_df = preview_df
                    st.dataframe(preview_df.head(50))
                except Exception as e:
                    st.error(f"Preview failed: {e}")

# -----------------------------
# Generate Tab
# -----------------------------
with tab_generate:
    st.subheader("Generate Full Dataset")
    if st.button("Run Generate"):
        valid, errs = validate_columns_config(st.session_state.columns_config)
        mapping_errors = validate_subcat_mapping(st.session_state.columns_config)
        if not valid or mapping_errors:
            st.error("Fix schema issues before generation.")
            for e in errs:
                st.write(f"• {e}")
            for e in mapping_errors:
                st.write(f"• {e}")
        else:
            with st.spinner("Generating full dataset..."):
                try:
                    dfs = []
                    remaining = num_generate
                    progress = st.progress(0)
                    BATCH = 100
                    while remaining > 0:
                        n = min(BATCH, remaining)
                        chunk_df = backend_preview_raw(st.session_state.columns_config, model_alias, n)
                        chunk_df = sanitize_llm_bold(chunk_df, st.session_state.columns_config)
                        chunk_df = sanitize_llm_text(chunk_df, st.session_state.columns_config)
                        dfs.append(chunk_df)
                        remaining -= n
                        progress.progress(int(((num_generate - remaining) / num_generate) * 100))

                    full_df = pd.concat(dfs, ignore_index=True)
                    full_df = add_derived_columns_and_drop_hidden(full_df, st.session_state.columns_config)
                    st.session_state.generated_df = full_df

                    paths = save_all_formats(full_df, save_dir, f"{use_case}_synthetic")
                    st.success(f"Generated {len(full_df)} rows.")
                    st.download_button("Download CSV", Path(paths["csv"]).read_bytes(), file_name=Path(paths["csv"]).name)
                    st.download_button("Download JSON", Path(paths["json"]).read_bytes(), file_name=Path(paths["json"]).name)
                except Exception as e:
                    st.error(f"Generate failed: {e}")

# -----------------------------
# Analytics Tab
# -----------------------------
with tab_analytics:
    st.subheader("Analytics")
    df = st.session_state.generated_df if not st.session_state.generated_df.empty else st.session_state.staging_df
    if df.empty:
        st.info("No data available yet.")
    else:
        num_df = df.select_dtypes(include="number")
        obj_df = df.select_dtypes(exclude="number")

        st.markdown("### Numerical Summary")
        if not num_df.empty:
            st.dataframe(num_df.describe().T)
        else:
            st.info("No numerical columns.")

        st.markdown("### Categorical/Text Summary")
        if not obj_df.empty:
            summary_rows = []
            for col in obj_df.columns:
                series = obj_df[col].fillna("")
                top = series.value_counts().head(3)
                summary_rows.append({
                    "column": col,
                    "non_null": int(series.ne("").sum()),
                    "unique": int(series.nunique()),
                    "top_values": ", ".join([f"{k} ({v})" for k, v in top.items()])
                })
            st.dataframe(pd.DataFrame(summary_rows))
        else:
            st.info("No categorical/text columns.")
