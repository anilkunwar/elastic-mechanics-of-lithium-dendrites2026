"""
Battery Science CSV to JSON Converter
- Reads CSV files from local 'metadatabase' folder (auto-detects battery-science-knowledge-graph paths)
- Accepts user uploads
- Assigns unique IDs to all records for Knowledge Graph node/edge mapping
- Supports deduplication by Model, ID, or other primary keys
- FIXED: Strict JSON compliance — all missing values output as `null` (no bare NaN/Inf)
"""

import streamlit as st
import pandas as pd
import json
import uuid
import os
import math
import numpy as np
from pathlib import Path
from datetime import datetime
from io import StringIO
from typing import List, Tuple, Dict, Any


def sanitize_for_json(obj):
    """
    Recursively replace NaN, Inf, -Inf, numpy NA, pandas NA/NaT with None.
    Ensures output is strictly JSON-compliant (no bare NaN literals).
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.generic):
        if np.issubdtype(type(obj), np.floating) or np.issubdtype(type(obj), np.complexfloating):
            val = float(obj)
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        return obj.item()
    else:
        try:
            import pandas as pd
            if pd.isna(obj) and obj is not None and not isinstance(obj, (str, bool, int, float)):
                return None
        except Exception:
            pass
        return obj


def parse_csv_content(file_content: str, source_filename: str) -> List[Dict[str, Any]]:
    """
    Parse standard CSV content into a list of sanitized dictionaries.
    """
    try:
        # Read CSV, force string dtype to preserve leading zeros, keep_default_na=False prevents auto-conversion to np.nan
        df = pd.read_csv(StringIO(file_content), encoding='utf-8', dtype=str, keep_default_na=False)

        # Replace empty/whitespace-only cells AND common missing-value strings with None
        df = df.replace(r'^\s*$', None, regex=True)
        df = df.replace(['NaN', 'nan', 'NAN', 'NA', 'N/A', 'null', 'NULL', 'None', 'none', ''], None)

        # Final pandas-level sweep: any remaining pandas/numpy NA → None
        df = df.where(df.notna(), None)

        records = df.to_dict(orient='records')

        # CRITICAL: sanitize so no float NaN survives into JSON
        records = [sanitize_for_json(record) for record in records]

        # Add tracking metadata to each record for KG provenance
        for record in records:
            record['unique_id'] = str(uuid.uuid4())
            record['source_file'] = source_filename
            record['import_timestamp'] = datetime.now().isoformat()

        return records
    except pd.errors.EmptyDataError:
        st.warning(f"⚠️ Empty CSV file: {source_filename}")
        return []
    except Exception as e:
        st.warning(f"⚠️ Failed to parse {source_filename}: {type(e).__name__}: {e}")
        return []


def get_csv_files_from_folder(folder_path: str) -> List[Path]:
    """Scan a folder for all .csv files."""
    normalized_path = os.path.normpath(folder_path)

    if not os.path.exists(normalized_path) or not os.path.isdir(normalized_path):
        return []

    csv_files = []
    try:
        for filename in os.listdir(normalized_path):
            if filename.lower().endswith('.csv'):
                file_full_path = os.path.join(normalized_path, filename)
                if os.path.isfile(file_full_path):
                    csv_files.append(Path(file_full_path))
    except PermissionError:
        st.warning(f"⚠️ Permission denied reading folder: {normalized_path}")
        return []

    csv_files.sort(key=lambda p: p.name)
    return csv_files


def find_metadatabase_folder() -> str:
    """Auto-detect the metadatabase folder, prioritizing the battery-science-knowledge-graph structure."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    candidates = [
        os.path.join(script_dir, "battery-science-knowledge-graph", "metadatabase"),
        os.path.join(script_dir, "metadatabase"),
        os.path.join(os.getcwd(), "battery-science-knowledge-graph", "metadatabase"),
        os.path.join(os.getcwd(), "metadatabase"),
        "metadatabase",
    ]

    for path in candidates:
        normalized = os.path.normpath(path)
        if os.path.exists(normalized) and os.path.isdir(normalized):
            has_csv = any(f.lower().endswith('.csv') for f in os.listdir(normalized))
            if has_csv:
                return path

    return os.path.join("metadatabase")


def process_file_from_path(file_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read a CSV file from disk and parse it."""
    filename = file_path.name
    file_str_path = str(file_path)

    try:
        content = None
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
            try:
                with open(file_str_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            raise ValueError("Could not decode file with any supported encoding")

        records = parse_csv_content(content, filename)
        summary = {"total_records": len(records), "status": "OK" if records else "Empty"}
        return records, summary

    except Exception as e:
        st.error(f"❌ Error reading `{filename}`: {type(e).__name__}: {e}")
        return [], {"error": f"{type(e).__name__}: {str(e)}", "total_records": 0, "status": "Error"}


def process_uploaded_files(uploaded_files: List) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Process user-uploaded files."""
    all_records: List[Dict[str, Any]] = []
    file_summaries: Dict[str, Any] = {}

    for uploaded_file in uploaded_files:
        try:
            file_bytes = uploaded_file.read()
            content = None

            try:
                content = file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                for encoding in ['utf-8-sig', 'latin-1', 'cp1252']:
                    try:
                        content = file_bytes.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue

            if content is None:
                raise UnicodeDecodeError("utf-8", file_bytes[:100], 0, 100, "Could not decode")

            records = parse_csv_content(content, uploaded_file.name)
            all_records.extend(records)
            file_summaries[uploaded_file.name] = {"total_records": len(records), "status": "OK" if records else "Empty"}

        except Exception as e:
            st.error(f"❌ Error processing upload `{uploaded_file.name}`: {e}")
            file_summaries[uploaded_file.name] = {"error": str(e), "total_records": 0, "status": "Error"}

    return all_records, file_summaries


def deduplicate_records(records: List[Dict[str, Any]], key_field: str = 'Model') -> Tuple[List[Dict[str, Any]], int]:
    """Remove duplicate records based on a key field (e.g., Model, ID)."""
    if not records:
        return [], 0

    seen = set()
    unique = []
    duplicates_removed = 0

    for record in records:
        # Try the specified key field, fallback to unique_id
        key = str(record.get(key_field, record.get('unique_id', ''))).strip().lower()
        
        if key and key not in seen:
            seen.add(key)
            unique.append(record)
        else:
            duplicates_removed += 1

    return unique, duplicates_removed


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def main():
    st.set_page_config(
        page_title="Battery Science CSV to JSON Converter",
        page_icon="🔋",
        layout="wide",
        menu_items={
            'Get Help': 'https://github.com/anilkunwar/elastic-mechanics-of-lithium-dendrites2026',
            'Report a bug': 'https://github.com/anilkunwar/elastic-mechanics-of-lithium-dendrites2026/issues',
            'About': "### Battery Science CSV to JSON Converter\n\nConvert structured battery research data (e.g., lithium dendrite geometry and structure) into a unified, strict JSON format for Knowledge Graph construction."
        }
    )

    st.title("🔋 Battery Science CSV to JSON Converter")
    st.markdown(
        "**Convert structured research CSVs into unified JSON for Knowledge Graphs**\n\n"
        "**Features:**\n"
        "- ✅ Auto-detect CSV files in `metadatabase` folder\n"
        "- ✅ Upload additional CSV files manually\n"
        "- ✅ Assign unique UUID v4 to every record (ideal for KG nodes)\n"
        "- ✅ Optional deduplication by Model, ID, or custom key\n"
        "- ✅ **Strict JSON compliance**: All missing values safely converted to `null`"
    )

    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("📁 Source Folder")
        default_folder = find_metadatabase_folder()
        folder_path_input = st.text_input(
            "CSV folder path",
            value=default_folder,
            help="Auto-detected from common project locations."
        )

        st.subheader("🔁 Deduplication")
        enable_dedup = st.checkbox("Remove duplicate records", value=False)
        dedup_field = st.selectbox(
            "Primary key for deduplication",
            options=['Model', 'ID', 'Geometry', 'unique_id'],
            index=0,
            disabled=not enable_dedup,
            help="Field to use as primary identifier when removing duplicates"
        )

        st.subheader("🔍 Debug")
        show_debug = st.checkbox("Show path diagnostics", value=False)
        if show_debug:
            resolved_path = os.path.normpath(os.path.join(os.getcwd(), folder_path_input)) if not os.path.isabs(folder_path_input) else os.path.normpath(folder_path_input)
            st.code(f"Resolved: {resolved_path}\nExists: {os.path.exists(resolved_path)}")

    st.divider()

    # Folder scanning
    st.subheader("📁 Folder Source")
    folder_csv_paths = []
    if folder_path_input:
        check_path = os.path.normpath(folder_path_input) if os.path.isabs(folder_path_input) else os.path.normpath(os.path.join(os.getcwd(), folder_path_input))
        if os.path.exists(check_path) and os.path.isdir(check_path):
            folder_csv_paths = get_csv_files_from_folder(check_path)
            if folder_csv_paths:
                st.success(f"✅ Found {len(folder_csv_paths)} CSV file(s) in folder")
            else:
                st.info(f"ℹ️ No CSV files found in `{check_path}`")
        else:
            st.warning(f"⚠️ Folder not found: `{check_path}`")

    # File upload section
    st.subheader("📤 Upload Additional Files")
    uploaded_files = st.file_uploader(
        "Select CSV files to upload (multiple allowed)",
        type=['csv'],
        accept_multiple_files=True,
        help="Merge with folder sources. Ideal for testing new model configurations."
    )

    st.divider()
    if st.button("🚀 Convert to JSON", type="primary", use_container_width=True):
        all_records: List[Dict[str, Any]] = []
        file_summaries: Dict[str, Any] = {}

        total_sources = len(folder_csv_paths) + (len(uploaded_files) if uploaded_files else 0)
        if total_sources == 0:
            st.warning("⚠️ No source files found. Add files to the folder or upload CSV files to continue.")
            st.stop()

        progress_bar = st.progress(0)
        processed_count = 0

        # Process folder files
        if folder_csv_paths:
            for file_path in folder_csv_paths:
                records, summary = process_file_from_path(file_path)
                all_records.extend(records)
                file_summaries[file_path.name] = summary
                processed_count += 1
                progress_bar.progress(processed_count / max(total_sources, 1))

        # Process uploaded files
        if uploaded_files:
            upload_records, upload_summaries = process_uploaded_files(uploaded_files)
            all_records.extend(upload_records)
            file_summaries.update(upload_summaries)
            processed_count += len(uploaded_files)
            progress_bar.progress(1.0)

        # Deduplication step
        if enable_dedup and all_records:
            original_count = len(all_records)
            all_records, removed_count = deduplicate_records(all_records, key_field=dedup_field)
            if removed_count > 0:
                st.info(f"🔁 Deduplication: Removed {removed_count} duplicate(s), {len(all_records)} unique record(s) remaining")
            else:
                st.success(f"🔁 No duplicates found based on '{dedup_field}'")

        if not all_records:
            st.error("❌ No records found in any source files. Please check file formatting and encoding.")
            st.stop()

        # Success summary
        total_records = len(all_records)
        total_json_size = len(json.dumps(all_records, ensure_ascii=False, allow_nan=False))
        st.success(f"✅ **Success!** Processed **{total_records:,} record(s)**")
        st.caption(f"Output size: ~{format_file_size(total_json_size)} (uncompressed JSON)")

        # Preview section tailored for Battery Science KG
        with st.expander("🔍 Preview: First 5 Records", expanded=False):
            for i, record in enumerate(all_records[:5], 1):
                # Prioritize battery-specific columns for preview
                display_keys = ['Model', 'Geometry', 'Ratio', 'Performance', 'Metric', 'Stress', 'Area_Fraction']
                preview_lines = []
                
                for key in display_keys:
                    matched_key = next((k for k in record.keys() if k.lower().replace(' ', '_') == key.lower()), None)
                    if matched_key and record[matched_key]:
                        preview_lines.append(f"**{matched_key}:** `{record[matched_key]}`")
                
                # Fallback: if no known keys, just show first 3 keys
                if not preview_lines:
                    keys_to_show = [k for k in list(record.keys())[:4] if k not in ['unique_id', 'source_file', 'import_timestamp']]
                    preview_lines = [f"**{k}:** `{record[k]}`" for k in keys_to_show if record[k]]
                
                st.markdown(f"**Record {i}** (`{record.get('unique_id', '')[:8]}...`)")
                st.markdown(" | ".join(preview_lines))
                st.divider()

        # JSON output
        st.subheader("💾 Download Combined JSON")
        combined_json = json.dumps(all_records, indent=2, ensure_ascii=False, allow_nan=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_filename = f"battery_kg_data_{timestamp}.json"

        st.download_button(
            label=f"⬇️ Download JSON ({total_records:,} records)",
            data=combined_json,
            file_name=combined_filename,
            mime="application/json",
            use_container_width=True
        )

        # Advanced Export Options
        with st.expander("🔧 Advanced Export Options"):
            col_a, col_b = st.columns(2)
            with col_a:
                minified_json = json.dumps(all_records, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
                st.download_button(
                    label="⬇️ Download Minified JSON",
                    data=minified_json,
                    file_name=f"battery_kg_data_min_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True
                )
            with col_b:
                id_df = pd.DataFrame([{'unique_id': r['unique_id'], 'Model': r.get('Model', 'N/A'), 'source': r.get('source_file')} for r in all_records])
                st.download_button(
                    label="⬇️ Download IDs & Models (CSV)",
                    data=id_df.to_csv(index=False),
                    file_name=f"battery_kg_ids_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    st.divider()
    st.caption(
        f"Battery Science CSV to JSON Converter • "
        f"Working dir: {os.getcwd()} • "
        "All records assigned UUID v4 unique identifiers • Strict JSON compliance enforced"
    )


if __name__ == "__main__":
    main()
