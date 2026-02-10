import glob
import pandas as pd
import os
AMT = "Amount(in USD)"

CANON_COLS = [
    "Startup Name", "Founding Date", "City", "Industry/Vertical",
    "Sub-Vertical", "Founders", "Investors", AMT,
    "Investment Stage",
]
MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

def _parse_year_month(filepath):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    parts = basename.split("_")
    if len(parts) >= 2:
        month = parts[0].lower()
        month = MONTHS.get(month)
        year = int(parts[-1]) if parts[-1].isdigit() else None
        return year, month
    return None, None
def normalize_selected_columns(d):
    for c in d.select_dtypes(include="object").columns:
        d[c] = d[c].astype(str).str.strip()
    return d
def load_data(data_dir="data"):
    paths = glob.glob(os.path.join(data_dir, "*", "*.csv"))
    if not paths:
        paths = glob.glob(os.path.join(data_dir, "*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under: {data_dir}")

    frames = []
    for p in sorted(paths):
        d = pd.read_csv(p)

        if set(d.columns[:len(CANON_COLS)]).issubset({"0", "1", "2", "3", "4", "5", "6", "7", "8"}):
            d = d.loc[:, ~d.columns.str.contains("^Unnamed", case=False)]
            rename_map = {str(i): col for i, col in enumerate(CANON_COLS)}
            d = d.rename(columns=rename_map)

        d = d.drop(columns=[c for c in d.columns if c.isdigit()], errors="ignore")
        d.columns = [c.strip() for c in d.columns]
        d = d.loc[:, ~d.columns.str.contains("^Unnamed", case=False)]

        for c in CANON_COLS:
            if c not in d.columns:
                d[c] = ""

        has_amount = d[AMT].notna()
        has_startup = d["Startup Name"].astype(str).str.strip().ne("") & d["Startup Name"].astype(str).ne("nan")
        has_founders = d["Founders"].astype(str).str.strip().ne("") & d["Founders"].astype(str).ne("nan")
        has_investors = d["Investors"].astype(str).str.strip().ne("") & d["Investors"].astype(str).ne("nan")
        d = d[has_amount & (has_startup | has_founders | has_investors)]
        if d.empty:
            continue

        d = normalize_selected_columns(d)

        

        d[AMT] = (
            d[AMT]
            .astype(str)
            .str.replace(r"[^0-9.]", "", regex=True)
            .replace("", None)
        )
        d[AMT] = pd.to_numeric(d[AMT], errors="coerce")

        MAX_DEAL = 5_000_000_000
        d.loc[d[AMT] > MAX_DEAL, AMT] = None

        year, month = _parse_year_month(p)
        d["_source_file"] = os.path.basename(p).lower()
        d["_source_year"] = year
        d["_source_month"] = month

        for c in d.select_dtypes(include="object").columns:
            d[c] = d[c].str.replace(r"[\u200e\u200f\u200b\u200c\u200d\ufeff\t]", "", regex=True)

        frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    dedup_cols = [c for c in data.columns if c not in ("_source_file", "_source_year", "_source_month")]
    data = data.drop_duplicates(subset=dedup_cols, keep="first")
    return data
df = load_data("data")
df.to_csv('data.csv')