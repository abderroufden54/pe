"""
PE Intelligence Agent
Conversational AI for Private Equity analysts.
Uses LangChain + Groq to reason over startup funding data.

Run:
    python agent.py              # demo
    python agent.py -i           # interactive
    streamlit run app.py         # web UI
"""

import os
import re
import pandas as pd
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

AMT = "Amount(in USD)"

CANON_COLS = [
    "Startup Name", "Founding Date", "City", "Industry/Vertical",
    "Sub-Vertical", "Founders", "Investors", AMT,
    "Investment Stage",
]

DISPLAY_COLS = [
    "Startup Name", "Founding Date", "City", "Industry/Vertical",
    "Sub-Vertical", "Founders", "Investors", AMT,
    "Investment Stage", "_source_year", "_source_month", "_country", "_continent",
]


def _truncate(text, max_chars=3000):
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n... (truncated)"
    return text


# load cleaned dataset
df = pd.read_csv("data.csv")


# -------------------------------------------------------------------
# entity detection
# -------------------------------------------------------------------

def detect_entity(name):
    """Check if name exists in startup, founder, investor, or sub-vertical columns."""
    results = []
    columns_to_search = {
        "Startup Name": ["Startup Name", "City", "Industry/Vertical"],
        "Founders":     ["Startup Name", "Founders"],
        "Investors":    ["Startup Name", "Investors", "Industry/Vertical",
                         AMT, "Investment Stage"],
        "Sub-Vertical": ["Sub-Vertical", "Industry/Vertical"],
    }
    for col, display_cols in columns_to_search.items():
        valid_cols = [c for c in display_cols if c in df.columns]
        mask = df[col].str.contains(name, case=False, na=False)
        if mask.any():
            results.append(f"-- Found in {col} --")
            results.append(df[mask][valid_cols].head(1).to_string(index=False))
            results.append("")

    if results:
        return True, f"FOUND IN DATASET: '{name}'\n\n" + "\n".join(results)
    else:
        return False, (
            f"NOT FOUND IN DATASET: '{name}'\n"
            f"Searched: Startup Name, Founders, Investors, Sub-Vertical\n"
            f"Use web_search to find external information about '{name}'."
        )


# -------------------------------------------------------------------
# tools
# -------------------------------------------------------------------

@tool
def lookup_entity(entity_name: str) -> str:
    """Check if a startup, founder, investor, or sub-vertical exists in the dataset.
    Always call this first when the user asks about a specific entity by name.
    Returns:
    - FOUND: entity data from the dataset
    - NOT FOUND: you MUST then call web_search
    Args:
        entity_name: Name to look up (e.g. "Razorpay", "Kunal Shah", "Sequoia")
    """
    found, details = detect_entity(entity_name)
    return details


class AnalyzeArgs(BaseModel):
    operation: str = Field(description="One of: top, bottom, group, filter, count, sum, unique, trend, investor_network, landscape, crosstab, stats")
    column: Optional[str] = Field(default=None, description="Column to operate on. Use 'Amount(in USD)' for numeric ops. Use category columns for count/unique/crosstab. Not needed for filter/landscape/stats.")
    group_by: Optional[str] = Field(default=None, description="Column to group by (for group, trend, crosstab)")
    filters: Optional[list] = Field(
        default=None,
        description="List of filter dicts. One dict=AND, multiple dicts=OR. String=contains, number=exact, list=OR, dict{min/max/gt/lt}=range, dict{not: val}=negation. e.g. [{'City':'Bangalore','_source_year':2021}]"
    )
    top_n: Optional[int] = Field(default=10, description="Number of results to return")
    sort_by: Optional[str] = Field(default=None, description="Sort by: 'total', 'count', or 'average'. Default: 'total")
    ascending: Optional[bool] = Field(default=False, description=""Sort ascending (True) or descending (False). Default: False")


def _apply_filters(result_df, filters, allowed_cols):
    """Apply filter dicts to dataframe. Returns filtered df or error string."""
    if not filters or not isinstance(filters, list):
        return result_df

    masks = []
    for condition in filters:
        mask = pd.Series(True, index=result_df.index)
        for k, v in condition.items():
            if k not in allowed_cols:
                continue

            # 1. Negation: {"City": {"not": "Mumbai"}}
            if isinstance(v, dict) and "not" in v:
                neg_val = v["not"]
                if isinstance(neg_val, list):
                    pattern = "|".join(str(x) for x in neg_val)
                    mask &= ~result_df[k].astype(str).str.contains(pattern, case=False, na=False)
                elif isinstance(neg_val, (int, float)):
                    mask &= result_df[k] != neg_val
                else:
                    mask &= ~result_df[k].astype(str).str.contains(neg_val, case=False, na=False)

            # 2. Numeric range: {"min": 50M, "max": 100M, "gt": 10M, "lt": 50M}
            elif isinstance(v, dict) and any(op in v for op in ("min", "max", "gt", "lt")):
                if not pd.api.types.is_numeric_dtype(result_df[k]):
                    return f"Cannot apply range filter: '{k}' is not numeric."
                for op in ("min", "max", "gt", "lt"):
                    if op in v and not isinstance(v[op], (int, float)):
                        return f"Range value for '{op}' must be a number, got: {v[op]}"
                if "min" in v:
                    mask &= result_df[k] >= v["min"]
                if "max" in v:
                    mask &= result_df[k] <= v["max"]
                if "gt" in v:
                    mask &= result_df[k] > v["gt"]
                if "lt" in v:
                    mask &= result_df[k] < v["lt"]

            # 3. Item count: {"Founders": {"min_items": 3}}
            elif isinstance(v, dict) and any(op in v for op in ("min_items", "max_items")):
                if k not in ("Investors", "Founders"):
                    return f"min_items/max_items only works on Investors or Founders, not '{k}'"
                item_counts = result_df[k].fillna("").str.split(",").apply(
                    lambda x: len([i for i in x if i.strip() and i.strip() != "nan"])
                )
                if "min_items" in v:
                    mask &= item_counts >= v["min_items"]
                if "max_items" in v:
                    mask &= item_counts <= v["max_items"]

            # 4. String length: {"Startup Name": {"min_len": 4}}
            elif isinstance(v, dict) and any(op in v for op in ("min_len", "max_len")):
                str_lens = result_df[k].fillna("").astype(str).str.len()
                if "min_len" in v:
                    mask &= str_lens >= v["min_len"]
                if "max_len" in v:
                    mask &= str_lens <= v["max_len"]

            # 5. Starts/ends with: {"Startup Name": {"starts_with": "A"}}
            elif isinstance(v, dict) and any(op in v for op in ("starts_with", "ends_with")):
                col_str = result_df[k].fillna("").astype(str)
                if "starts_with" in v:
                    mask &= col_str.str.lower().str.startswith(v["starts_with"].lower())
                if "ends_with" in v:
                    mask &= col_str.str.lower().str.endswith(v["ends_with"].lower())

            # 6. Exact match: {"City": {"exact": "New Delhi"}}
            elif isinstance(v, dict) and "exact" in v:
                mask &= result_df[k].fillna("").astype(str).str.lower() == v["exact"].lower()

            # 7. Unknown dict → error
            elif isinstance(v, dict):
                return f"Unsupported filter for '{k}': {v}"

            # 8. OR list: ["Bangalore", "Mumbai"]
            elif isinstance(v, list):
                mask &= result_df[k].astype(str).str.contains(
                    "|".join(str(x) for x in v), case=False, na=False
                )

            # 9. Numeric exact: 2021
            elif isinstance(v, (int, float)):
                mask &= result_df[k] == v

            # 10. String contains: "Bangalore"
            else:
                mask &= result_df[k].astype(str).str.contains(str(v), case=False, na=False)

        masks.append(mask)

    combined = masks[0]
    for m in masks[1:]:
        combined |= m
    return result_df[combined]


def analyze_dataset_function(operation, column=None, group_by=None,
                             filters=None, ascending=False,
                             sort_by="total", top_n=10):
    ALLOWED_COLS = CANON_COLS + ["_source_year", "_source_month", "_country", "_continent"]
    top_n = top_n or 10
    ascending = ascending if ascending is not None else False

    if operation not in ("filter", "landscape", "stats") and not column:
        return "Missing 'column' parameter."
    if column and column not in ALLOWED_COLS:
        return f"Invalid column: {column}. Allowed: {ALLOWED_COLS}"

    result_df = df.copy()
    result_df[AMT] = pd.to_numeric(result_df[AMT], errors="coerce")

    result_df = _apply_filters(result_df, filters, ALLOWED_COLS)
    if isinstance(result_df, str):
        return result_df

    if operation in ("top", "bottom", "sum", "landscape", "trend", "group", "stats"):
        if operation in ("landscape", "stats"):
            result_df = result_df.dropna(subset=[AMT])
        elif column and pd.api.types.is_numeric_dtype(result_df.get(column, pd.Series())):
            result_df = result_df.dropna(subset=[column])
        if result_df.empty:
            return "No valid numeric data after filtering."

    if result_df.empty:
        return "No data matches the filters."

    # top
    if operation == "top":
        if not pd.api.types.is_numeric_dtype(result_df[column]):
            return f"'{column}' is not numeric. Use 'count' instead."
        out = result_df.nlargest(top_n, column)[DISPLAY_COLS]
        return out.to_string(index=False)

    # bottom
    elif operation == "bottom":
        if not pd.api.types.is_numeric_dtype(result_df[column]):
            return f"'{column}' is not numeric."
        out = result_df.nsmallest(top_n, column)[DISPLAY_COLS]
        return out.to_string(index=False)

    # group
    elif operation == "group":
        if not group_by or group_by not in ALLOWED_COLS:
            return f"Invalid group_by: {group_by}. Allowed: {ALLOWED_COLS}"

        col = AMT
        actual_group = "startup_id" if group_by == "Startup Name" else group_by

        out = (
            result_df.groupby(actual_group)[col]
            .agg(total="sum", count="count", average="mean")
        )
        sort_col = sort_by if sort_by in ("total", "count", "average") else "total"
        out = out.sort_values(sort_col, ascending=ascending).head(top_n)

        if actual_group == "startup_id":
            id_to_name = result_df.drop_duplicates("startup_id").set_index("startup_id")["Startup Name"]
            out.index = out.index.map(id_to_name)

        out["total"] = out["total"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
        out["average"] = out["average"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
        return out.to_string()

    # filter
    elif operation == "filter":
        if len(result_df) <= top_n:
            out = result_df[DISPLAY_COLS]
        else:
            out = result_df[DISPLAY_COLS].sample(top_n)
        total = len(result_df)
        return f"Showing {len(out)} of {total} matching rows\n\n" + out.to_string(index=False)

    # count
    elif operation == "count":
        out = result_df[column].value_counts()
        if ascending:
            out = out.sort_values(ascending=True)
        out = out.head(top_n)
        return out.to_string()

    # sum
    elif operation == "sum":
        if not pd.api.types.is_numeric_dtype(result_df[column]):
            return f"'{column}' is not numeric."
        total = result_df[column].sum()
        count = result_df[column].notna().sum()
        avg = result_df[column].mean()
        return f"Total: ${total:,.0f}\nDeals: {count}\nAverage: ${avg:,.0f}"

    # unique
    elif operation == "unique":
        if column in ("Investors", "Founders"):
            all_vals = result_df[column].dropna().str.split(",").explode().str.strip()
            all_vals = all_vals[(all_vals != "") & (all_vals != "nan")]
            unique = all_vals.nunique()
            top_vals = all_vals.value_counts().head(top_n)
            lines = [f"Unique {column}: {unique}\n"]
            for name, count in top_vals.items():
                lines.append(f"{name:30s} | {count} deal(s)")
            return "\n".join(lines)
        else:
            unique = result_df[column].nunique()
            top_vals = result_df[column].value_counts().head(top_n)
            lines = [f"Unique {column}: {unique}\n"]
            for name, count in top_vals.items():
                lines.append(f"{name:30s} | {count}")
            return "\n".join(lines)

    # trend
    elif operation == "trend":
        if group_by not in ("_source_year", "_source_month"):
            return "trend only works with group_by='_source_year' or '_source_month'"
        if group_by == "_source_month":
            month_order = list(range(1, 13))
            out = (
                result_df.groupby("_source_month")[column]
                .agg(total="sum", deals="count")
            )
            out = out.reindex(month_order).dropna(how="all").fillna(0)
        else:
            out = (
                result_df.groupby("_source_year")[column]
                .agg(total="sum", deals="count")
                .sort_index()
            )
        out["total"] = out["total"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "$0")
        return out.to_string()

    # investor_network
    elif operation == "investor_network":
        if column != "Investors":
            return "investor_network only works with column='Investors'"
        co_investors = {}
        for _, row in result_df.iterrows():
            investors = str(row.get("Investors", ""))
            if not investors or investors == "nan":
                continue
            names = [inv.strip() for inv in investors.split(",") if inv.strip()]
            for name in names:
                if name not in co_investors:
                    co_investors[name] = {"deals": 0, "startup_ids": set(), "total": 0}
                co_investors[name]["deals"] += 1
                sid = str(row.get("startup_id", ""))
                co_investors[name]["startup_ids"].add(sid)
                amt = row.get(AMT, 0)
                if pd.notna(amt):
                    co_investors[name]["total"] += amt

        sort_key = sort_by if sort_by in ("deals", "total") else "deals"
        sorted_inv = sorted(co_investors.items(), key=lambda x: x[1][sort_key], reverse=not ascending)
        id_to_name = result_df.drop_duplicates("startup_id").set_index("startup_id")["Startup Name"]
        lines = []
        for inv, data in sorted_inv[:top_n]:
            names = [id_to_name.get(s, s) for s in list(data["startup_ids"])[:5]]
            lines.append(f"{inv:30s} | {data['deals']} deals | ${data['total']:,.0f} | {', '.join(names)}")
        return "\n".join(lines) if lines else "No investor data found."

    # landscape
    elif operation == "landscape":
        lines = []

        total = result_df[AMT].sum()
        deals = result_df[AMT].notna().sum()
        avg = result_df[AMT].mean()
        median = result_df[AMT].median()
        lines.append("OVERVIEW")
        lines.append(f"Total Funding: ${total:,.0f} | Deals: {deals} | Avg Deal: ${avg:,.0f} | Median: ${median:,.0f}")
        lines.append("")

        lines.append("TOP DEALS")
        top = result_df.nlargest(top_n, AMT)[["startup_id", "Startup Name", AMT, "Investment Stage", "City"]]
        top[AMT] = top[AMT].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
        lines.append(top.to_string(index=False))
        lines.append("")

        lines.append("BY CITY")
        cities = (
            result_df.groupby("City")[AMT]
            .agg(total="sum", deals="count")
            .sort_values("total", ascending=False)
            .head(5)
        )
        cities["total"] = cities["total"].apply(lambda x: f"${x:,.0f}")
        lines.append(cities.to_string())
        lines.append("")

        lines.append("BY STAGE")
        stages = result_df["Investment Stage"].value_counts().head(5)
        lines.append(stages.to_string())
        lines.append("")

        lines.append("YEARLY TREND")
        trend = (
            result_df.groupby("_source_year")[AMT]
            .agg(total="sum", deals="count")
            .sort_index()
        )
        trend["total"] = trend["total"].apply(lambda x: f"${x:,.0f}")
        lines.append(trend.to_string())
        lines.append("")

        lines.append("TOP INVESTORS")
        inv_counts = {}
        for _, row in result_df.iterrows():
            investors = str(row.get("Investors", ""))
            if not investors or investors == "nan":
                continue
            for inv in investors.split(","):
                inv = inv.strip()
                if inv:
                    inv_counts[inv] = inv_counts.get(inv, 0) + 1
        top_inv = sorted(inv_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        for inv, count in top_inv:
            lines.append(f"  {inv:30s} | {count} deals")

        return "\n".join(lines)

    # crosstab
    elif operation == "crosstab":
        if not group_by or group_by not in ALLOWED_COLS:
            return f"Invalid group_by: {group_by}. Allowed: {ALLOWED_COLS}"

        actual_col = "startup_id" if column == "Startup Name" else column
        actual_group = "startup_id" if group_by == "Startup Name" else group_by

        work = result_df.copy()

        if actual_col in ("Investors", "Founders"):
            work[actual_col] = work[actual_col].str.split(",")
            work = work.explode(actual_col)
            work[actual_col] = work[actual_col].str.strip()
            work = work[(work[actual_col].ne("")) & (work[actual_col].ne("nan"))]

        if actual_group in ("Investors", "Founders"):
            work[actual_group] = work[actual_group].str.split(",")
            work = work.explode(actual_group)
            work[actual_group] = work[actual_group].str.strip()
            work = work[(work[actual_group].ne("")) & (work[actual_group].ne("nan"))]

        if pd.api.types.is_numeric_dtype(result_df.get(column, pd.Series())):
            cross = (
                work.groupby(actual_group)[column]
                .agg(total="sum", deals="count")
                .reset_index()
                .sort_values("total", ascending=ascending)
            )
            cross["total"] = cross["total"].apply(lambda x: f"${x:,.0f}")
            return cross.head(top_n).to_string(index=False)
        else:
            cross = (
                work.groupby([actual_group, actual_col])
                .size()
                .reset_index(name="deals")
                .sort_values("deals", ascending=ascending)
            )
            top_per_group = (
                cross.sort_values("deals", ascending=ascending)
                .groupby(actual_group)
                .head(3)
                .sort_values([actual_group, "deals"], ascending=[True, ascending])
            )
            return top_per_group.head(top_n * 3).to_string(index=False)

    # stats
    elif operation == "stats":
        col = AMT
        total = result_df[col].sum()
        count = result_df[col].notna().sum()
        avg = result_df[col].mean()
        median = result_df[col].median()
        minimum = result_df[col].min()
        maximum = result_df[col].max()

        full_total = df[AMT].sum()
        pct = (total / full_total * 100) if full_total > 0 else 0

        multi_round = result_df.groupby("startup_id").size()
        multi = multi_round[multi_round > 1]

        lines = [
            f"Total: ${total:,.0f}",
            f"Deals: {count}",
            f"Average: ${avg:,.0f}",
            f"Median: ${median:,.0f}",
            f"Min: ${minimum:,.0f}",
            f"Max: ${maximum:,.0f}",
            f"Share of dataset: {pct:.1f}%",
            f"",
            f"Startups with multiple rounds: {len(multi)}",
        ]
        if not multi.empty:
            id_to_name = result_df.drop_duplicates("startup_id").set_index("startup_id")["Startup Name"]
            for sid, rounds in multi.sort_values(ascending=False).head(top_n).items():
                name = id_to_name.get(sid, sid)
                lines.append(f"  {name:30s} | {rounds} rounds")

        return "\n".join(lines)

    else:
        return f"Unknown operation: {operation}. Use: top, bottom, group, filter, count, sum, unique, trend, investor_network, landscape, crosstab, stats"


@tool(args_schema=AnalyzeArgs)
def analyze_dataset(operation, column=None, group_by=None,
                    filters=None, top_n=10, sort_by=None, ascending=False) -> str:
    """Analyze startup funding data. Safe structured queries — no code execution.
    Args:
        Operations:
            - top: Largest N by column → "Top 5 funded deals"
            - bottom: Smallest N by column → "Least funded deals"
            - group: Aggregate by category → "Funding by industry" / "Compare 2020 vs 2021"
            - filter: Find matching rows → "FinTech startups in Bangalore"
            - count: Value frequency → "Deals per investment stage"
            - sum: Total across dataset → "Total capital deployed"
            - unique: Distinct values → "How many unique investors?"
            - trend: Time-based trend → "Monthly funding trend"
            - investor_network: Investor activity → "Most active investors"
            - landscape: Full market map (no column needed) → "Map the FinTech landscape"
            - crosstab: Cross-column analysis → "Which cities dominate which sectors?"
            - stats: Advanced statistics (no column needed) → "What % of deals are FinTech?"
        column: Column to analyze (not required for filter/landscape/stats)
        group_by: Column to group by (for group, trend, crosstab)
        Filters: list of dicts. One dict=AND, multiple dicts=OR.
            contains:      {"City": "Bangalore"}
            exact:         {"City": {"exact": "New Delhi"}}
            negation:      {"City": {"not": "Mumbai"}}
            negation list: {"City": {"not": ["Mumbai", "Delhi"]}}
            OR list:       {"City": ["Bangalore", "Mumbai"]}
            numeric exact: {"_source_year": 2021}
            numeric range: {"Amount(in USD)": {"min": 1000000, "max": 50000000}}
            range ops:     {"Amount(in USD)": {"gt": 10000000, "lt": 50000000}}
            item count:    {"Founders": {"min_items": 3}}
            string length: {"Startup Name": {"min_len": 4}}
            starts with:   {"Startup Name": {"starts_with": "A"}}
            ends with:     {"Industry/Vertical": {"ends_with": "Tech"}}  
        top_n: Number of results (default 10)
        sort_by: Sort by 'total', 'count', or 'average'
        ascending: Sort ascending (True) or descending (False)
    """
    result = analyze_dataset_function(operation, column, group_by, filters, ascending, sort_by, top_n)
    return _truncate(result)


_ddg = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Search the web for info not in the dataset.
    Use after lookup_entity returns NOT FOUND,, or for:
    - Recent news, valuations, funding rounds
    - Competitive landscape
    - Any entity not in the dataset

    Args:
        query: Search query (e.g. "Razorpay latest funding round valuation")
    """
    try:
        return _ddg.run(query)[:4000]
    except Exception as e:
        return f"Web search error: {e}"


tools = [lookup_entity, analyze_dataset, web_search]


# -------------------------------------------------------------------
# agent setup
# -------------------------------------------------------------------

_industries = sorted(df["Industry/Vertical"].dropna().unique().tolist())[:5]
_cities = sorted(df["City"].dropna().unique().tolist())[:5]
_stages = sorted([s for s in df["Investment Stage"].dropna().unique().tolist() if s and s != "nan"])[:5]
_years = sorted(df["_source_year"].dropna().unique().tolist())

SYSTEM_PROMPT = f"""You are a Financial Intelligence Research Agent for Private Equity analysts.

## Dataset
{len(df)} startup funding records across {len(_years)} years ({_years}).
Each row = one deal/funding round. A startup can have multiple deals.

CSV schema (columns):
- Startup Name: name of the startup/company (may be shared by different companies)
- Founders: founder names (comma-separated)
- Investors: investor names (comma-separated)
- Amount(in USD): funding amount in USD (may be missing)
- Investment Stage: e.g., Seed, Series A, Series B, etc.
- Founding Date: founding date (may be missing)
- City: startup headquarters city
- Industry/Vertical: broad sector (e.g., FinTech, EdTech, E-commerce)
- Sub-Vertical: specific segment within the vertical
- _source_year: year the record belongs to (e.g., 2020)
- _source_month: month as integer (1-12, e.g., 2 for February)
- _country: country of the startup (e.g., India, USA, UK) -- auto-mapped from City
- _continent: continent (Asia, Europe, North America, Oceania)

Examples -- Industries: {_industries} | Cities: {_cities} | Stages: {_stages}

""" + r"""## Tool Usage

1. **lookup_entity** -- call FIRST when user mentions a specific name
   - FOUND -> answer ONLY from dataset, tag [Source: Dataset]. Do NOT call web_search.
   - NOT FOUND -> call web_search, tag [Source: External]

2. **analyze_dataset** -- for analytics across rows. ONE call per question.
   12 operations: top, bottom, group, filter, count, sum, unique, trend, investor_network, landscape, crosstab, stats

   Examples:
   - "Top 10 funded deals" -> operation="top", column="Amount(in USD)"
   - "Least funded deals" -> operation="bottom", column="Amount(in USD)"
   - "Funding by industry" -> operation="group", group_by="Industry/Vertical"
   - "Average funding per startup" -> operation="group", group_by="Startup Name", sort_by="average"
   - "Industry with lowest avg funding" -> operation="group", group_by="Industry/Vertical", sort_by="average", ascending=True, top_n=1
   - "Compare 2020 vs 2021" -> operation="group", group_by="_source_year"
   - "Compare Europe vs Asia" -> operation="group", group_by="_continent"
   - "Funding by country" -> operation="group", group_by="_country"
   - "FinTech startups in Bangalore" -> operation="unique", column="Startup Name", filters=[{"Industry/Vertical": "FinTech", "City": "Bangalore"}]
   - "Startups NOT in New York" -> operation="unique", column="Startup Name", filters=[{"City": {"not": "New York"}}]
   - "Non-FinTech startups" -> operation="unique", column="Startup Name", filters=[{"Industry/Vertical": {"not": "FinTech"}}]
   - "Deals outside 2020" -> operation="filter", filters=[{"_source_year": {"not": 2020}}]
   - "Startups in Bangalore OR Mumbai" -> operation="unique", column="Startup Name", filters=[{"City": "Bangalore"}, {"City": "Mumbai"}]
   - "Deals above $50M" -> operation="filter", filters=[{"Amount(in USD)": {"min": 50000000}}]
   - "Deals between $10M and $50M" -> operation="filter", filters=[{"Amount(in USD)": {"gt": 10000000, "lt": 50000000}}]
   - "Deals per stage" -> operation="count", column="Investment Stage"
   - "Total funding" -> operation="sum", column="Amount(in USD)"
   - "Most active investors" -> operation="investor_network", column="Investors"
   - "How many unique investors?" -> operation="unique", column="Investors"
   - "Monthly funding trend" -> operation="trend", column="Amount(in USD)", group_by="_source_month"
   - "Who co-invests with Sequoia?" -> operation="investor_network", column="Investors", filters=[{"Investors": "Sequoia"}]
   - "Map FinTech landscape" -> operation="landscape", filters=[{"Industry/Vertical": "FinTech"}]
   - "Which cities dominate which sectors?" -> operation="crosstab", column="Industry/Vertical", group_by="City"
   - "What % of deals are FinTech?" -> operation="stats", filters=[{"Industry/Vertical": "FinTech"}]
   - "Startups in Europe" -> operation="filter", filters=[{"_continent": "Europe"}]
   - "Non-Indian startups" -> operation="filter", filters=[{"_country": {"not": "India"}}]
   - "Total raised by CRED" → lookup_entity("CRED") → FOUND → analyze_dataset(operation="sum", column="Amount(in USD)", filters=[{"Startup Name": "CRED"}])
   - Tag [Source: Dataset]

3. **web_search** -- for info NOT in dataset
   - Only after lookup_entity confirms NOT FOUND
   - Tag [Source: External]

## Boundaries
- You are ONLY a Financial Intelligence Research Agent for Private Equity analysts.
- Dataset covers startup funding records from 2020-2021.
- ONLY answer questions related to startups, funding, investors, industries, market mapping, and due diligence.
- If user asks about a startup ,founder, investor, or sub-vertical NOT in the dataset:
  - Say "X is not in our dataset (2020-2021)"
  - Use web_search for brief context if relevant to PE landscape
- If the user asks about anything unrelated (sports, celebrities, general knowledge, etc.), politely decline and redirect.

## Rules
- ALWAYS tag: [Source: Dataset] or [Source: External]
- If entity NOT FOUND, say "X is not in our dataset" before giving external info
- Format amounts: $15M not 15000000
- NEVER use HTML tags. Use plain text only.
- Do not invent facts. If you cannot verify, say so.
- OUTPUT RULE: Always include actual values from tools. Never respond with generic filler.
- Use ONE tool call per question when possible.
- Be concise, analytical -- write like a PE research analyst
"""


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

agent = create_agent(llm, tools=tools, system_prompt=SYSTEM_PROMPT)


# -------------------------------------------------------------------
# chat
# -------------------------------------------------------------------

messages = []
MAX_HISTORY = 6


def chat(user_input):
    """Send a message and get a response."""
    global messages
    if len(messages) > MAX_HISTORY:
        messages = messages[-MAX_HISTORY:]
    messages.append({"role": "user", "content": user_input})

    try:
        result = agent.invoke({"messages": messages})
    except Exception as e:
        return f"Agent error: {e}"

    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "type") and msg.type == "ai" and isinstance(msg.content, str) and msg.content.strip():
            clean = msg.content.replace("<br>", "\n")
            clean = re.sub(r"<[^>]+>", "", clean)
            messages.append({"role": "assistant", "content": clean})
            return clean

    return "Could not generate a response."


def reset():
    """Clear conversation history."""
    global messages
    messages = []


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if "--interactive" in sys.argv or "-i" in sys.argv:
        print("=" * 50)
        print("  PE Intelligence Agent")
        print("  Type 'quit' to exit, 'reset' to clear history")
        print("=" * 50)
        while True:
            try:
                q = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() in ("quit", "exit", "q"):
                break
            if q.lower() == "reset":
                reset()
                continue
            print(f"\nAgent: {chat(q)}")
    else:
        print("PE Intelligence Agent -- Demo\n")

        print("Q: Top 5 most funded deals?")
        print(f"A: {chat('What are the top 5 most funded deals? Show industry, investors, city.')}\n")

        print("Q: Tell me about Razorpay")
        print(f"A: {chat('Tell me about Razorpay -- what stage are they at?')}")
