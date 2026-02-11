# 📊 PE Intelligence Agent

Conversational AI for Private Equity analysts. Built with **LangChain `create_agent`** + **Groq**.

## How It Works
```
User asks "Tell me about CRED"
  → lookup_entity("CRED")            # Step 1: check dataset
  → ✅ FOUND (FinTech, Mumbai)        # exists in our data
  → Response [Source: Dataset]        # answer from dataset only

User asks "Tell me about Stripe"
  → lookup_entity("Stripe")          # Step 1: check dataset
  → ❌ NOT FOUND                      # not in our data
  → web_search("Stripe funding")     # Step 2: search web
  → Response [Source: External]       # clearly labeled

User asks "Top 10 funded deals?"
  → analyze_dataset(operation="top", column="Amount(in USD)")
  → Response [Source: Dataset]

User asks "Startups with 3+ founders?"
  → analyze_dataset(operation="filter", filters=[{"Founders": {"min_items": 3}}])
  → Response [Source: Dataset]
```

## Architecture
```
agent.py
├── load_data()           → loads CSVs, cleans, deduplicates into DataFrame
├── detect_entity()       → existence check across 4 columns (1 row, minimal output)
├── _apply_filters()      → 10 filter types (contains, exact, range, negation, etc.)
├── analyze_dataset_function() → 12 structured operations (no exec/eval)
├── @tool lookup_entity   → wraps detect_entity for the LLM
├── @tool analyze_dataset → wraps analyze_dataset_function for the LLM
├── @tool web_search      → DuckDuckGo fallback
├── create_agent(llm, tools, system_prompt)
└── chat() / reset()
```

## Setup
```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_api_key_here
```

## Run
```bash
# CLI demo
python agent.py

# Interactive chat
python agent.py --interactive

# Web UI
streamlit run app.py
```

## Files
```
pe-intelligence-agent/
├── agent.py            # Data loading, tools, agent, chat
├── app.py              # Streamlit UI (imports from agent.py)
├── data.csv          # Pre-processed dataset (2,119 rows)
├── requirements.txt
├── .env                # API key (not committed)
├── .gitignore
├── README.md
└── data/
    ├── 2020/
    │   ├── Jan_2020.csv
    │   ├── Feb_2020.csv
    │   └── ...
    └── 2021/
        ├── Jan_2021.csv
        └── ...
```

## Dataset

2,119 startup funding records (2020–2021) across 1,659 unique startups. Columns: startup_id,Startup Name, Founding Date, City, Industry/Vertical, Sub-Vertical, Founders, Investors, Amount(in USD), Investment Stage, _country, _continent,_source_month,_source_year.

## Tools

| Tool | Purpose | When |
|---|---|---|
| `lookup_entity` | Check if entity exists in dataset | FIRST, for any named entity |
| `analyze_dataset` | 12 structured operations (top, group, filter, trend, etc.) | Analytics across rows |
| `web_search` | DuckDuckGo search | Only after lookup_entity returns NOT FOUND |

## Filter System

| Type | Example |
|---|---|
| Contains | `{"City": "Bangalore"}` |
| Exact | `{"City": {"exact": "New Delhi"}}` |
| Negation | `{"City": {"not": "Mumbai"}}` |
| OR list | `{"City": ["Bangalore", "Mumbai"]}` |
| Numeric range | `{"Amount(in USD)": {"min": 1000000, "max": 50000000}}` |
| Item count | `{"Founders": {"min_items": 3}}` |
| String length | `{"Startup Name": {"min_len": 4}}` |
| Starts/ends with | `{"Startup Name": {"starts_with": "A"}}` |

## Operations

| Operation | Example Query |
|---|---|
| `top` / `bottom` | "Top 10 funded deals" |
| `group` | "Funding by industry" / "Compare 2020 vs 2021" |
| `filter` | "FinTech startups in Bangalore" |
| `count` | "Deals per investment stage" |
| `sum` | "Total capital deployed" |
| `unique` | "How many unique investors?" |
| `trend` | "Monthly funding trend" |
| `investor_network` | "Most active investors" |
| `landscape` | "Map the FinTech landscape" |
| `crosstab` | "Which cities dominate which sectors?" |
| `stats` | "What % of deals are FinTech?" |
