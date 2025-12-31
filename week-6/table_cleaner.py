import re
import json
from html import unescape

#copy pasted from gpt to make table data more readable

def _strip_tags(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</(p|div|tr|li|h1|h2|h3|h4|h5|h6)>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = unescape(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()

def _parse_html_table_to_rows(table_html: str):
    # grab rows
    rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", table_html, flags=re.I | re.S)
    if not rows:
        # some tables use <th> without wrapping in <tr>
        # fallback: treat any <th> blocks as one row
        th_blocks = re.findall(r"<th\b[^>]*>(.*?)</th>", table_html, flags=re.I | re.S)
        rows = th_blocks

    parsed = []
    for r in rows:
        # cells can be <td> or <th>
        cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", r, flags=re.I | re.S)
        cells = [_strip_tags(c) for c in cells]
        # drop empty rows
        if any(c.strip() for c in cells):
            parsed.append(cells)

    # header rows sometimes appear as consecutive <th> outside <tr>
    # also, your sample has <th><td ...></td><td>Quarter Ended...</td></th>
    if not parsed:
        cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", table_html, flags=re.I | re.S)
        cells = [_strip_tags(c) for c in cells]
        if any(c.strip() for c in cells):
            parsed.append(cells)

    # normalize row lengths
    max_len = max((len(r) for r in parsed), default=0)
    parsed = [r + [""] * (max_len - len(r)) for r in parsed]
    return parsed

def extract_tables_compact(html: str):
    tables = re.findall(r"<table\b[^>]*>.*?</table>", html, flags=re.I | re.S)
    out = []
    for t in tables:
        rows = _parse_html_table_to_rows(t)
        if rows:
            out.append(rows)
    return out

def rows_to_tsv(rows):
    # compact + readable for LLMs, usually fewer tokens than markdown tables
    def clean_cell(c: str) -> str:
        c = c.replace("\n", " ").strip()
        c = re.sub(r"\s+", " ", c)
        return c
    lines = ["\t".join(clean_cell(c) for c in row).rstrip() for row in rows]
    return "\n".join(lines).strip()

def clean_table_record(record: dict):
    html = record.get("html", "")
    tables = extract_tables_compact(html)

    cleaned_tables = []
    for i, rows in enumerate(tables, 1):
        cleaned_tables.append({
            "table_index": i,
            "tsv": rows_to_tsv(rows),
            "n_rows": len(rows),
            "n_cols": len(rows[0]) if rows else 0,
        })

    # keep only the tables (drop headings/footnotes/etc to save tokens)
    return {
        "table_id": record.get("table_id"),
        "title": record.get("title"),
        "tables": cleaned_tables,
    }

# ---- example on your input ----
record = {
    "table_id": 1,
    "title": "Q1 2024 Financial Highlights (unaudited)",
    "html": """<h1>Q1 2024 Financial Highlights (unaudited)</h1>..."""  # put your html here
}

cleaned = clean_table_record(record)
print(json.dumps(cleaned, ensure_ascii=False, indent=2))

# if you want: for LLM context, you can join just the TSVs:
# context = "\n\n---\n\n".join(t["tsv"] for t in cleaned["tables"])
