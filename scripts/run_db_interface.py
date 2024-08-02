import gradio as gr
import os
import sys
import sqlite3
import json
import re
import pandas as pd
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from create_db import ArxivDatabase
from config import DEFAULT_TABLES_DIR, DEFAULT_INTERFACE_MODEL_ID, canned_queries
from src.utils.utils import set_env_vars

db = None


def truncate_or_wrap_text(text, max_length=50, wrap=False):
    """Truncate text to a maximum length, adding ellipsis if truncated, or wrap if specified."""
    if wrap:
        return "\n".join(
            text[i : i + max_length] for i in range(0, len(text), max_length)
        )
    return text[:max_length] + "..." if len(text) > max_length else text


def format_url(url):
    """Format URL to be more compact in the table."""
    return url.split("/")[-1] if url.startswith("http") else url


def get_available_databases():
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tables_dir = os.path.join(ROOT, DEFAULT_TABLES_DIR)
    return [f for f in os.listdir(tables_dir) if f.endswith(".db")]


def load_database(db_name, model_id=None):
    global db
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    db_path = os.path.join(ROOT, DEFAULT_TABLES_DIR, db_name)
    if not os.path.exists(db_path):
        return f"Database {db_name} does not exist."
    db = ArxivDatabase(db_path, model_id)
    db.init_db()  # Ensure tables are created
    if db.is_db_empty:
        return f"Database loaded from {db_path}, but it is empty. Please populate it with data."
    return f"Database loaded from {db_path}"


def query_db(query, is_sql, limit=None, wrap=False):
    global db
    if db is None:
        return pd.DataFrame({"Error": ["Please load a database first."]})

    try:
        cursor = db.conn.cursor()

        # Remove any trailing semicolons and combine all lines
        query = " ".join(query.strip().split("\n")).rstrip(";")

        # Apply limit if provided
        if limit is not None:
            if "LIMIT" in query.upper():
                # Replace existing LIMIT clause
                query = re.sub(
                    r"LIMIT\s+\d+", f"LIMIT {limit}", query, flags=re.IGNORECASE
                )
            else:
                # Add LIMIT clause
                query += f" LIMIT {limit}"

        cursor.execute(query)

        # Fetch column names
        column_names = [description[0] for description in cursor.description]

        # Fetch all results
        results = cursor.fetchall()

        # Create a pandas DataFrame
        df = pd.DataFrame(results, columns=column_names)

        # Process results to handle JSON fields and format output
        for column in df.columns:
            if df[column].dtype == "object":
                df[column] = df[column].apply(
                    lambda x: (
                        format_url(x)
                        if column == "url"
                        else truncate_or_wrap_text(x, wrap=wrap)
                    )
                )

        return df

    except sqlite3.Error as e:
        return pd.DataFrame({"Error": [str(e)]})


css = """
#selected-query {
    max-height: 100px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# ArXiv Database Query Interface")

    with gr.Row():
        db_dropdown = gr.Dropdown(
            choices=get_available_databases(), label="Select Database"
        )
        load_db_btn = gr.Button("Load Database", size="sm")

    with gr.Row():
        status = gr.Textbox(label="Status")

    with gr.Row():
        wrap_checkbox = gr.Checkbox(label="Wrap long text", value=False)
        canned_query_dropdown = gr.Dropdown(
            choices=[q[0] for q in canned_queries], label="Select Query", scale=3
        )
        limit_input = gr.Number(label="Limit", value=10000, step=1, minimum=1, scale=1)
        selected_query = gr.Textbox(
            label="Selected Query",
            interactive=False,
            scale=2,
            show_label=True,
            show_copy_button=True,
            elem_id="selected-query",
        )
        canned_query_submit = gr.Button("Submit Query", size="sm", scale=1)

    with gr.Row():
        sql_input = gr.Textbox(label="Custom SQL Query", lines=3, scale=4)
        sql_submit = gr.Button("Submit Custom SQL", size="sm", scale=1)

    output = gr.DataFrame(label="Results", wrap=True)

    def update_selected_query(query_description):
        for desc, sql in canned_queries:
            if desc == query_description:
                return sql
        return ""

    def submit_canned_query(query_description, limit, wrap):
        for desc, sql in canned_queries:
            if desc == query_description:
                return query_db(sql, True, limit, wrap)
        return pd.DataFrame({"Error": ["Selected query not found."]})

    load_db_btn.click(load_database, inputs=[db_dropdown], outputs=status)
    canned_query_dropdown.change(
        update_selected_query, inputs=[canned_query_dropdown], outputs=[selected_query]
    )
    canned_query_submit.click(
        submit_canned_query,
        inputs=[canned_query_dropdown, limit_input, wrap_checkbox],
        outputs=output,
    )
    sql_submit.click(
        query_db,
        inputs=[sql_input, gr.Checkbox(value=True), limit_input, wrap_checkbox],
        outputs=output,
    )

if __name__ == "__main__":
    set_env_vars()
    demo.launch(share=True)

# Add this line at the very end of your script, outside of any function or class
demo.launch()
