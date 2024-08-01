import gradio as gr
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from create_db import ArxivDatabase
from config import DEFAULT_TABLES_DIR, DEFAULT_INTERFACE_MODEL_ID
from src.utils.utils import set_env_vars

db = None


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


def load_model():
    global db
    if db is None:
        return "Please load a database first."
    return db.load_model()


def query_db(query, is_sql):
    global db
    if db is None:
        return "Please load a database first."
    return db.query_db(query, is_sql)


# Canned queries
canned_queries = [
    ("List all papers", "SELECT * FROM papers LIMIT 10"),
    (
        "Count papers by category",
        "SELECT primary_category, COUNT(*) FROM papers GROUP BY primary_category",
    ),
    (
        "Top 10 authors with most papers",
        "SELECT authors, COUNT(*) as paper_count FROM papers GROUP BY authors ORDER BY paper_count DESC LIMIT 10",
    ),
    (
        "Papers with 'quantum' in abstract",
        "SELECT paper_id, abstract FROM papers WHERE abstract LIKE '%quantum%' LIMIT 5",
    ),
    (
        "Most common concepts",
        "SELECT concept, COUNT(*) as concept_count FROM predictions GROUP BY concept ORDER BY concept_count DESC LIMIT 10",
    ),
]

with gr.Blocks() as demo:
    gr.Markdown("# ArXiv Database Query Interface")

    with gr.Row():
        db_dropdown = gr.Dropdown(
            choices=get_available_databases(), label="Select Database"
        )
        load_db_btn = gr.Button("Load Database", size="sm")
        model_id_input = gr.Textbox(
            label="Model ID from Huggingface (optional)",
            placeholder=f"Leave blank for default ({DEFAULT_INTERFACE_MODEL_ID})",
        )
        load_model_btn = gr.Button("Load Model (ETA: 1 min)", size="sm")

    with gr.Row():
        status = gr.Textbox(label="Status")

    with gr.Row():
        canned_query_dropdown = gr.Dropdown(
            choices=[q[0] for q in canned_queries], label="Select Canned Query"
        )
        canned_query_submit = gr.Button("Submit Canned Query", size="sm")

    with gr.Row():
        sql_input = gr.Textbox(label="Custom SQL Query")
        sql_submit = gr.Button("Submit Custom SQL Query", size="sm")

    # Commented out NL query section
    with gr.Row():
        nl_input = gr.Textbox(label="Natural Language Query")
        nl_submit = gr.Button("Submit NL Query", size="sm")

    output = gr.Textbox(label="Results")

    def submit_canned_query(query_description):
        for desc, sql in canned_queries:
            if desc == query_description:
                return query_db(sql, True)
        return "Selected query not found."

    load_db_btn.click(
        load_database, inputs=[db_dropdown, model_id_input], outputs=status
    )
    load_model_btn.click(load_model, outputs=status)
    canned_query_submit.click(
        submit_canned_query, inputs=[canned_query_dropdown], outputs=output
    )
    sql_submit.click(
        query_db, inputs=[sql_input, gr.Checkbox(value=True)], outputs=output
    )
    nl_submit.click(
        query_db, inputs=[nl_input, gr.Checkbox(value=False)], outputs=output
    )

if __name__ == "__main__":
    set_env_vars()
    demo.queue()
    demo.launch(share=True)
