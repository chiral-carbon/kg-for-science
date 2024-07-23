import gradio as gr
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from create_db import ArxivDatabase
from config import DEFAULT_TABLES_DIR
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


def query_db(question, is_sql):
    global db
    if db is None:
        return "Please load a database first."
    return db.query_db(question, is_sql)


with gr.Blocks() as demo:
    gr.Markdown("# ArXiv Database Query Interface")

    with gr.Row():
        db_dropdown = gr.Dropdown(
            choices=get_available_databases(), label="Select Database"
        )
        load_db_btn = gr.Button("Load Database", size="sm")
        model_id_input = gr.Textbox(
            label="Model ID from Huggingface (optional)",
            placeholder="Leave blank for default (meta-llama/Meta-Llama-3-8B-Instruct)",
        )
        load_model_btn = gr.Button("Load Model (ETA: 1 min)", size="sm")

    with gr.Row():
        status = gr.Textbox(label="Status")

    with gr.Row():
        nl_input = gr.Textbox(label="Natural Language Query")
        sql_input = gr.Textbox(label="SQL Query")

    with gr.Row():
        nl_submit = gr.Button("Submit NL Query", size="sm")
        sql_submit = gr.Button("Submit SQL Query", size="sm")

    output = gr.Textbox(label="Results")

    load_db_btn.click(
        load_database, inputs=[db_dropdown, model_id_input], outputs=status
    )
    load_model_btn.click(load_model, outputs=status)
    nl_submit.click(
        query_db, inputs=[nl_input, gr.Checkbox(value=False)], outputs=output
    )
    sql_submit.click(
        query_db, inputs=[sql_input, gr.Checkbox(value=True)], outputs=output
    )

if __name__ == "__main__":
    set_env_vars()
    demo.launch(share=True)
