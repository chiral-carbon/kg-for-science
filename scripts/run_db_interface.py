import gradio as gr
import os
import json
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import re
import sys
import sqlite3

from plotly.subplots import make_subplots
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


# def load_database(db_name, model_id=None):
#     global db
#     ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#     db_path = os.path.join(ROOT, DEFAULT_TABLES_DIR, db_name)
#     if not os.path.exists(db_path):
#         return f"Database {db_name} does not exist."
#     db = ArxivDatabase(db_path, model_id)
#     db.init_db()  # Ensure tables are created
#     if db.is_db_empty:
#         return f"Database loaded from {db_path}, but it is empty. Please populate it with data."
#     return f"Database loaded from {db_path}"


def query_db(query, is_sql, limit=None, wrap=False):
    global db
    if db is None:
        return pd.DataFrame({"Error": ["Please load a database first."]})

    try:
        cursor = db.conn.cursor()

        query = " ".join(query.strip().split("\n")).rstrip(";")

        if limit is not None:
            if "LIMIT" in query.upper():
                # Replace existing LIMIT clause
                query = re.sub(
                    r"LIMIT\s+\d+", f"LIMIT {limit}", query, flags=re.IGNORECASE
                )
            else:
                query += f" LIMIT {limit}"

        cursor.execute(query)

        column_names = [description[0] for description in cursor.description]

        results = cursor.fetchall()

        df = pd.DataFrame(results, columns=column_names)

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


def generate_concept_cooccurrence_graph(db_path):
    conn = sqlite3.connect(db_path)
    query = """
    WITH concept_pairs AS (
        SELECT p1.concept AS concept1, p2.concept AS concept2, p1.paper_id
        FROM predictions p1
        JOIN predictions p2 ON p1.paper_id = p2.paper_id AND p1.concept < p2.concept
        WHERE p1.tag_type = 'object' AND p2.tag_type = 'object'
    )
    SELECT concept1, concept2, COUNT(DISTINCT paper_id) AS co_occurrences
    FROM concept_pairs
    GROUP BY concept1, concept2
    HAVING co_occurrences > 5
    ORDER BY co_occurrences DESC;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    G = nx.from_pandas_edgelist(df, "concept1", "concept2", "co_occurrences")
    pos = nx.spring_layout(G)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            size=10,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
        ),
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in G.adjacency():
        node_adjacencies.append(len(adjacencies))
        node_text.append(f"{node}<br># of connections: {len(adjacencies)}")

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Concept Co-occurrence Network",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig


def load_database_with_graphs(db_name):
    global db
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    db_path = os.path.join(ROOT, DEFAULT_TABLES_DIR, db_name)
    if not os.path.exists(db_path):
        return f"Database {db_name} does not exist.", None
    db = ArxivDatabase(db_path)
    db.init_db()
    if db.is_db_empty:
        return (
            f"Database loaded from {db_path}, but it is empty. Please populate it with data.",
            None,
        )

    # Generate graph
    graph = generate_concept_cooccurrence_graph(db_path)

    return f"Database loaded from {db_path}", graph


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
        status = gr.Textbox(label="Status")

    with gr.Row():
        graph_output = gr.Plot(label="Concept Co-occurrence Graph")

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

    load_db_btn.click(
        load_database_with_graphs, inputs=[db_dropdown], outputs=[status, graph_output]
    )
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

demo.launch()
