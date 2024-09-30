import gradio as gr
import os
import json
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import re
import sys
import sqlite3
import tempfile
import time
import uvicorn

from contextlib import contextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from gradio.routes import mount_gradio_app
from plotly.subplots import make_subplots
from tabulate import tabulate
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEFAULT_TABLES_DIR,
    DEFAULT_INTERFACE_MODEL_ID,
    COOCCURRENCE_QUERY,
    canned_queries,
)
from scripts.create_db import ArxivDatabase
from src.utils.utils import set_env_vars

set_env_vars()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db: Optional[ArxivDatabase] = None

last_update_time = 0
update_delay = 0.5  # Delay in seconds


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
    return [
        f
        for f in os.listdir(tables_dir)
        if not f.endswith(".db") and not f.endswith(".md")
    ]


def query_db(query, is_sql, limit=None, wrap=False):
    global db
    if db is None:
        return pd.DataFrame({"Error": ["Please load a database first."]})

    try:
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()

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
        return pd.DataFrame({"Error": [f"Database error: {str(e)}"]})
    except Exception as e:
        return pd.DataFrame({"Error": [f"An unexpected error occurred: {str(e)}"]})


def generate_concept_cooccurrence_graph(db_path, tag_type=None):
    conn = sqlite3.connect(db_path)

    query = COOCCURRENCE_QUERY
    if tag_type and tag_type != "All":
        query = query.replace(
            "WHERE p1.tag_type = p2.tag_type",
            f"WHERE p1.tag_type = p2.tag_type AND p1.tag_type = '{tag_type}'",
        )

    df = pd.read_sql_query(query, conn)
    conn.close()

    G = nx.from_pandas_edgelist(df, "concept1", "concept2", "co_occurrences")
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    edge_trace = go.Scatter(
        x=[], y=[], line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines"
    )

    node_trace = go.Scatter(
        x=[],
        y=[],
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

    def update_traces(selected_node=None, depth=0):
        nonlocal edge_trace, node_trace

        if selected_node and depth > 0:
            nodes_to_show = set([selected_node])
            frontier = set([selected_node])
            for _ in range(depth):
                new_frontier = set()
                for node in frontier:
                    new_frontier.update(G.neighbors(node))
                nodes_to_show.update(new_frontier)
                frontier = new_frontier
            sub_G = G.subgraph(nodes_to_show)
        else:
            sub_G = G

        edge_x, edge_y = [], []
        for edge in sub_G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace.x = edge_x
        edge_trace.y = edge_y

        node_x, node_y = [], []
        for node in sub_G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace.x = node_x
        node_trace.y = node_y

        node_adjacencies = []
        node_text = []
        for node in sub_G.nodes():
            adjacencies = list(G.adj[node])
            node_adjacencies.append(len(adjacencies))
            node_text.append(f"{node}<br># of connections: {len(adjacencies)}")

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

    update_traces()

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f'Concept Co-occurrence Network {f"({tag_type})" if tag_type and tag_type != "All" else ""}',
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

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"visible": [True, True]}],
                        label="Full Graph",
                        method="update",
                    ),
                    dict(
                        args=[
                            {
                                "visible": [True, True],
                                "xaxis.range": [-1, 1],
                                "yaxis.range": [-1, 1],
                            }
                        ],
                        label="Core View",
                        method="relayout",
                    ),
                    dict(
                        args=[
                            {
                                "visible": [True, True],
                                "xaxis.range": [-0.2, 0.2],
                                "yaxis.range": [-0.2, 0.2],
                            }
                        ],
                        label="Detailed View",
                        method="relayout",
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top",
            ),
        ]
    )

    return fig, G, pos, update_traces


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

    graph, _, _, _ = generate_concept_cooccurrence_graph(db_path)
    return f"Database loaded from {db_path}", graph


css = """
#selected-query {
    max-height: 100px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}
"""


def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# ArXiv Database Query Interface")

        with gr.Row():
            db_dropdown = gr.Dropdown(
                choices=get_available_databases(),
                label="Select Database",
                value=get_available_databases(),
            )
            # load_db_btn = gr.Button("Load Database", size="sm")
            status = gr.Textbox(label="Status")

        with gr.Row():
            graph_output = gr.Plot(label="Concept Co-occurrence Graph")

        with gr.Row():
            tag_type_dropdown = gr.Dropdown(
                choices=[
                    "All",
                    "model",
                    "task",
                    "dataset",
                    "field",
                    "modality",
                    "method",
                    "object",
                    "property",
                    "instrument",
                ],
                label="Select Tag Type",
                value="All",
            )
            highlight_input = gr.Textbox(label="Highlight Concepts (comma-separated)")

        with gr.Row():
            node_dropdown = gr.Dropdown(label="Select Node", choices=[])
            depth_slider = gr.Slider(
                minimum=0, maximum=5, step=1, value=0, label="Connection Depth"
            )
            update_graph_button = gr.Button("Update Graph")

        with gr.Row():
            wrap_checkbox = gr.Checkbox(label="Wrap long text", value=False)
            canned_query_dropdown = gr.Dropdown(
                choices=[q[0] for q in canned_queries], label="Select Query", scale=3
            )
            limit_input = gr.Number(
                label="Limit", value=10000, step=1, minimum=1, scale=1
            )
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

        with gr.Row():
            nl_query_input = gr.Textbox(
                label="Natural Language Query", lines=2, scale=4
            )
            nl_query_submit = gr.Button("Convert to SQL", size="sm", scale=1)

        output = gr.DataFrame(label="Results", wrap=True)

        with gr.Row():
            copy_button = gr.Button("Copy as Markdown")
            download_button = gr.Button("Download as CSV")

        def debounced_update_graph(
            db_name, tag_type, highlight_concepts, selected_node, depth
        ):
            global last_update_time

            current_time = time.time()
            if current_time - last_update_time < update_delay:
                return None, []  # Return early if not enough time has passed

            last_update_time = current_time

            if not db_name:
                return None, []

            ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            db_path = os.path.join(ROOT, DEFAULT_TABLES_DIR, db_name)
            fig, G, pos, update_traces = generate_concept_cooccurrence_graph(
                db_path, tag_type
            )

            if isinstance(selected_node, list):
                selected_node = selected_node[0] if selected_node else None

            highlight_nodes = (
                [node.strip() for node in highlight_concepts.split(",")]
                if highlight_concepts
                else []
            )
            primary_node = highlight_nodes[0] if highlight_nodes else None

            if primary_node and primary_node in G.nodes():
                # Apply node selection and depth filter
                nodes_to_show = set([primary_node])
                if depth > 0:
                    frontier = set([primary_node])
                    for _ in range(depth):
                        new_frontier = set()
                        for node in frontier:
                            new_frontier.update(G.neighbors(node))
                        nodes_to_show.update(new_frontier)
                        frontier = new_frontier

                sub_G = G.subgraph(nodes_to_show)

                # Update traces with the filtered graph
                edge_x, edge_y = [], []
                for edge in sub_G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                fig.data[0].x = edge_x
                fig.data[0].y = edge_y

                node_x, node_y = [], []
                for node in sub_G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)

                fig.data[1].x = node_x
                fig.data[1].y = node_y

                # Color nodes based on their distance from the primary node and highlight status
                node_colors = []
                node_sizes = []
                for node in sub_G.nodes():
                    if node in highlight_nodes:
                        node_colors.append(
                            "rgba(255,0,0,1)"
                        )  # Red for highlighted nodes
                        node_sizes.append(15)
                    else:
                        distance = nx.shortest_path_length(
                            sub_G, source=primary_node, target=node
                        )
                        intensity = max(0, 1 - (distance / (depth + 1)))
                        node_colors.append(f"rgba(0,0,255,{intensity})")
                        node_sizes.append(10)

                fig.data[1].marker.color = node_colors
                fig.data[1].marker.size = node_sizes

                # Update node text
                node_text = [
                    f"{node}<br># of connections: {len(list(G.neighbors(node)))}"
                    for node in sub_G.nodes()
                ]
                fig.data[1].text = node_text

                # Get connected nodes for dropdown
                connected_nodes = sorted(list(G.neighbors(primary_node)))
            else:
                # If no primary node or it's not in the graph, show the full graph
                connected_nodes = sorted(list(G.nodes()))

            return fig, connected_nodes

        def update_node_dropdown(highlight_concepts):
            if not highlight_concepts or not db:
                return gr.Dropdown(choices=[])

            ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            db_path = os.path.join(ROOT, DEFAULT_TABLES_DIR, db.db_path)
            _, G, _, _ = generate_concept_cooccurrence_graph(db_path)

            primary_node = highlight_concepts.split(",")[0].strip()
            if primary_node in G.nodes():
                connected_nodes = sorted(list(G.neighbors(primary_node)))
                return gr.Dropdown(choices=connected_nodes)
            else:
                return gr.Dropdown(choices=[])

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

        def copy_as_markdown(df):
            return df.to_markdown()

        def download_as_csv(df):
            if df is None or df.empty:
                return None

            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".csv"
            ) as temp_file:
                df.to_csv(temp_file.name, index=False)
                temp_file_path = temp_file.name

            return temp_file_path

        def nl_to_sql(nl_query):
            # Placeholder function for natural language to SQL conversion
            return f"SELECT * FROM papers WHERE abstract LIKE '%{nl_query}%' LIMIT 10;"

        db_dropdown.change(
            load_database_with_graphs,
            inputs=[db_dropdown],
            outputs=[status, graph_output],
        )

        # db_dropdown.change(
        #     debounced_update_graph,
        #     inputs=[db_dropdown, tag_type_dropdown, highlight_input, node_dropdown, depth_slider],
        #     outputs=[graph_output, node_dropdown],
        # )

        tag_type_dropdown.change(
            debounced_update_graph,
            inputs=[
                db_dropdown,
                tag_type_dropdown,
                highlight_input,
                node_dropdown,
                depth_slider,
            ],
            outputs=[graph_output, node_dropdown],
        )

        highlight_input.change(
            update_node_dropdown,
            inputs=[highlight_input],
            outputs=[node_dropdown],
        )
        # node_dropdown.change(
        #     debounced_update_graph,
        #     inputs=[db_dropdown, tag_type_dropdown, highlight_input, node_dropdown, depth_slider],
        #     outputs=[graph_output, node_dropdown],
        # )

        # depth_slider.change(
        #     debounced_update_graph,
        #     inputs=[db_dropdown, tag_type_dropdown, highlight_input, node_dropdown, depth_slider],
        #     outputs=[graph_output, node_dropdown],
        # )
        update_graph_button.click(
            debounced_update_graph,
            inputs=[
                db_dropdown,
                tag_type_dropdown,
                highlight_input,
                node_dropdown,
                depth_slider,
            ],
            outputs=[graph_output, node_dropdown],
        )
        canned_query_dropdown.change(
            update_selected_query,
            inputs=[canned_query_dropdown],
            outputs=[selected_query],
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
        copy_button.click(
            copy_as_markdown,
            inputs=[output],
            outputs=[gr.Textbox(label="Markdown Output", show_copy_button=True)],
        )
        download_button.click(
            download_as_csv, inputs=[output], outputs=[gr.File(label="CSV Output")]
        )
        # nl_query_submit.click(nl_to_sql, inputs=[nl_query_input], outputs=[sql_input])

    return demo


demo = create_demo()

# def close_db():
#     global db
#     if db is not None:
#         db.close()
#         db = None


def launch():
    print("Launching Gradio app...", flush=True)
    shared_demo = demo.launch(share=True, prevent_thread_lock=True)

    if isinstance(shared_demo, tuple):
        if len(shared_demo) >= 2:
            local_url, share_url = shared_demo[:2]
        else:
            local_url, share_url = shared_demo[0], "N/A"
    else:
        local_url = getattr(shared_demo, "local_url", "N/A")
        share_url = getattr(shared_demo, "share_url", "N/A")

    print(f"Local URL: {local_url}", flush=True)
    print(f"Shareable link: {share_url}", flush=True)

    print(
        "Gradio app launched.",
        flush=True,
    )

    # Keep the script running
    demo.block_thread()


if __name__ == "__main__":
    launch()

# Mount the Gradio app
# app = mount_gradio_app(app, demo, path="/")

# print(f"Shareable link: {demo.share_url}")

# @app.exception_handler(Exception)
# async def exception_handler(request: Request, exc: Exception):
#     print(f"An error occurred: {str(exc)}")
#     return {"error": str(exc)}

# @contextmanager
# def get_db_connection():
#     global db
#     conn = db.conn.cursor().connection
#     try:
#         yield conn
#     finally:
#         conn.close()

# @app.on_event("startup")
# async def startup_event():
#     global db
#     ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#     db_path = os.path.join(ROOT, DEFAULT_TABLES_DIR, get_available_databases()[0])  # Use the first available database
#     db = ArxivDatabase(db_path)
#     db.init_db()

# @app.on_event("shutdown")
# async def shutdown_event():
#     if db is not None:
#         db.close()


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=7860)
