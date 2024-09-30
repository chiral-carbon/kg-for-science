import gradio as gr
import os
import json
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import re
import sys
import sqlite3
import time

from flask import jsonify
from plotly.subplots import make_subplots
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEFAULT_TABLES_DIR,
    DEFAULT_INTERFACE_MODEL_ID,
    COOCCURRENCE_QUERY,
    canned_queries,
)
from scripts.create_db import ArxivDatabase
from src.utils.utils import set_env_vars

db = None


def truncate_or_wrap_text(text, max_length=50, wrap=False):
    if wrap:
        return "\n".join(
            text[i : i + max_length] for i in range(0, len(text), max_length)
        )
    return text[:max_length] + "..." if len(text) > max_length else text


def format_url(url):
    return url.split("/")[-1] if url.startswith("http") else url


def get_available_databases():
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tables_dir = os.path.join(ROOT, DEFAULT_TABLES_DIR)
    return [f for f in os.listdir(tables_dir) if f.endswith(".db")]


def query_db(query, is_sql, limit=None, wrap=False):
    global db
    if db is None:
        return pd.DataFrame({"Error": ["Please load a database first."]})

    try:
        cursor = db.conn.cursor()
        query = " ".join(query.strip().split("\n")).rstrip(";")
        if limit is not None:
            if "LIMIT" in query.upper():
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


def get_graph_data(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(COOCCURRENCE_QUERY, conn)
    conn.close()

    nodes = set()
    for _, row in df.iterrows():
        nodes.add((row["concept1"], row["tag_type"]))
        nodes.add((row["concept2"], row["tag_type"]))

    graph_data = {
        "nodes": [{"id": node[0], "type": node[1]} for node in nodes],
        "links": [
            {
                "source": row["concept1"],
                "target": row["concept2"],
                "value": row["co_occurrences"],
            }
            for _, row in df.iterrows()
        ],
    }
    return json.dumps(graph_data)


def generate_concept_cooccurrence_graph(db_path):
    return get_graph_data(db_path)


def load_database_with_graphs(db_name):
    global db
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    db_path = os.path.join(ROOT, DEFAULT_TABLES_DIR, db_name)
    print(f"Attempting to load database from: {db_path}")
    if not os.path.exists(db_path):
        print(f"Database {db_name} does not exist.")
        return f"Database {db_name} does not exist.", None
    db = ArxivDatabase(db_path)
    db.init_db()
    if db.is_db_empty:
        print(f"Database loaded from {db_path}, but it is empty.")
        return (
            f"Database loaded from {db_path}, but it is empty. Please populate it with data.",
            None,
        )

    graph_data = get_graph_data(db_path)
    print("Graph data generated:")
    print(graph_data)  # Print first 1000 characters of graph data
    return f"Database loaded from {db_path}", graph_data


js = """
let simulation;

function createGraph(graphData) {
    console.log("Creating graph with data:", graphData);
    if (!graphData) {
        console.error("No graph data provided");
        return;
    }
    
    let data;
    try {
        data = JSON.parse(graphData);
        console.log("Parsed data:", data);
    } catch (error) {
        console.error("Error parsing graph data:", error);
        return;
    }

    const svg = d3.select("#graph-svg");
    if (svg.empty()) {
        console.error("SVG element not found");
        return;
    }
    
    svg.selectAll("*").remove();
    
    const width = svg.node().getBoundingClientRect().width;
    const height = svg.node().getBoundingClientRect().height;
    console.log("SVG dimensions:", width, height);

    // Create the simulation with all the forces
    simulation = d3.forceSimulation(data.nodes)
        .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collide", d3.forceCollide(20))
        .force("x", d3.forceX())
        .force("y", d3.forceY());

    const link = svg.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(data.links)
        .enter().append("line")
        .attr("stroke-width", d => Math.sqrt(d.value))
        .attr("stroke", "#999");

    const node = svg.append("g")
        .attr("class", "nodes")
        .selectAll("circle")
        .data(data.nodes)
        .enter().append("circle")
        .attr("r", 5)
        .attr("fill", d => d.type === "object" ? "#ff0000" : "#0000ff")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    node.append("title")
        .text(d => `${d.id} (${d.type})`);

    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);
    });

    // Zoom functionality
    const zoom = d3.zoom()
        .scaleExtent([0.1, 10])
        .on("zoom", (event) => {
            svg.selectAll("g").attr("transform", event.transform);
        });

    svg.call(zoom);

    console.log("Graph rendering complete");
}

function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

function updateGraph(graphData) {
    console.log("Updating graph with new data");
    createGraph(graphData);
}

// Attempt to create the graph when the script loads
document.addEventListener('DOMContentLoaded', (event) => {
    console.log("DOM fully loaded");
    const graphDataElement = document.getElementById('graph-data');
    if (graphDataElement) {
        console.log("Found graph-data element");
        const graphData = graphDataElement.textContent || graphDataElement.value;
        if (graphData) {
            console.log("Initial graph data found, rendering...");
            createGraph(graphData);
        } else {
            console.log("No initial graph data found");
        }
    } else {
        console.log("Could not find graph-data element");
    }
});

// Set up a MutationObserver to watch for changes in the graph data
const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        if (mutation.type === 'childList' || (mutation.type === 'attributes' && mutation.attributeName === 'value')) {
            const graphDataElement = document.getElementById('graph-data');
            if (graphDataElement) {
                const graphData = graphDataElement.textContent || graphDataElement.value;
                console.log("New graph data detected:", graphData);
                if (graphData) {
                    createGraph(graphData);
                }
            }
        }
    });
});

// Start observing the graph-data element once the DOM is loaded
document.addEventListener('DOMContentLoaded', (event) => {
    const graphDataElement = document.getElementById('graph-data');
    if (graphDataElement) {
        console.log("Found graph-data element, starting observer");
        observer.observe(graphDataElement, { childList: true, subtree: true, attributes: true });
    } else {
        console.error("Could not find graph-data element for observation");
    }
});
"""

css = """
#graph-container {
    border: 2px solid #ccc;
    margin: 10px;
    padding: 10px;
    height: 500px;
    width: 100%;
}
#graph-svg {
    width: 100%;
    height: 100%;
}
#selected-query {
    max-height: 100px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}
"""


def create_demo():
    with gr.Blocks(css=css) as demo:
        gr.Markdown("# ArXiv Database Query Interface")

        with gr.Row():
            db_dropdown = gr.Dropdown(
                choices=get_available_databases(),
                label="Select Database",
                value=(
                    get_available_databases()[0] if get_available_databases() else None
                ),
            )
            status = gr.Textbox(label="Status")

        with gr.Row():
            graph_container = gr.HTML(
                f"""
                <div id="graph-container">
                    <svg id="graph-svg"></svg>
                </div>
            """
            )
            graph_data = gr.JSON(
                label="Graph Data", visible=False, elem_id="graph-data"
            )

        gr.HTML(
            f"""
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <script>{js}</script>
        """
        )

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

        def load_selected_database(db_name):
            if db_name:
                return load_database_with_graphs(db_name)
            return "No database selected.", None

        db_dropdown.change(
            load_database_with_graphs,
            inputs=[db_dropdown],
            outputs=[status, graph_data],
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

    return demo


demo = create_demo()


def launch():
    print("Launching Gradio app...", flush=True)
    demo.launch(share=True)
    print(
        "Gradio app launched. If you don't see a URL above, there might be network restrictions.",
        flush=True,
    )


if __name__ == "__main__":
    launch()
