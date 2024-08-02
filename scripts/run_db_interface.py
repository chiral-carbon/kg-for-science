import gradio as gr
import os
import sys
import sqlite3
import json
import re
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from create_db import ArxivDatabase
from config import DEFAULT_TABLES_DIR, DEFAULT_INTERFACE_MODEL_ID
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


def load_model():
    global db
    if db is None:
        return "Please load a database first."
    return db.load_model()


def query_db(query, is_sql, limit=None, wrap=False):
    global db
    if db is None:
        return "Please load a database first."

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

        # Process results to handle JSON fields and format output
        processed_results = []
        for row in results:
            processed_row = []
            for item in row:
                if isinstance(item, str):
                    if item.startswith("[") and item.endswith("]"):
                        try:
                            parsed_item = json.loads(item)
                            if isinstance(parsed_item, list):
                                item = ", ".join(parsed_item)
                        except json.JSONDecodeError:
                            pass
                    if column_names[len(processed_row)] == "url":
                        item = format_url(item)
                    else:
                        item = truncate_or_wrap_text(item, wrap=wrap)
                processed_row.append(item)
            processed_results.append(processed_row)

        # Create a formatted table
        table = tabulate(
            processed_results, headers=column_names, tablefmt="pipe", numalign="left"
        )

        return f"Query executed successfully. Results:\n\n{table}"
    except sqlite3.Error as e:
        return f"An error occurred: {str(e)}"


# Updated canned queries
canned_queries = [
    (
        "Modalities in Physics and Astronomy papers",
        """
     SELECT DISTINCT LOWER(concept) AS concept
     FROM predictions
     JOIN (
         SELECT paper_id, url
         FROM papers
         WHERE primary_category LIKE '%physics.space-ph%'
            OR primary_category LIKE '%astro-ph.%'
     ) AS paper_ids
     ON predictions.paper_id = paper_ids.paper_id 
     WHERE predictions.tag_type = 'modality'
     ORDER BY concept
     """,
    ),
    (
        "Datasets in Evolutionary Biology that use PDEs",
        """
    WITH pde_predictions AS (
    SELECT paper_id, concept AS pde_concept, tag_type AS pde_tag_type
    FROM predictions
    WHERE tag_type IN ('method', 'model')
      AND (
          LOWER(concept) LIKE '%pde%'
          OR LOWER(concept) LIKE '%partial differential equation%'
      )
    )
    SELECT DISTINCT 
        papers.paper_id,
        papers.url,
        LOWER(p_dataset.concept) AS dataset,
        pde_predictions.pde_concept AS pde_related_concept,
        pde_predictions.pde_tag_type AS pde_related_type
    FROM papers
    JOIN pde_predictions ON papers.paper_id = pde_predictions.paper_id
    LEFT JOIN predictions p_dataset ON papers.paper_id = p_dataset.paper_id
    WHERE papers.primary_category LIKE '%q-bio.PE%'
    AND (p_dataset.tag_type = 'dataset' OR p_dataset.tag_type IS NULL)
    ORDER BY papers.paper_id, dataset, pde_related_concept;
    """,
    ),
    (
        "Trends in objects of study in Cosmology since 2019",
        """
    SELECT 
        substr(papers.updated_on, 1, 7) as year_month, 
        predictions.concept as object, 
        COUNT(DISTINCT papers.paper_id) as paper_count 
    FROM 
        papers 
    JOIN 
        predictions ON papers.paper_id = predictions.paper_id 
    WHERE 
        papers.primary_category LIKE '%astro-ph.CO%' 
        AND predictions.tag_type = 'object' 
        AND CAST(SUBSTR(papers.updated_on, 2, 4) AS INTEGER) >= 2019
    GROUP BY 
        year_month, object 
    ORDER BY 
        year_month DESC, paper_count DESC
    """,
    ),
    (
        "New datasets in fluid dynamics since 2020",
        """
    WITH ranked_datasets AS (
    SELECT 
        p.paper_id,
        p.updated_on,
        p.primary_category,
        pred.concept AS dataset,
        ROW_NUMBER() OVER (PARTITION BY pred.concept ORDER BY p.updated_on ASC) AS rn
    FROM 
        papers p
    JOIN 
        predictions pred ON p.paper_id = pred.paper_id
    WHERE 
        pred.tag_type = 'dataset'
        AND p.primary_category LIKE '%physics.flu-dyn%'
        AND CAST(SUBSTR(p.updated_on, 2, 4) AS INTEGER) >= 2020 
    )
    SELECT 
        paper_id,
        updated_on,
        primary_category,
        dataset
    FROM 
        ranked_datasets
    WHERE 
        rn = 1
    ORDER BY 
        updated_on ASC
    LIMIT 1000
    """,
    ),
    (
        "Evolutionary biology datasets that use spatiotemporal dynamics",
        """
    WITH evo_bio_papers AS (
    SELECT paper_id
    FROM papers
    WHERE primary_category LIKE '%q-bio.PE%' 
    ),
    spatiotemporal_keywords AS (
        SELECT 'spatio-temporal' AS keyword
        UNION SELECT 'spatiotemporal'
        UNION SELECT 'spatial and temporal'
        UNION SELECT 'space-time'
        UNION SELECT 'geographic distribution'
        UNION SELECT 'phylogeograph'
        UNION SELECT 'biogeograph'
        UNION SELECT 'dispersal'
        UNION SELECT 'migration'
        UNION SELECT 'range expansion'
        UNION SELECT 'population dynamics'
    )
    SELECT DISTINCT
        p.paper_id,
        p.updated_on,
        p.abstract,
        d.concept AS dataset,
        GROUP_CONCAT(DISTINCT stk.keyword) AS spatiotemporal_keywords_found
    FROM 
        evo_bio_papers ebp
    JOIN 
        papers p ON ebp.paper_id = p.paper_id
    JOIN 
        predictions d ON p.paper_id = d.paper_id
    JOIN 
        predictions st ON p.paper_id = st.paper_id
    JOIN 
        spatiotemporal_keywords stk
    WHERE 
        d.tag_type = 'dataset'
        AND st.tag_type = 'modality'
        AND LOWER(st.concept) LIKE '%' || stk.keyword || '%'
    GROUP BY 
        p.paper_id, p.updated_on, p.abstract, d.concept
    ORDER BY 
        p.updated_on DESC
    """,
    ),
    (
        "What percentage of papers use only galaxy or spectra, or both or neither?",
        """
    WITH paper_modalities AS (
    SELECT 
        p.paper_id,
        MAX(CASE WHEN LOWER(pred.concept) LIKE '%galaxy%' THEN 1 ELSE 0 END) AS uses_galaxy_images,
        MAX(CASE WHEN LOWER(pred.concept) LIKE '%spectr%' THEN 1 ELSE 0 END) AS uses_spectra
    FROM 
        papers p
    LEFT JOIN 
        predictions pred ON p.paper_id = pred.paper_id
    WHERE 
            p.primary_category LIKE '%astro-ph%'
            AND pred.tag_type = 'modality'
        GROUP BY 
            p.paper_id
    ),
    categorized_papers AS (
        SELECT
            CASE
                WHEN uses_galaxy_images = 1 AND uses_spectra = 1 THEN 'Both'
                WHEN uses_galaxy_images = 1 THEN 'Only Galaxy Images'
                WHEN uses_spectra = 1 THEN 'Only Spectra'
                ELSE 'Neither'
            END AS category,
            COUNT(*) AS paper_count
        FROM 
            paper_modalities
        GROUP BY 
            CASE
                WHEN uses_galaxy_images = 1 AND uses_spectra = 1 THEN 'Both'
                WHEN uses_galaxy_images = 1 THEN 'Only Galaxy Images'
                WHEN uses_spectra = 1 THEN 'Only Spectra'
                ELSE 'Neither'
            END
    )
    SELECT
        category,
        paper_count,
        ROUND(CAST(paper_count AS FLOAT) / (SELECT SUM(paper_count) FROM categorized_papers) * 100, 2) AS percentage
    FROM
        categorized_papers
    ORDER BY
        paper_count DESC
    """,
    ),
    (
        "If we include timeseries or the next biggest data modality, how much does coverage change?",
        """
    WITH modality_counts AS (
    SELECT 
        LOWER(concept) AS modality,
        COUNT(DISTINCT paper_id) AS usage_count
    FROM 
        predictions
    WHERE 
        tag_type = 'modality'
        AND LOWER(concept) NOT LIKE '%galaxy image%'
        AND LOWER(concept) NOT LIKE '%spectr%'
    GROUP BY 
        LOWER(concept)
    ORDER BY 
        usage_count DESC
    LIMIT 1
    ),
    paper_modalities AS (
        SELECT 
            p.paper_id,
            MAX(CASE WHEN LOWER(pred.concept) LIKE '%galaxy image%' THEN 1 ELSE 0 END) AS uses_galaxy_images,
            MAX(CASE WHEN LOWER(pred.concept) LIKE '%spectr%' THEN 1 ELSE 0 END) AS uses_spectra,
            MAX(CASE WHEN LOWER(pred.concept) LIKE (SELECT '%' || modality || '%' FROM modality_counts) THEN 1 ELSE 0 END) AS uses_third_modality
        FROM 
            papers p
        LEFT JOIN 
            predictions pred ON p.paper_id = pred.paper_id
        WHERE 
            p.primary_category LIKE '%astro-ph%'
            AND pred.tag_type = 'modality'
        GROUP BY 
            p.paper_id
    ),
    coverage_before AS (
        SELECT
            SUM(CASE WHEN uses_galaxy_images = 1 OR uses_spectra = 1 THEN 1 ELSE 0 END) AS covered_papers,
            COUNT(*) AS total_papers
        FROM 
            paper_modalities
    ),
    coverage_after AS (
        SELECT
            SUM(CASE WHEN uses_galaxy_images = 1 OR uses_spectra = 1 OR uses_third_modality = 1 THEN 1 ELSE 0 END) AS covered_papers,
            COUNT(*) AS total_papers
        FROM 
            paper_modalities
    )
    SELECT
        (SELECT modality FROM modality_counts) AS third_modality,
        ROUND(CAST(covered_papers AS FLOAT) / total_papers * 100, 2) AS coverage_before_percent,
        ROUND(CAST((SELECT covered_papers FROM coverage_after) AS FLOAT) / total_papers * 100, 2) AS coverage_after_percent,
        ROUND(CAST((SELECT covered_papers FROM coverage_after) AS FLOAT) / total_papers * 100, 2) - 
        ROUND(CAST(covered_papers AS FLOAT) / total_papers * 100, 2) AS coverage_increase_percent
    FROM
        coverage_before
    """,
    ),
    (
        "List all papers",
        "SELECT paper_id, abstract AS abstract_preview, authors, primary_category FROM papers",
    ),
    (
        "Count papers by category",
        "SELECT primary_category, COUNT(*) as paper_count FROM papers GROUP BY primary_category ORDER BY paper_count DESC",
    ),
    (
        "Top authors with most papers",
        """
     WITH author_papers AS (
         SELECT json_each.value AS author
         FROM papers, json_each(papers.authors)
     )
     SELECT author, COUNT(*) as paper_count 
     FROM author_papers 
     GROUP BY author 
     ORDER BY paper_count DESC
     """,
    ),
    (
        "Papers with 'quantum' in abstract",
        "SELECT paper_id, abstract AS abstract_preview FROM papers WHERE abstract LIKE '%quantum%'",
    ),
    (
        "Most common concepts",
        "SELECT concept, COUNT(*) as concept_count FROM predictions GROUP BY concept ORDER BY concept_count DESC",
    ),
    (
        "Papers with multiple authors",
        """
     SELECT paper_id, json_array_length(authors) as author_count, authors
     FROM papers
     WHERE json_array_length(authors) > 1
     ORDER BY author_count DESC
     """,
    ),
]

css = """
#Results {font-family: monospace;}
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

    # with gr.Row():
    #     model_id_input = gr.Textbox(label="Model ID (optional)", placeholder=f"Default: {DEFAULT_INTERFACE_MODEL_ID}", scale=2)
    #     load_model_btn = gr.Button("Load Model (ETA: 1 min)", size="sm", scale=1)
    #     nl_input = gr.Textbox(label="Natural Language Query", lines=2, scale=3)
    #     nl_submit = gr.Button("Submit NL Query", size="sm", scale=1)

    output = gr.Textbox(label="Results", lines=20, elem_id="Results", scale=4)

    def update_selected_query(query_description):
        for desc, sql in canned_queries:
            if desc == query_description:
                return sql
        return ""

    def submit_canned_query(query_description, limit, wrap):
        for desc, sql in canned_queries:
            if desc == query_description:
                return query_db(sql, True, str(limit), wrap)
        return "Selected query not found."

    def submit_nl_query(nl_query):
        # Placeholder for NL to SQL conversion
        return "Natural Language query processing not implemented yet."

    load_db_btn.click(load_database, inputs=[db_dropdown], outputs=status)
    # load_model_btn.click(load_model, outputs=status)
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
    # nl_submit.click(submit_nl_query, inputs=[nl_input], outputs=output)

if __name__ == "__main__":
    set_env_vars()
    demo.launch(share=True)

# Add this line at the very end of your script, outside of any function or class
demo.launch()
