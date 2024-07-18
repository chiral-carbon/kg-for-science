import click
import json
import os
import sqlite3
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_TABLES_DIR, DEFAULT_MODEL_ID, DEFAULT_INTERFACE_MODEL_ID
from src.processing.generate import get_sentences, generate_prediction
from src.utils.utils import load_model_and_tokenizer, set_env_vars


class ArxivDatabase:
    def __init__(self, db_path, model_id=None):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.model_id = model_id if model_id else DEFAULT_INTERFACE_MODEL_ID
        self.model = None
        self.tokenizer = None
        self.is_db_empty = True

    def init_db(self):
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS papers
                        (paper_id TEXT PRIMARY KEY, title TEXT, abstract TEXT, authors TEXT, 
                        categories TEXT, url TEXT, sentence_count INTEGER)"""
        )

        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS predictions
                        (id INTEGER PRIMARY KEY AUTOINCREMENT, paper_id TEXT, sentence_index INTEGER, 
                        tag_type TEXT, concept TEXT,
                        FOREIGN KEY (paper_id) REFERENCES papers(paper_id))"""
        )

        print("Database and tables created successfully.")
        self.is_db_empty = self.is_empty()

    def is_empty(self):
        try:
            self.cursor.execute("SELECT COUNT(*) FROM papers")
            count = self.cursor.fetchone()[0]
            return count == 0
        except sqlite3.OperationalError:
            return True

    def populate_db(self, data_path, predictions_path):
        papers_info = self._insert_papers(data_path)
        self._insert_predictions(predictions_path, papers_info)
        print("Database population completed.")

    def _insert_papers(self, data_path):
        papers_info = []
        with open(data_path, "r") as f:
            for line in f:
                paper = json.loads(line)
                sentence_count = len(get_sentences(paper["title"])) + len(
                    get_sentences(paper["abstract"])
                )
                papers_info.append((paper["id"], sentence_count))
                self.cursor.execute(
                    """INSERT OR REPLACE INTO papers VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        paper["id"],
                        paper["title"],
                        paper["abstract"],
                        json.dumps(paper["authors"]),
                        json.dumps(paper["categories"]),
                        paper["pdf_url"],
                        sentence_count,
                    ),
                )
        return papers_info

    def _insert_predictions(self, predictions_path, papers_info):
        with open(predictions_path, "r") as f:
            predictions = json.load(f)
            predicted_tags = predictions["predicted_tags"]

            current_paper_index = 0
            current_sentence_index = 0

            for pred in predicted_tags:
                if current_paper_index >= len(papers_info):
                    print(
                        f"Warning: More predictions than papers. Stopping at paper index {current_paper_index}"
                    )
                    break

                paper_id, sentence_count = papers_info[current_paper_index]

                for tag_type, concepts in pred.items():
                    for concept in concepts:
                        self.cursor.execute(
                            """INSERT INTO predictions (paper_id, sentence_index, tag_type, concept) 
                                        VALUES (?, ?, ?, ?)""",
                            (paper_id, current_sentence_index, tag_type, concept),
                        )

                current_sentence_index += 1
                if current_sentence_index >= sentence_count:
                    current_paper_index += 1
                    current_sentence_index = 0

    def load_model(self):
        if self.model is None:
            try:
                self.model, self.tokenizer = load_model_and_tokenizer(self.model_id)
                return f"Model {self.model_id} loaded successfully."
            except Exception as e:
                return f"Error loading model: {str(e)}"
        else:
            return "Model is already loaded."

    def natural_language_to_sql(self, question):
        system_prompt = """You are an assistant who converts natural language questions to SQL queries to query a database of scientific papers.
        The database has two tables:
        1. papers (paper_id, title, abstract, authors, categories, url, sentence_count)
        2. predictions (id, paper_id, sentence_index, tag_type, concept)
        When searching for concepts, always use the predictions table and join it with the papers table."""

        prefix = "Convert the following question to a SQL query:\n\n" "Question: "

        sql_query = generate_prediction(
            self.model, self.tokenizer, prefix, question, "sql", system_prompt
        )

        sql_query = sql_query.replace("SQL Query:", "").replace("assistant", "").strip()

        return sql_query

    def execute_query(self, sql_query):
        try:
            self.cursor.execute(sql_query)
            results = self.cursor.fetchall()
            return results if results else []
        except sqlite3.Error as e:
            return [(f"An error occurred: {e}",)]

    def query_db(self, question, is_sql):
        if self.is_db_empty:
            return "The database is empty. Please populate it with data first."

        try:
            if is_sql:
                sql_query = question.strip()
            else:
                nl_to_sql = self.natural_language_to_sql(question)
                sql_query = nl_to_sql.replace("```sql", "").replace("```", "").strip()

            results = self.execute_query(sql_query)

            output = f"SQL Query: {sql_query}\n\nResults:\n"
            if isinstance(results, list):
                if len(results) > 0:
                    for row in results:
                        output += str(row) + "\n"
                else:
                    output += "No results found."
            else:
                output += str(results)  # In case of an error message

            return output
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def close(self):
        self.conn.commit()
        self.conn.close()


def check_db_exists(db_path):
    return os.path.exists(db_path) and os.path.getsize(db_path) > 0


@click.command()
@click.option(
    "--data_path", help="Path to the data file containing the papers information."
)
@click.option("--predictions_path", help="Path to the predictions file.")
@click.option("--db_name", default="arxiv.db", help="Name of the database to create.")
@click.option(
    "--force", is_flag=True, help="Force overwrite if database already exists"
)
def main(data_path, predictions_path, db_name, force):
    set_env_vars()

    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tables_dir = os.path.join(ROOT, DEFAULT_TABLES_DIR)
    os.makedirs(tables_dir, exist_ok=True)
    db_path = os.path.join(tables_dir, db_name)

    db_exists = check_db_exists(db_path)

    db = ArxivDatabase(db_path)
    db.init_db()  # Always initialize the database structure

    if db_exists and not db.is_db_empty:
        if not force:
            print(f"Warning: The database '{db_name}' already exists and is not empty.")
            overwrite = input("Do you want to overwrite it? (y/N): ").lower().strip()
            if overwrite != "y":
                print("Operation cancelled.")
                db.close()
                return
        else:
            print(
                f"Warning: Overwriting existing database '{db_name}' due to --force flag."
            )

    db.populate_db(data_path, predictions_path)
    db.close()

    print(f"Database created and populated at: {db_path}")


if __name__ == "__main__":
    main()
