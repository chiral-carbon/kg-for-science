import arxiv
import click
import json
import os

from datetime import datetime
from tqdm import tqdm


@click.command()
@click.option(
    "--search_query", default="astro-ph", help="Search query for arXiv papers"
)
@click.option("--max_results", default=1000, help="Maximum number of results to fetch")
@click.option(
    "--sort_by",
    type=click.Choice(["relevance", "last_updated_date", "submitted_date"]),
    default="last_updated_date",
    help="Criterion to sort results by",
)
@click.option(
    "--sort_order",
    type=click.Choice(["asc", "desc"]),
    default="desc",
    help="Sort order (ascending or descending)",
)
@click.option("--out_file", default=None, help="Output file name")
@click.option(
    "--annotations_file",
    default="data/human_annotations.jsonl",
    help="File with manual annotations that is reserved",
)
def main(search_query, max_results, sort_by, sort_order, out_file, annotations_file):
    annotations = []
    with open(annotations_file, "r") as f:
        for line in f:
            annotations.append(json.loads(line))

    titles = set(ex["title"] for ex in annotations)
    assert len(titles) == len(annotations)

    results = get_arxiv_papers(search_query, max_results, sort_by, sort_order, titles)

    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    if out_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = (
            f"{search_query}_{max_results}_{sort_by}_{sort_order}_{timestamp}.jsonl"
        )

    with open(os.path.join(data_dir, out_file), "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    click.echo(f"Saved {len(results)} results to {out_file}")


def get_arxiv_papers(
    search_query, max_results, sort_by="relevance", sort_order="desc", annotated=None
):
    client = arxiv.Client()

    sort_criterion = {
        "relevance": arxiv.SortCriterion.Relevance,
        "last_updated_date": arxiv.SortCriterion.LastUpdatedDate,
        "submitted_date": arxiv.SortCriterion.SubmittedDate,
    }[sort_by]

    sort_order = (
        arxiv.SortOrder.Descending
        if sort_order == "desc"
        else arxiv.SortOrder.Ascending
    )

    search = arxiv.Search(
        query=search_query,
        max_results=None,
        sort_by=sort_criterion,
        sort_order=sort_order,
    )

    non_overlapping_results = []
    pbar = tqdm(total=max_results, desc="Fetching papers")
    for result in client.results(search):
        if result.title not in annotated:
            non_overlapping_results.append(
                {
                    "id": result.entry_id,
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "summary": result.summary,
                    "published": result.published.isoformat(),
                    "updated": result.updated.isoformat(),
                    "pdf_url": result.pdf_url,
                    "doi": result.doi,
                    "links": [link.href for link in result.links],
                    "journal_reference": result.journal_ref,
                    "primary_category": result.primary_category,
                    "categories": result.categories,
                }
            )
            pbar.update(1)
            if len(non_overlapping_results) >= max_results:
                break

    pbar.close()

    return non_overlapping_results


if __name__ == "__main__":
    main()
