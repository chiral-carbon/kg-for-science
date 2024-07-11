import arxiv
import click
import json

@click.command()
@click.option('--search_query', default="astro-ph", help='Search query for arXiv papers')
@click.option('--max_results', default=1000, help='Maximum number of results to fetch')
@click.option('--sort_by', type=click.Choice(['relevance', 'last_updated_date', 'submitted_date']), 
              default='last_updated_date', help='Criterion to sort results by')
@click.option('--sort_order', type=click.Choice(['asc', 'desc']), default='desc', 
              help='Sort order (ascending or descending)')
@click.option('--out_file', default=None, help='Output file name')
def main(search_query, max_results, sort_by, sort_order, output):
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    if out_file is None:
        out_file = f"{search_query}_{max_results}_arxiv_papers.json"

    results = get_arxiv_papers(search_query, max_results, sort_by, sort_order)

    with open(os.path.join(data_dir, out_file), "w") as f:
        for r in results["id"]:
            f.write(json.dumps(r) + "\n")
    click.echo(f"Results written to {data_dir}/{out_file}")

def get_arxiv_papers(search_query, max_results, sort_by="relevance", sort_order="desc"):
    client = arxiv.Client()

    sort_criterion = {
        "relevance": arxiv.SortCriterion.Relevance,
        "last_updated_date": arxiv.SortCriterion.LastUpdatedDate,
        "submitted_date": arxiv.SortCriterion.SubmittedDate
    }[sort_by]

    sort_order = arxiv.SortOrder.Descending if sort_order == "desc" else arxiv.SortOrder.Ascending

    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=sort_criterion,
        sort_order=sort_order
    )

    dict_results = {"id": []}
    for result in client.results(search):
        dict_results["id"].append({
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
        })

    return dict_results

if __name__ == "__main__":
    main()