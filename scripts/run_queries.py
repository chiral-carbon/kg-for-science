import sqlite3
from collections import defaultdict
import argparse


def get_tags_with_papers(db_path, tag_type):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
    SELECT pred.concept, p.paper_id, p.title
    FROM predictions pred
    JOIN papers p ON pred.paper_id = p.paper_id
    WHERE pred.tag_type = ?
    """

    cursor.execute(query, (tag_type,))
    results = cursor.fetchall()

    conn.close()

    tag_dict = defaultdict(list)
    for concept, paper_id, title in results:
        tag_dict[concept].append((paper_id, title))

    return tag_dict


def compare_tags(db1_path, db2_path, tag_type):
    db1_tags = get_tags_with_papers(db1_path, tag_type)
    db2_tags = get_tags_with_papers(db2_path, tag_type)

    all_tags = set(db1_tags.keys()) | set(db2_tags.keys())

    common_tags = []
    only_in_db1 = []
    only_in_db2 = []

    for tag in all_tags:
        papers1 = db1_tags.get(tag, [])
        papers2 = db2_tags.get(tag, [])

        if papers1 and papers2:
            common_tags.append((tag, papers1, papers2))
        elif papers1:
            only_in_db1.append((tag, papers1))
        else:
            only_in_db2.append((tag, papers2))

    return common_tags, only_in_db1, only_in_db2


def print_results(common_tags, only_in_db1, only_in_db2, tag_type):
    print(f"Comparison of '{tag_type}' tags between two databases:")
    print(
        f"Total unique tags in DB1: {len(set(t[0] for t in common_tags + only_in_db1))}"
    )
    print(
        f"Total unique tags in DB2: {len(set(t[0] for t in common_tags + only_in_db2))}"
    )
    print(f"Tags common to both databases: {len(common_tags)}")
    print(f"Tags only in DB1: {len(only_in_db1)}")
    print(f"Tags only in DB2: {len(only_in_db2)}")

    print("\nSample of common tags:")
    for tag, papers1, papers2 in sorted(
        common_tags, key=lambda x: len(x[1]) + len(x[2]), reverse=True
    )[:5]:
        print(f"Tag: {tag}")
        print(f"  Count in DB1: {len(papers1)}")
        print(f"  Count in DB2: {len(papers2)}")
        print("  Sample papers from DB1:")
        for paper_id, title in papers1[:2]:
            print(f"    - {paper_id}: {title}")
        print("  Sample papers from DB2:")
        for paper_id, title in papers2[:2]:
            print(f"    - {paper_id}: {title}")
        print()

    print("\nSample of tags only in DB1:")
    for tag, papers in sorted(only_in_db1, key=lambda x: len(x[1]), reverse=True)[:3]:
        print(f"Tag: {tag}, Count: {len(papers)}")
        print("  Sample papers:")
        for paper_id, title in papers[:2]:
            print(f"    - {paper_id}: {title}")

    print("\nSample of tags only in DB2:")
    for tag, papers in sorted(only_in_db2, key=lambda x: len(x[1]), reverse=True)[:3]:
        print(f"Tag: {tag}, Count: {len(papers)}")
        print("  Sample papers:")
        for paper_id, title in papers[:2]:
            print(f"    - {paper_id}: {title}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare tag types between two databases."
    )
    parser.add_argument("db1_path", nargs="?", help="Path to the first database")
    parser.add_argument("db2_path", nargs="?", help="Path to the second database")
    parser.add_argument(
        "tag_type",
        nargs="?",
        help="Tag type to compare (e.g., 'modality', 'method', etc.)",
    )
    parser.add_argument(
        "--db1_path", dest="db1_path_option", help="Path to the first database"
    )
    parser.add_argument(
        "--db2_path", dest="db2_path_option", help="Path to the second database"
    )
    parser.add_argument(
        "--tag_type",
        dest="tag_type_option",
        help="Tag type to compare (e.g., 'modality', 'method', etc.)",
    )
    args = parser.parse_args()

    # Use named arguments if provided, otherwise use positional arguments
    db1_path = args.db1_path_option or args.db1_path
    db2_path = args.db2_path_option or args.db2_path
    tag_type = args.tag_type_option or args.tag_type

    if not all([db1_path, db2_path, tag_type]):
        parser.error(
            "All three arguments (db1_path, db2_path, and tag_type) are required."
        )

    common_tags, only_in_db1, only_in_db2 = compare_tags(db1_path, db2_path, tag_type)
    print_results(common_tags, only_in_db1, only_in_db2, tag_type)


if __name__ == "__main__":
    main()
