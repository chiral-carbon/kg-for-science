import sqlite3
import json
from rake_nltk import Rake
from collections import Counter
import os
import random
import sys
import uuid
from datetime import datetime
import argparse

# Initialize RAKE
r = Rake()

parser = argparse.ArgumentParser(description='Compare RAKE and LLM extractions for randomly selected papers from each domain')
parser.add_argument('-n', type=int, default=3, help='Number of papers to analyze')
args = parser.parse_args()

# Database paths - adjusted for your environment
database_paths = {
    "astrophysics_json": "/scratch/ad6489/kg-for-science/data/databases/Astrophysics (response=JSON)",
    "astrophysics_readable": "/scratch/ad6489/kg-for-science/data/databases/Astrophysics (response=readable)",
    "fluid_dynamics_json": "/scratch/ad6489/kg-for-science/data/databases/Fluid Dynamics (response=JSON)",
    "fluid_dynamics_readable": "/scratch/ad6489/kg-for-science/data/databases/Fluid Dynamics (response=readable)",
    "evolutionary_biology_json": "/scratch/ad6489/kg-for-science/data/databases/Evolutionary Biology (response=JSON)",
    "evolutionary_biology_readable": "/scratch/ad6489/kg-for-science/data/databases/Evolutionary Biology (response=readable)"
}

# Domain mapping
domains = {
    "astrophysics": ("astrophysics_json", "astrophysics_readable"),
    "fluid_dynamics": ("fluid_dynamics_json", "fluid_dynamics_readable"),
    "evolutionary_biology": ("evolutionary_biology_json", "evolutionary_biology_readable")
}

# Create log file with UUID
run_id = str(uuid.uuid4())
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"extraction_comparison_{run_id}.log")

# Create .gitignore if it doesn't exist
gitignore_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gitignore")
if not os.path.exists(gitignore_path):
    with open(gitignore_path, 'w') as gitignore:
        gitignore.write("logs/\n")
elif "logs/" not in open(gitignore_path).read():
    with open(gitignore_path, 'a') as gitignore:
        gitignore.write("logs/\n")

# Redirect stdout to log file
log_handle = open(log_file, 'w')
sys.stdout = log_handle

def get_random_papers(domain, count=3):
    """
    Get random papers from a specific domain
    """
    json_db, readable_db = domains[domain]
    
    try:
        # Try JSON database first
        conn = sqlite3.connect(database_paths[json_db])
        cursor = conn.cursor()
        
        # Get papers that have predictions
        cursor.execute("""
            SELECT DISTINCT p.paper_id 
            FROM papers p
            JOIN predictions pr ON p.paper_id = pr.paper_id
            LIMIT 100
        """)
        all_papers = cursor.fetchall()
        conn.close()
        
        if not all_papers:
            # Fall back to readable database
            conn = sqlite3.connect(database_paths[readable_db])
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT p.paper_id 
                FROM papers p
                JOIN predictions pr ON p.paper_id = pr.paper_id
                LIMIT 100
            """)
            all_papers = cursor.fetchall()
            conn.close()
        
        # Select random papers
        if len(all_papers) <= count:
            return [p[0] for p in all_papers]
        else:
            return [p[0] for p in random.sample(all_papers, count)]
            
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []

def get_paper_data_and_llm_concepts(paper_id, db_path):
    """
    Retrieve paper data and LLM extracted concepts from the database
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get paper data
        cursor.execute("""
            SELECT paper_id, abstract, authors, primary_category, url, updated_on, sentence_count
            FROM papers 
            WHERE paper_id = ?
        """, (paper_id,))
        paper_data = cursor.fetchone()
        
        if not paper_data:
            return None, []
        
        # Get LLM extracted concepts
        cursor.execute("""
            SELECT sentence_index, tag_type, concept 
            FROM predictions 
            WHERE paper_id = ?
            ORDER BY sentence_index
        """, (paper_data[0],))
        llm_concepts = cursor.fetchall()
        
        # Debug info
        if not llm_concepts:
            print(f"WARNING: No LLM concepts found for paper {paper_id} in database {db_path}")
            # Check if predictions table exists and has data
            cursor.execute("SELECT COUNT(*) FROM predictions")
            pred_count = cursor.fetchone()[0]
            print(f"Total predictions in database: {pred_count}")
            
            # Check if this paper_id exists in predictions
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE paper_id = ?", (paper_data[0],))
            paper_pred_count = cursor.fetchone()[0]
            print(f"Predictions for this paper: {paper_pred_count}")
        
        conn.close()
        return paper_data, llm_concepts
    
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None, []

def get_rake_concepts(text):
    """
    Extract concepts using RAKE
    """
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases_with_scores()

def compare_extractions():
    """
    Compare RAKE and LLM extractions for randomly selected papers from each domain
    """
    print(f"Extraction Comparison Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run ID: {run_id}")
    print(f"Log file: {log_file}")
    print("\nThis analysis compares RAKE keyword extraction with LLM-based concept extraction")
    print("for randomly selected papers from three scientific domains.\n")

    # Statistics tracking
    domain_stats = {
        "astrophysics": {"papers": 0, "rake_concepts": 0, "llm_concepts": 0, "concept_types": Counter()},
        "fluid_dynamics": {"papers": 0, "rake_concepts": 0, "llm_concepts": 0, "concept_types": Counter()},
        "evolutionary_biology": {"papers": 0, "rake_concepts": 0, "llm_concepts": 0, "concept_types": Counter()}
    }
    
    total_papers_found = 0
    total_papers_analyzed = 0
    
    # Get random papers from each domain
    all_papers = {}
    for domain in domains:
        all_papers[domain] = get_random_papers(domain, args.n)
        total_papers_found += len(all_papers[domain])
    
    for domain, paper_ids in all_papers.items():
        print(f"\n{'='*80}\nAnalyzing papers from domain: {domain}\n{'='*80}")
        
        for paper_id in paper_ids:
            print(f"\n{'-'*60}\nPaper: {paper_id}\n{'-'*60}")
            
            # Get database paths for this domain
            json_db, readable_db = domains[domain]
            
            # Try both databases to find the paper with concepts
            paper_data, llm_concepts = get_paper_data_and_llm_concepts(paper_id, database_paths[json_db])
            format_used = "JSON"
            
            if not paper_data or not llm_concepts:
                paper_data, llm_concepts = get_paper_data_and_llm_concepts(paper_id, database_paths[readable_db])
                format_used = "readable"
                
            if not paper_data:
                print(f"Paper not found in any database: {paper_id}")
                continue

            total_papers_analyzed += 1
                
            full_paper_id, abstract, authors, primary_category, url, updated_on, sentence_count = paper_data
            
            # Get RAKE concepts
            rake_concepts = get_rake_concepts(abstract)
            
            # Count concept types from LLM extraction
            concept_types = Counter([tag_type for _, tag_type, _ in llm_concepts])

            # Update statistics
            domain_stats[domain]["papers"] += 1
            domain_stats[domain]["rake_concepts"] += len(rake_concepts)
            domain_stats[domain]["llm_concepts"] += len(llm_concepts)
            domain_stats[domain]["concept_types"].update(concept_types)
            
            # Print results
            print(f"Format used: {format_used}")
            print(f"Paper ID: {full_paper_id}")
            print(f"Authors: {authors}")
            print(f"Category: {primary_category}")
            print(f"URL: {url}")
            print(f"Updated on: {updated_on}")
            print(f"Sentence count: {sentence_count}")
            print(f"Abstract: {abstract}")
            
            print("\nRAKE keywords:")
            for score, keyword in rake_concepts:
                print(f"  {score:.2f}: {keyword}")
            
            print("\nLLM extracted concepts:")
            print(f"  Total concepts: {len(llm_concepts)}")
            print(f"  Concept types: {dict(concept_types)}")
            
            print("\nAll LLM concepts by type:")
            # Group concepts by type
            concepts_by_type = {}
            for _, tag_type, concept in llm_concepts:
                if tag_type not in concepts_by_type:
                    concepts_by_type[tag_type] = []
                concepts_by_type[tag_type].append(concept)
            
            # Print all concepts by type
            for tag_type, concepts in concepts_by_type.items():
                print(f"  {tag_type}:")
                for concept in concepts:
                    print(f"    - {concept}")
    
    # Print summary tables
    print("\n\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80 + "\n")
    
    print(f"Total papers found: {total_papers_found}")
    print(f"Total papers successfully analyzed: {total_papers_analyzed}")
    print()
    
    # Table 1: Average number of concepts by domain
    print("Table 1: Average Number of Extracted Concepts by Domain")
    print("-" * 80)
    print(f"{'Domain':<25} {'Papers':<10} {'RAKE (avg)':<15} {'LLM (avg)':<15} {'RAKE/LLM Ratio':<15}")
    print("-" * 80)
    
    total_rake = 0
    total_llm = 0
    total_papers = 0
    
    for domain, stats in domain_stats.items():
        if stats["papers"] > 0:
            rake_avg = stats["rake_concepts"] / stats["papers"]
            llm_avg = stats["llm_concepts"] / stats["papers"]
            ratio = rake_avg / llm_avg if llm_avg > 0 else "N/A"
            ratio_str = f"{ratio:.2f}" if isinstance(ratio, float) else ratio
            print(f"{domain:<25} {stats['papers']:<10} {rake_avg:<15.1f} {llm_avg:<15.1f} {ratio_str:<15}")
            
            total_rake += stats["rake_concepts"]
            total_llm += stats["llm_concepts"]
            total_papers += stats["papers"]

    # Add overall averages
    if total_papers > 0:
        overall_rake_avg = total_rake / total_papers
        overall_llm_avg = total_llm / total_papers
        overall_ratio = overall_rake_avg / overall_llm_avg if overall_llm_avg > 0 else "N/A"
        overall_ratio_str = f"{overall_ratio:.2f}" if isinstance(overall_ratio, float) else overall_ratio
        
        print("-" * 80)
        print(f"{'OVERALL':<25} {total_papers:<10} {overall_rake_avg:<15.1f} {overall_llm_avg:<15.1f} {overall_ratio_str:<15}")
    
    print("-" * 80 + "\n")
    
    # Table 2: Distribution of LLM concept types by domain
    print("Table 2: Distribution of LLM Concept Types by Domain")
    print("-" * 100)
    print(f"{'Domain':<25} {'LLM Concept Types':<75}")
    print("-" * 100)
    
    # Collect all concept types across domains
    all_concept_types = set()
    for stats in domain_stats.values():
        all_concept_types.update(stats["concept_types"].keys())
    
    # Create a combined counter for all domains
    all_domains_counter = Counter()
    
    for domain, stats in domain_stats.items():
        if stats["papers"] > 0 and stats["llm_concepts"] > 0:
            # Calculate percentages
            concept_percentages = {}
            for concept_type, count in stats["concept_types"].items():
                percentage = (count / stats["llm_concepts"]) * 100
                concept_percentages[concept_type] = percentage
                all_domains_counter[concept_type] += count
            
            # Sort by percentage (descending)
            sorted_concepts = sorted(concept_percentages.items(), key=lambda x: x[1], reverse=True)

            # Format as string
            concept_str = ", ".join([f"{ctype} ({perc:.1f}%)" for ctype, perc in sorted_concepts])
            
            print(f"{domain:<25} {concept_str}")
    
    # Add overall distribution
    if sum(all_domains_counter.values()) > 0:
        print("-" * 100)
        overall_percentages = {}
        total_concepts = sum(all_domains_counter.values())
        
        for concept_type, count in all_domains_counter.items():
            overall_percentages[concept_type] = (count / total_concepts) * 100
        
        sorted_overall = sorted(overall_percentages.items(), key=lambda x: x[1], reverse=True)
        overall_str = ", ".join([f"{ctype} ({perc:.1f}%)" for ctype, perc in sorted_overall])
        
        print(f"{'OVERALL':<25} {overall_str}")
    
    print("-" * 100)
    
    # Table 3: Detailed concept type distribution
    if all_concept_types and sum(all_domains_counter.values()) > 0:
        print("\nTable 3: Detailed Concept Type Distribution by Domain (%)")
        
        # Create header
        header = f"{'Domain':<25}"
        for concept_type in sorted(all_concept_types):
            header += f"{concept_type:<15}"
        print("-" * (25 + 15 * len(all_concept_types)))
        print(header)
        print("-" * (25 + 15 * len(all_concept_types)))
        
        # Print percentages for each domain
        for domain, stats in domain_stats.items():
            if stats["papers"] > 0 and stats["llm_concepts"] > 0:
                row = f"{domain:<25}"
                for concept_type in sorted(all_concept_types):
                    count = stats["concept_types"].get(concept_type, 0)
                    percentage = (count / stats["llm_concepts"]) * 100 if stats["llm_concepts"] > 0 else 0
                    row += f"{percentage:<15.1f}"
                print(row)
        
        # Print overall percentages
        print("-" * (25 + 15 * len(all_concept_types)))
        overall_row = f"{'OVERALL':<25}"
        total_concepts = sum(all_domains_counter.values())
        for concept_type in sorted(all_concept_types):
            count = all_domains_counter.get(concept_type, 0)
            percentage = (count / total_concepts) * 100 if total_concepts > 0 else 0
            overall_row += f"{percentage:<15.1f}"
        print(overall_row)
        print("-" * (25 + 15 * len(all_concept_types)))

     # Add warning if no LLM concepts were found
    if all(stats["llm_concepts"] == 0 for stats in domain_stats.values()):
        print("\nWARNING: No LLM concepts were found for any papers!")
        print("This could be due to:")
        print("1. The database schema doesn't match what the script expects")
        print("2. The predictions (LLM concepts) are stored in a different format or location")
        print("3. The database doesn't contain any predictions")
        print("\nPlease check the database structure and update the script accordingly.")
    

    print(f"\n\nAnalysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to {log_file}")
    
    # Print to the actual console where the log file is saved
    original_stdout = sys.__stdout__
    original_stdout.write(f"Extraction comparison results saved to: {log_file}\n")

if __name__ == "__main__":
    try:
        compare_extractions()
    finally:
        # Make sure to close the log file
        if log_handle:
            sys.stdout = sys.__stdout__  # Restore original stdout
            log_handle.close()
            print(f"Log file closed: {log_file}")

