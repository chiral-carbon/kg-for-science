#!/bin/bash

# List of all astrophysics and physics subcategories
categories=(
    # "astro-ph.CO" "astro-ph.EP" "astro-ph.GA" "astro-ph.HE" "astro-ph.IM" "astro-ph.SR"
    "physics.acc-ph" "physics.app-ph" "physics.ao-ph" "physics.atm-clus" "physics.atom-ph"
    "physics.bio-ph" "physics.chem-ph" "physics.class-ph" "physics.comp-ph" "physics.data-an"
    "physics.flu-dyn" "physics.gen-ph" "physics.geo-ph" "physics.hist-ph" "physics.ins-det"
    "physics.med-ph" "physics.optics" "physics.ed-ph" "physics.soc-ph" "physics.plasm-ph"
    "physics.pop-ph" "physics.space-ph"
)

# Set the number of papers to collect per category
papers_per_category=200

# Set the date range for the last 10 years
end_date=$(date +%Y%m%d)
start_date=$(date -d "10 years ago" +%Y%m%d)

# Create a temporary directory to store individual category results
temp_dir=$(mktemp -d)

# Loop through each category and run the Python script
for category in "${categories[@]}"; do
    echo "Collecting papers for category: $category"
    search_query="cat:$category AND submittedDate:[$start_date TO $end_date]"
    python scripts/collect_data.py --search_query "$search_query" --max_results $papers_per_category --sort_by submitted_date --sort_order desc --out_file "${temp_dir}/${category}.jsonl"
done

# Combine all individual files into one
output_file="stratified_arxiv_data_$(date +%Y%m%d_%H%M%S).jsonl"
cat "${temp_dir}"/*.jsonl > "$output_file"

# Clean up temporary files
rm -rf "$temp_dir"

echo "Data collection complete. Results saved in $output_file"

# Optional: Shuffle the final file to ensure randomness
shuf "$output_file" -o "${output_file%.jsonl}_shuffled.jsonl"

echo "Shuffled results saved in ${output_file%.jsonl}_shuffled.jsonl"