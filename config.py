DEFAULT_MODEL_ID = "Meta-Llama-3-70B-Instruct"
DEFAULT_INTERFACE_MODEL_ID = "NumbersStation/nsql-llama-2-7B"
DEFAULT_KIND = "json"
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_FEW_SHOT_NUM = 3
DEFAULT_FEW_SHOT_SELECTION = "random"
DEFAULT_SAVE_INTERVAL = 3
DEFAULT_RES_DIR = "results"
DEFAULT_LOG_DIR = "logs"
DEFAULT_TABLES_DIR = "tables"
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
        substr(papers.updated_on, 2, 7) as year_month, 
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
        p.url,
        pred.concept AS dataset,
        p.updated_on,
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
        url,
        dataset,
        updated_on
    FROM 
        ranked_datasets
    WHERE 
        rn = 1
    ORDER BY 
        updated_on ASC    
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
        UNION SELECT 'spatio-temporal'
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
        MAX(CASE WHEN LOWER(pred.concept) LIKE '%imag%' THEN 1 ELSE 0 END) AS uses_galaxy_images,
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
        "What are all the next highest data modalities after images and spectra?",
        """
        SELECT 
            LOWER(concept) AS modality,
            COUNT(DISTINCT paper_id) AS usage_count
        FROM 
            predictions
        WHERE 
            tag_type = 'modality'
            AND LOWER(concept) NOT LIKE '%imag%'
            AND LOWER(concept) NOT LIKE '%spectr%'
        GROUP BY 
            LOWER(concept)
        ORDER BY 
            usage_count DESC
        """,
    ),
    (
        "If we include the next biggest data modality, how much does coverage change?",
        """
    WITH modality_counts AS (
    SELECT 
        LOWER(concept) AS modality,
        COUNT(DISTINCT paper_id) AS usage_count
    FROM 
        predictions
    WHERE 
        tag_type = 'modality'
        AND LOWER(concept) NOT LIKE '%imag%'
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
            MAX(CASE WHEN LOWER(pred.concept) LIKE '%imag%' THEN 1 ELSE 0 END) AS uses_galaxy_images,
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
        "Coverage if we select the next 5 highest modalities?",
        """
    WITH ranked_modalities AS (
    SELECT 
        LOWER(concept) AS modality,
        COUNT(DISTINCT paper_id) AS usage_count,
        ROW_NUMBER() OVER (ORDER BY COUNT(DISTINCT paper_id) DESC) AS rank
    FROM 
        predictions
    WHERE 
        tag_type = 'modality'
        AND LOWER(concept) NOT LIKE '%imag%'
        AND LOWER(concept) NOT LIKE '%spectr%'
    GROUP BY 
        LOWER(concept)
    ),
    paper_modalities AS (
        SELECT 
            p.paper_id,
            MAX(CASE WHEN LOWER(pred.concept) LIKE '%imag%' THEN 1 ELSE 0 END) AS uses_images,
            MAX(CASE WHEN LOWER(pred.concept) LIKE '%spectr%' THEN 1 ELSE 0 END) AS uses_spectra,
            MAX(CASE WHEN rm.rank = 1 THEN 1 ELSE 0 END) AS uses_modality_1,
            MAX(CASE WHEN rm.rank = 2 THEN 1 ELSE 0 END) AS uses_modality_2,
            MAX(CASE WHEN rm.rank = 3 THEN 1 ELSE 0 END) AS uses_modality_3,
            MAX(CASE WHEN rm.rank = 4 THEN 1 ELSE 0 END) AS uses_modality_4,
            MAX(CASE WHEN rm.rank = 5 THEN 1 ELSE 0 END) AS uses_modality_5
        FROM 
            papers p
        LEFT JOIN 
            predictions pred ON p.paper_id = pred.paper_id
        LEFT JOIN
            ranked_modalities rm ON LOWER(pred.concept) = rm.modality
        WHERE 
            p.primary_category LIKE '%astro-ph%'
            AND pred.tag_type = 'modality'
        GROUP BY 
            p.paper_id
    ),
    cumulative_coverage AS (
        SELECT
            'Images and Spectra' AS modalities,
            0 AS added_modality_rank,
            SUM(CASE WHEN uses_images = 1 OR uses_spectra = 1 THEN 1 ELSE 0 END) AS covered_papers,
            COUNT(*) AS total_papers
        FROM 
            paper_modalities
        
        UNION ALL
        
        SELECT
            'Images, Spectra, and Modality 1' AS modalities,
            1 AS added_modality_rank,
            SUM(CASE WHEN uses_images = 1 OR uses_spectra = 1 OR uses_modality_1 = 1 THEN 1 ELSE 0 END) AS covered_papers,
            COUNT(*) AS total_papers
        FROM 
            paper_modalities
        
        UNION ALL
        
        SELECT
            'Images, Spectra, Modality 1, and 2' AS modalities,
            2 AS added_modality_rank,
            SUM(CASE WHEN uses_images = 1 OR uses_spectra = 1 OR uses_modality_1 = 1 OR uses_modality_2 = 1 THEN 1 ELSE 0 END) AS covered_papers,
            COUNT(*) AS total_papers
        FROM 
            paper_modalities
        
        UNION ALL
        
        SELECT
            'Images, Spectra, Modality 1, 2, and 3' AS modalities,
            3 AS added_modality_rank,
            SUM(CASE WHEN uses_images = 1 OR uses_spectra = 1 OR uses_modality_1 = 1 OR uses_modality_2 = 1 OR uses_modality_3 = 1 THEN 1 ELSE 0 END) AS covered_papers,
            COUNT(*) AS total_papers
        FROM 
            paper_modalities
        
        UNION ALL
        
        SELECT
            'Images, Spectra, Modality 1, 2, 3, and 4' AS modalities,
            4 AS added_modality_rank,
            SUM(CASE WHEN uses_images = 1 OR uses_spectra = 1 OR uses_modality_1 = 1 OR uses_modality_2 = 1 OR uses_modality_3 = 1 OR uses_modality_4 = 1 THEN 1 ELSE 0 END) AS covered_papers,
            COUNT(*) AS total_papers
        FROM 
            paper_modalities
        
        UNION ALL
        
        SELECT
            'Images, Spectra, Modality 1, 2, 3, 4, and 5' AS modalities,
            5 AS added_modality_rank,
            SUM(CASE WHEN uses_images = 1 OR uses_spectra = 1 OR uses_modality_1 = 1 OR uses_modality_2 = 1 OR uses_modality_3 = 1 OR uses_modality_4 = 1 OR uses_modality_5 = 1 THEN 1 ELSE 0 END) AS covered_papers,
            COUNT(*) AS total_papers
        FROM 
            paper_modalities
        )
        SELECT
            cc.modalities,
            COALESCE(rm.modality, 'N/A') AS added_modality,
            rm.usage_count AS added_modality_usage,
            ROUND(CAST(cc.covered_papers AS FLOAT) / cc.total_papers * 100, 2) AS coverage_percent,
            ROUND(CAST(cc.covered_papers AS FLOAT) / cc.total_papers * 100, 2) - 
            LAG(ROUND(CAST(cc.covered_papers AS FLOAT) / cc.total_papers * 100, 2), 1, 0) OVER (ORDER BY cc.added_modality_rank) AS coverage_increase_percent
        FROM
            cumulative_coverage cc
        LEFT JOIN
            ranked_modalities rm ON cc.added_modality_rank = rm.rank
        ORDER BY
            cc.added_modality_rank
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
