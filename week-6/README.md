# Use cases and applications of LLMs

[Week 6 Slides](Use%20cases%20and%20applications%20of%20LLMs%20-%20Lecture%206.pdf)

# Lab
1. Query tables in natural language. See [query_tables.py](query_tables.py)

   a. Improve code by loading all sections to avoid hard-coding a section with tables

   b. Test the capabilities of reasoning with table data: can it sum up numbers or do some other calculation?
3. Generate synthetic query with misspellings - only task description is available you need to write the code in [synthetic_data.py](synthetic_data.py)

   a. Find the queries in [web_search_queries.csv](web_search_queries.csv), containing example queries from different topics and use cases, like map search, job search, travel and tourism, general knowledge and learning

   b. Implement code, that will load one query at a time and generate up to N misspellings.

   c. Improve robustness of the method, for example, skip known abbreviations, like JFK.

   d. Make misspellings capture a variety of error types (phonetic, omission, transposition, repetition)?

   e. Test the resulting query variants with your favourite web search engine. Are results equal for the same for all variants of a given query? If not, why do you think this is happening?
