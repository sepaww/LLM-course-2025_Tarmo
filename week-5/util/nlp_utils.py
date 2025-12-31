from stqdm import stqdm
import re

def sentencize(pages_and_texts: list[dict], nlp):
    for item in stqdm(pages_and_texts):
        item["sentences"] = list(nlp(item["text"]).sents)

        # Make sure all sentences are strings
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]

        # Count the sentences
        item["page_sentence_count_spacy"] = len(item["sentences"])


# chunking, i.e. grouping sentences into chunks of text
# Define split size to turn groups of sentences into chunks
num_sentence_chunk_size = 10

# Create a function that recursively splits a list into desired sizes
def split_list(input_list: list,
               slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into two lists of [[10], [7]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

# Perform sentence chunking
def chunk(pages_and_texts: list[dict]):
    # Loop through pages and texts and split sentences into chunks
    for item in stqdm(pages_and_texts):
        item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                             slice_size=num_sentence_chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])

# Convert chunks into text elements ready for embedding
def chunks_to_text_elems(pages_and_texts: list[dict]) -> list[dict]:
    pages_and_chunks = []
    for item in stqdm(pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]

            # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1',
                                           joined_sentence_chunk)  # ".A" -> ". A" for any full-stop/capital letter combo
            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # Get stats about the chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4  # 1 token = ~4 characters

            pages_and_chunks.append(chunk_dict)
    return pages_and_chunks