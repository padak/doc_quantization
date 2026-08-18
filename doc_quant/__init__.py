"""doc_quant: decontextualization pipeline for Markdown anonymization.

Documents are tokenized and split into fixed-size token chunks stored in a
key-value database under random IDs. Chunks are sent (shuffled, without any
ordering metadata) to the Anthropic Batch API for name detection, and the
documents are reassembled in redacted form.
"""
