"""Observability web app over the `doc_quant` anonymization pipeline.

The package is a thin layer: chunking, synthetic mixing, detection parsing and
redaction all stay in `doc_quant`. What lives here is the HTTP surface that
makes each of those steps visible, plus the user-facing settings that override
`config/config.json` at runtime.
"""
