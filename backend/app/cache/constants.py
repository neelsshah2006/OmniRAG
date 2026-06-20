"""
Cache TTL configuration.

All values are in seconds.
"""

# Query embeddings rarely change
# Avoid recomputing same embeddings repeatedly
EMBEDDING_CACHE_TTL = 60 * 60 * 24
# 24 hours


# Retrieved chunks may change when documents update
RETRIEVAL_CACHE_TTL = 60 * 60
# 1 hour


# LLM answers depend on context,
# keep but invalidate aggressively later
LLM_RESPONSE_CACHE_TTL = 60 * 60 * 6
# 6 hours


# Active chat sessions
SESSION_CACHE_TTL = 60 * 60 * 2
# 2 hours


# HyDE generated documents
HYDE_CACHE_TTL = 60 * 60 * 24
# 24 hours
