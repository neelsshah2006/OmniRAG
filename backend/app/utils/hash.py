import hashlib
from typing import BinaryIO


def sha256_stream(
    stream: BinaryIO,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Computes the SHA-256 hash of a binary stream.

    The stream position is restored before returning.
    """

    hasher = hashlib.sha256()

    stream.seek(0)

    while chunk := stream.read(chunk_size):
        hasher.update(chunk)

    stream.seek(0)

    return hasher.hexdigest()
