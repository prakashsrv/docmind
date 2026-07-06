from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    document_id: str
    chunk_index: int
    text: str
    embedding: list[float] | None = None
    metadata: dict = field(default_factory=dict)
