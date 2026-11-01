import math
import re
from collections import Counter
from dataclasses import dataclass

from .config import DATA_DIR

TOKEN = re.compile(r"[a-z0-9€%]+(?:[.,][0-9]+)?")
STOPWORDS = set(
    "a an the and or of to in on for is are be by with at from as it this that what which who how when "
    "do does can must may our we i you my your per any all within than".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    section: str
    text: str


def load_documents() -> dict[str, str]:
    return {p.stem: p.read_text() for p in sorted((DATA_DIR / "docs").glob("*.md"))}


def chunk_document(doc_id: str, text: str) -> list[Chunk]:
    lines = text.strip().splitlines()
    title = lines[0].lstrip("# ").strip()
    chunks, section, buf = [], "Overview", []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            chunks.append(Chunk(f"{doc_id}#{len(chunks)}", doc_id, title, section, body))

    for line in lines[1:]:
        if line.startswith("## "):
            flush()
            section, buf = line[3:].strip(), []
        else:
            buf.append(line)
    flush()
    return chunks


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.4, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        # Title and section are indexed with the body so "expense policy" finds the right doc
        self.tfs = [Counter(tokenize(f"{c.title} {c.section} {c.text}")) for c in chunks]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lengths) / len(self.lengths)
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, top_k: int = 4) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        scored = []
        for chunk, tf, dl in zip(self.chunks, self.tfs, self.lengths):
            score = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    norm = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                    score += self.idf[t] * norm
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


DOCUMENTS = load_documents()
INDEX = BM25Index([c for doc_id, text in DOCUMENTS.items() for c in chunk_document(doc_id, text)])
