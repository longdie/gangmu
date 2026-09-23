"""Validated, resumable embedding caches for the released benchmark.

The cache key binds ordered IDs and exact stripped inputs to the model and
preprocessing. Raw and normalized statements must never share a cache key.
"""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from filelock import FileLock

from .config import GangmuConfig, get_config, resolve_project_path
from .paths import ROOT_DIR


def validate_vectors(matrix: np.ndarray, rows: int, dimensions: int = 2048) -> None:
    if matrix.shape != (rows, dimensions) or matrix.dtype != np.float32:
        raise ValueError(f"Embedding matrix must have shape {(rows, dimensions)} and dtype float32.")
    for start in range(0, rows, 2048):
        chunk = matrix[start:start + 2048]
        if not np.isfinite(chunk).all() or not np.allclose(np.linalg.norm(chunk, axis=1), 1, atol=1e-4, rtol=0):
            raise ValueError("Embedding vectors must be finite and L2-normalized.")


def ordered_text_hash(ids: Sequence[int], texts: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(zip(ids, texts)), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class EmbeddingCache:
    def __init__(self, *, config: GangmuConfig | None = None, directory: str | Path | None = None,
                 embed: Callable[[str], np.ndarray] | None = None):
        self.config = config or get_config()
        self.directory = resolve_project_path(directory) if directory else ROOT_DIR / 'outputs' / 'embedding_cache'
        self._embed = embed

    def matrix(self, name: str, ids: Sequence[int], texts: Sequence[str], *, supplied: Path | None = None,
               supplied_model: str | None = None, supplied_text_hash: str | None = None,
               supplied_sha256: str | None = None) -> np.ndarray:
        ids = [int(i) for i in ids]
        texts = [str(t).strip() for t in texts]
        if len(ids) != len(texts) or len(set(ids)) != len(ids) or any(not t for t in texts):
            raise ValueError('Embedding IDs/texts must be aligned, unique, and nonempty.')
        model = self.config.model('embedding')
        dimensions = int(model.dimensions or 2048)
        text_hash = ordered_text_hash(ids, texts)
        # The release supplies normalized query vectors. An altered text/model
        # must regenerate them together with the knowledge-base vectors.
        if supplied is not None and supplied.exists() and supplied_model == model.name and supplied_text_hash == text_hash:
            if supplied_sha256 is not None and file_hash(supplied) != supplied_sha256:
                raise ValueError('Supplied embedding file hash does not match the frozen release.')
            matrix = np.load(supplied, mmap_mode='r', allow_pickle=False)
            validate_vectors(matrix, len(ids), dimensions)
            return matrix
        identity = {'format': 1, 'name': name, 'model': model.name, 'dimensions': dimensions,
                    'dtype': 'float32', 'l2_normalized': True, 'preprocessing': 'strip',
                    'ordered_ids_and_texts_sha256': text_hash, 'rows': len(ids)}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        folder = self.directory / name / key
        folder.mkdir(parents=True, exist_ok=True)
        final = folder / 'embeddings.npy'
        with FileLock(str(folder / '.lock')):
            if final.exists():
                metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
                if metadata.get('identity') != identity or metadata.get('sha256') != file_hash(final):
                    raise ValueError('Embedding cache identity/checksum mismatch; remove the damaged cache and retry.')
                result = np.load(final, mmap_mode='r', allow_pickle=False)
                validate_vectors(result, len(ids), dimensions)
                return result
            partial = folder / 'partial.npy'
            done_path = folder / 'completed.npy'
            if partial.exists() and done_path.exists():
                matrix = np.load(partial, mmap_mode='r+', allow_pickle=False)
                done = np.load(done_path, allow_pickle=False)
                if matrix.shape != (len(ids), dimensions) or matrix.dtype != np.float32 or done.shape != (len(ids),) or done.dtype != bool:
                    raise ValueError('Embedding checkpoint shape/dtype is invalid.')
                validate_vectors(matrix[done], int(done.sum()), dimensions)
            else:
                matrix = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32, shape=(len(ids), dimensions))
                done = np.zeros(len(ids), dtype=bool)
                np.save(done_path, done, allow_pickle=False)
            missing = np.flatnonzero(~done).tolist()
            embed = self._embed
            if missing and embed is None:
                from src.cross_doc.clients import CanonicalStatementEmbedder
                embed = CanonicalStatementEmbedder(config=self.config).embed
            workers = self.config.get_int('embedding.parallel_workers', 5, minimum=1)
            completed_before = int(done.sum())
            if missing:
                print(f'Embedding {name}: {len(missing)} missing / {len(ids)} rows', flush=True)
            for start in range(0, len(missing), max(32, workers)):
                errors = []
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    tasks = {pool.submit(embed, texts[i]): i for i in missing[start:start + max(32, workers)]}
                    for future in as_completed(tasks):
                        i = tasks[future]
                        try:
                            vector = np.asarray(future.result(), dtype=np.float32)
                            if vector.shape != (dimensions,) or not np.isfinite(vector).all():
                                raise ValueError('Embedding API returned invalid dimensions/values.')
                            norm = float(np.linalg.norm(vector))
                            if norm <= 0:
                                raise ValueError('Embedding API returned a zero vector.')
                            matrix[i] = vector / norm
                            done[i] = True
                        except Exception as exc:
                            errors.append((i, type(exc).__name__))
                matrix.flush()
                temp_done = folder / 'completed.tmp.npy'
                np.save(temp_done, done, allow_pickle=False)
                os.replace(temp_done, done_path)
                if errors:
                    # Do not include potentially sensitive HTTP response bodies.
                    raise RuntimeError(f'Embedding failed for {len(errors)} rows ({errors[0][1]}); completed rows are saved.')
                if int(done.sum()) // 256 > (completed_before // 256):
                    print(f'Embedding {name}: {int(done.sum())}/{len(ids)}', flush=True)
                    completed_before = int(done.sum())
            validate_vectors(matrix, len(ids), dimensions)
            del matrix
            (folder / 'metadata.json').write_text(json.dumps({'identity': identity, 'sha256':file_hash(partial)}, indent=2) + '\n', encoding='utf-8')
            os.replace(partial, final)
            done_path.unlink(missing_ok=True)
            return np.load(final, mmap_mode='r', allow_pickle=False)
