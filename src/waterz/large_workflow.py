"""Workflow planning helpers for extra-large WaterZ decoding."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil
from typing import Sequence

from .orchestrator import TaskSpec

__all__ = [
    "BorderRef",
    "ChunkRef",
    "build_border_adjacency",
    "build_chunk_grid",
    "build_large_decode_tasks",
]


@dataclass(frozen=True)
class ChunkRef:
    """One logical chunk in ZYX order."""

    index: tuple[int, int, int]
    start: tuple[int, int, int]
    stop: tuple[int, int, int]

    @property
    def key(self) -> str:
        z, y, x = self.index
        return f"z{z}_y{y}_x{x}"

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.stop[i] - self.start[i] for i in range(3))


@dataclass(frozen=True)
class BorderRef:
    """One face adjacency between two chunks."""

    axis: str
    src: ChunkRef
    dst: ChunkRef

    @property
    def key(self) -> str:
        return f"{self.axis}:{self.src.key}->{self.dst.key}"


def build_chunk_grid(
    volume_shape: Sequence[int],
    chunk_shape: Sequence[int],
) -> list[ChunkRef]:
    """Split a volume into chunk-aligned boxes in ZYX order."""
    if len(volume_shape) != 3 or len(chunk_shape) != 3:
        raise ValueError("volume_shape and chunk_shape must both be length-3 ZYX tuples.")

    z_chunks = ceil(int(volume_shape[0]) / int(chunk_shape[0]))
    y_chunks = ceil(int(volume_shape[1]) / int(chunk_shape[1]))
    x_chunks = ceil(int(volume_shape[2]) / int(chunk_shape[2]))

    chunks: list[ChunkRef] = []
    for index in product(range(z_chunks), range(y_chunks), range(x_chunks)):
        start = tuple(index[i] * int(chunk_shape[i]) for i in range(3))
        stop = tuple(min(start[i] + int(chunk_shape[i]), int(volume_shape[i])) for i in range(3))
        chunks.append(ChunkRef(index=index, start=start, stop=stop))
    return chunks


def build_border_adjacency(chunks: Sequence[ChunkRef]) -> list[BorderRef]:
    """Enumerate +Z, +Y, +X face adjacencies for the chunk grid."""
    by_index = {chunk.index: chunk for chunk in chunks}
    edges: list[BorderRef] = []
    for chunk in chunks:
        z, y, x = chunk.index
        for axis, neighbor_index in (
            ("z", (z + 1, y, x)),
            ("y", (z, y + 1, x)),
            ("x", (z, y, x + 1)),
        ):
            neighbor = by_index.get(neighbor_index)
            if neighbor is not None:
                edges.append(BorderRef(axis=axis, src=chunk, dst=neighbor))
    return edges


def build_large_decode_tasks(
    chunks: Sequence[ChunkRef],
    borders: Sequence[BorderRef] | None = None,
    *,
    write_output: bool = False,
    output_format: str = "h5",
) -> list[TaskSpec]:
    """Build the staged task graph for chunked WaterZ decoding."""
    borders = list(build_border_adjacency(chunks) if borders is None else borders)

    tasks: list[TaskSpec] = []
    decode_ids: list[str] = []
    for chunk in chunks:
        spec = TaskSpec(
            name="decode_chunk",
            stage="decode",
            key=chunk.key,
            payload={
                "chunk_index": list(chunk.index),
                "chunk_start": list(chunk.start),
                "chunk_stop": list(chunk.stop),
            },
        )
        tasks.append(spec)
        decode_ids.append(spec.task_id)

    offsets_spec = TaskSpec(
        name="compute_offsets",
        stage="offsets",
        key="global",
        deps=tuple(decode_ids),
    )
    tasks.append(offsets_spec)

    connect_ids: list[str] = []
    for border in borders:
        spec = TaskSpec(
            name="connect_border",
            stage="connect",
            key=border.key,
            deps=(offsets_spec.task_id, f"decode:{border.src.key}", f"decode:{border.dst.key}"),
            payload={
                "axis": border.axis,
                "src_chunk": border.src.key,
                "dst_chunk": border.dst.key,
            },
        )
        tasks.append(spec)
        connect_ids.append(spec.task_id)

    relabel_spec = TaskSpec(
        name="reduce_relabel",
        stage="relabel",
        key="global",
        deps=tuple(connect_ids) if connect_ids else (offsets_spec.task_id,),
    )
    tasks.append(relabel_spec)

    apply_ids: list[str] = []
    for chunk in chunks:
        spec = TaskSpec(
            name="apply_relabel",
            stage="apply",
            key=chunk.key,
            deps=(f"decode:{chunk.key}", relabel_spec.task_id),
            payload={"chunk_key": chunk.key},
        )
        tasks.append(spec)
        apply_ids.append(spec.task_id)

    if write_output:
        tasks.append(
            TaskSpec(
                name="assemble_output",
                stage="assemble",
                key=output_format,
                deps=tuple(apply_ids),
                payload={"format": output_format},
            )
        )

    return tasks
