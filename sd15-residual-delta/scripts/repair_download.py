#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import requests


def remote_range(url: str, start: int, count: int) -> bytes:
    response = requests.get(
        url,
        headers={"Range": f"bytes={start}-{start + count - 1}"},
        timeout=60,
    )
    response.raise_for_status()
    if response.status_code != 206 or len(response.content) != count:
        raise RuntimeError(
            f"Range request failed: status={response.status_code}, bytes={len(response.content)}"
        )
    return response.content


def local_range(path: Path, start: int, count: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(start)
        return handle.read(count)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--expected-size", required=True, type=int)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.file)
    output = Path(args.output)
    extra = source.stat().st_size - args.expected_size
    if extra <= 0:
        raise RuntimeError(f"Source does not contain extra bytes: delta={extra}")
    probe = 4096
    if local_range(source, 0, probe) != remote_range(args.url, 0, probe):
        raise RuntimeError("File differs from official data at byte zero")
    segments: list[tuple[int, int]] = []
    cumulative_shift = 0
    search_start = 0
    while cumulative_shift < extra:
        low, high = search_start, args.expected_size - probe
        if local_range(source, high + cumulative_shift, probe) == remote_range(
            args.url, high, probe
        ):
            raise RuntimeError("Tail unexpectedly aligns before all extra bytes are removed")
        while high - low > probe:
            midpoint = ((low + high) // (2 * probe)) * probe
            if local_range(source, midpoint + cumulative_shift, probe) == remote_range(
                args.url, midpoint, probe
            ):
                low = midpoint
            else:
                high = midpoint
        window_start = max(search_start, low - probe)
        window_count = min(
            args.expected_size - window_start, high - window_start + probe * 2
        )
        local = local_range(source, window_start + cumulative_shift, window_count)
        remote = remote_range(args.url, window_start, window_count)
        difference = next(
            (
                index
                for index, (left, right) in enumerate(zip(local, remote))
                if left != right
            ),
            None,
        )
        if difference is None:
            raise RuntimeError("Could not locate the next differing byte")
        boundary = window_start + difference
        remaining_extra = extra - cumulative_shift
        verification_count = min(1024 * 1024, args.expected_size - boundary)
        official_tail = remote_range(args.url, boundary, verification_count)
        local_tail = local_range(
            source,
            boundary + cumulative_shift,
            verification_count + remaining_extra,
        )
        needle = official_tail[: min(64 * 1024, len(official_tail))]
        inserted = local_tail.find(needle, 1, remaining_extra + len(needle))
        if inserted <= 0 or inserted > remaining_extra:
            raise RuntimeError(
                f"Could not realign after mismatch at {boundary}; remaining extra={remaining_extra}"
            )
        if local_tail[inserted : inserted + verification_count] != official_tail:
            raise RuntimeError(f"Candidate realignment of {inserted} bytes did not verify")
        removal_start = boundary + cumulative_shift
        segments.append((removal_start, inserted))
        cumulative_shift += inserted
        search_start = boundary + probe
        print(
            f"segment={len(segments)} boundary={boundary} "
            f"source_offset={removal_start} removed={inserted} cumulative={cumulative_shift}"
        )
    tail_start = max(search_start, args.expected_size - 1024 * 1024)
    tail_count = args.expected_size - tail_start
    if local_range(source, tail_start + cumulative_shift, tail_count) != remote_range(
        args.url, tail_start, tail_count
    ):
        raise RuntimeError("Tail does not align after all detected removals")
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, output.open("wb") as writer:
        cursor = 0
        for removal_start, removal_length in segments:
            remaining = removal_start - cursor
            while remaining:
                chunk = reader.read(min(16 * 1024 * 1024, remaining))
                if not chunk:
                    raise EOFError("Unexpected EOF before corruption segment")
                writer.write(chunk)
                remaining -= len(chunk)
            reader.seek(removal_length, os.SEEK_CUR)
            cursor = removal_start + removal_length
        while chunk := reader.read(16 * 1024 * 1024):
            writer.write(chunk)
    actual_hash = sha256(output)
    print(f"segments={segments}")
    print(f"removed_bytes={sum(length for _, length in segments)}")
    print(f"output_size={output.stat().st_size}")
    print(f"sha256={actual_hash}")
    if output.stat().st_size != args.expected_size or actual_hash != args.expected_sha256:
        raise RuntimeError("Repaired file failed final size/hash validation")


if __name__ == "__main__":
    main()
