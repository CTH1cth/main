#!/usr/bin/env python3
"""Download a large HTTP resource concurrently with validated byte ranges."""

from __future__ import annotations

import argparse
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(args.url, method="HEAD")) as response:
        total = int(response.headers["Content-Length"])
        if "bytes" not in response.headers.get("Accept-Ranges", "").lower():
            raise RuntimeError("Server does not advertise HTTP byte-range support")

    workers = max(1, int(args.workers))
    part_dir = output.parent / f".{output.name}.parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    step = (total + workers - 1) // workers
    jobs = []
    for index in range(workers):
        start = index * step
        end = min(total - 1, (index + 1) * step - 1)
        if start <= end:
            jobs.append((index, start, end, part_dir / f"part-{index:03d}"))

    def download(job: tuple[int, int, int, Path]) -> tuple[int, int]:
        index, start, end, path = job
        expected = end - start + 1
        if path.exists() and path.stat().st_size == expected:
            return index, expected
        if path.exists() and path.stat().st_size > expected:
            path.unlink()
        for attempt in range(1, 9):
            current = path.stat().st_size if path.exists() else 0
            if current == expected:
                break
            request_start = start + current
            request = urllib.request.Request(
                args.url,
                headers={"Range": f"bytes={request_start}-{end}"},
            )
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status != 206:
                        raise RuntimeError(
                            f"Range {index} returned HTTP {response.status}, expected 206"
                        )
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {request_start}-{end}/"):
                        raise RuntimeError(
                            f"Unexpected Content-Range for part {index}: {content_range}"
                        )
                    with path.open("ab") as stream:
                        shutil.copyfileobj(response, stream, length=1024 * 1024)
            except Exception as error:
                current = path.stat().st_size if path.exists() else 0
                print(
                    f"part {index + 1} retry {attempt}/8 at {current}/{expected}: {error}",
                    flush=True,
                )
                time.sleep(min(2 * attempt, 10))
        actual = path.stat().st_size
        if actual != expected:
            raise RuntimeError(f"Part {index} has {actual} bytes, expected {expected}")
        return index, actual

    print(f"downloading {total} bytes in {len(jobs)} ranges", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download, job) for job in jobs]
        for future in as_completed(futures):
            index, size = future.result()
            print(f"part {index + 1}/{len(jobs)} complete: {size} bytes", flush=True)

    temporary = output.with_suffix(output.suffix + ".assembling")
    with temporary.open("wb") as destination:
        for _, _, _, path in jobs:
            with path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    if temporary.stat().st_size != total:
        raise RuntimeError(
            f"Assembled file has {temporary.stat().st_size} bytes, expected {total}"
        )
    temporary.replace(output)
    shutil.rmtree(part_dir)
    print(f"saved={output} bytes={output.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
