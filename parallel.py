"""
parallel.py
------------------
Generic parallel batch runner backed by ProcessPoolExecutor.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Tuple, Any

from tqdm import tqdm


def run_parallel(
    work_items: List[Any],
    worker_fn: Callable,
    n_workers: int = None,
    desc: str = "Processing",
) -> Tuple[List[Tuple[int, Any]], List[Any]]:
    """
    Run ``worker_fn`` on each item in ``work_items`` using multiple processes.

    Args:
        work_items: List of argument tuples, one per unit of work (e.g. one image).
        worker_fn:  Top-level, picklable callable with signature
                    ``worker_fn(item) -> (idx, result | None)``.
                    Return ``None`` as the result to signal a skipped/failed item.
        n_workers:  Number of parallel workers.  ``None`` defaults to
                    ``os.cpu_count()``.
        desc:       Label shown in the tqdm progress bar.

    Returns:
        results:  List of ``(idx, result)`` for every successfully processed item.
        failed:   List of items that raised an exception or returned ``None``.
    """
    if n_workers is None:
        n_workers = os.cpu_count()

    results: List[Tuple[int, Any]] = []
    failed: List[Any] = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        future_to_item = {pool.submit(worker_fn, item): item for item in work_items}
        for future in tqdm(
            as_completed(future_to_item),
            total=len(future_to_item),
            desc=desc,
        ):
            item = future_to_item[future]
            try:
                idx, result = future.result()
                if result is None:
                    failed.append(item)
                else:
                    results.append((idx, result))
            except Exception as exc:
                tqdm.write(f"  [FAIL] {item}: {exc}")
                failed.append(item)

    return results, failed
