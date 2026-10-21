# -*- coding: utf-8 -*-
"""multiprocessing 辅助函数。

背景：在共享机器上，POSIX 信号量的名字可能与其他用户的信号量撞车。
CPython 的 ``SemLock`` 用 ``sem_open(name, O_CREAT, ...)`` 建信号量，随后立刻
``sem_unlink(name)``；如果该名字恰好对应**别的用户**已经持有的信号量，
``sem_open`` 会把别人的打开，而 ``sem_unlink`` 因权限不足返回 EACCES，
最终表现为 ``PermissionError: [Errno 13] Permission denied``。

2026-09-23 实测：6 个 binning run 同时各自创建 ``spawn`` 上下文的 ``Pool(48)``
（合计约 290 个 worker 进程）时，6 个 run 在同一时刻全部因此崩溃在
``Cluster/first_cluster.py`` 的建池处。这类冲突是瞬时的，重试几次即可成功。

因此这里提供 ``make_spawn_pool``：建池失败（PermissionError / OSError）时
等待若干秒后重试，超过次数才抛出。
"""

import time

from CompleteBin.logger import get_logger

logger = get_logger()


def make_spawn_pool(ctx, num_workers, attempts=5, delay=5.0, label=""):
    """用给定上下文创建 Pool，遇到信号量类错误时重试。

    Args:
        ctx: ``multiprocessing.get_context("spawn")`` 之类的上下文。
        num_workers: 进程数。
        attempts: 最多尝试次数。
        delay: 每次失败后等待的秒数。
        label: 仅用于日志，标明调用者。

    Returns:
        Pool 对象（调用方负责关闭，推荐 ``with`` 使用）。
    """
    tag = f" [{label}]" if label else ""
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            pool = ctx.Pool(num_workers)
            if attempt > 1:
                logger.info(f"--> Pool({num_workers}) created on attempt {attempt}{tag}.")
            return pool
        except (PermissionError, OSError) as exc:
            last_exc = exc
            logger.warning(
                f"--> Pool({num_workers}) creation failed "
                f"({type(exc).__name__}: {exc}); attempt {attempt}/{attempts}{tag}."
            )
            if attempt < attempts:
                time.sleep(delay)
    logger.error(f"--> Pool({num_workers}) could not be created after {attempts} attempts{tag}.")
    raise last_exc
