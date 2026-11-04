"""Read-only frame snapshots and external OS counters, outside solver sources.

No trace/profile hooks, patched search functions, state copies, or added visited
sets. Native counters are sampled at 1 Hz; transient peaks can be missed.
"""
from __future__ import annotations

import ctypes
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import threading
import time

from planner_experiment.benchmark_suite.runtime import process_memory
from planner_experiment.benchmark_suite.worker import engine_worker


def cpu_seconds(pid):
    if os.name != 'nt':
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.GetProcessTimes.argtypes = [ctypes.c_void_p, *([ctypes.POINTER(ctypes.c_ulonglong)] * 4)]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        values = [ctypes.c_ulonglong() for _ in range(4)]
        if kernel.GetProcessTimes(handle, *(ctypes.byref(v) for v in values)):
            return (values[2].value + values[3].value) / 10_000_000
    finally:
        kernel.CloseHandle(handle)
    return None


def snapshot(main_ident):
    frame = sys._current_frames().get(main_ident)
    rows = []
    while frame is not None:
        filename = frame.f_code.co_filename.replace('\\', '/')
        name = frame.f_code.co_name
        local = frame.f_locals
        if filename.endswith('/planner_experiment/astar.py') and name in ('run', '_strategy_search', '_maximal_start_candidate'):
            row = {'engine': 'ASTAR', 'phase': name}
            for key in ('expanded', 'transitions', 'peak_open', 'reconstruction_count', 'weight', 'seed_transitions'):
                if isinstance(local.get(key), (int, float)):
                    row[key] = local[key]
            for key in ('nodes', 'labels', 'queue', 'closed', 'inconsistent'):
                if key in local:
                    row[key + '_count'] = len(local[key])
            rows.append(row)
        elif filename.endswith('/planner_experiment/ga.py') and name in ('run', '_decode', '_rollout', '_ranked_actions'):
            row = {'engine': 'GA', 'phase': name}
            for key in ('evaluations', 'generations', 'generation', 'transitions', 'cache_hits', 'branch_rollouts', 'used', 'phenotype_record_count'):
                if isinstance(local.get(key), (int, float)):
                    row[key] = local[key]
            for key in ('seen', 'frames', 'population', 'evaluated', 'ranked', 'scored'):
                if local.get(key) is not None:
                    row[key + '_count'] = len(local[key])
            state = local.get('state')
            if state is not None:
                row.update(completed=len(state.executions), running=len(state.running),
                           simulated_time=state.time, starts=sum(c for _, c in state.counts))
            rows.append(row)
        elif filename.endswith('/planner/partial_order.py') and name == 'plan':
            planner = local.get('self')
            if planner is not None:
                row = {'engine': 'LEGACY', 'phase': 'POP'}
                for key in ('instances', 'open_preconditions', 'causal_links', 'orderings'):
                    row[key + '_count'] = len(getattr(planner, key))
                row['native_stats'] = {k: v for k, v in vars(planner.stats).items() if isinstance(v, (int, float, str))}
                rows.append(row)
        elif filename.endswith('/services/planner_legacy_engine.py') and name == 'run_legacy_pipeline':
            rows.append({'engine': 'LEGACY', 'phase': 'pipeline',
                         'pop_available': 'pop' in local, 'scheduler_constructed': 'solver' in local})
        frame = frame.f_back
    return rows


def observed_worker(instance_path, engine, seconds, seed, solver_root, directory, pipe):
    stop = threading.Event()
    main_ident = threading.get_ident()
    stream = (Path(directory) / 'state-samples.jsonl').open('w', encoding='utf-8', buffering=1)
    started = None
    observer_cpu = 0.0
    thread = None

    def sample():
        nonlocal observer_cpu
        before = time.thread_time()
        try:
            data = {'elapsed_seconds': time.perf_counter() - started, 'frames': snapshot(main_ident)}
        except Exception as exc:
            data = {'elapsed_seconds': time.perf_counter() - started, 'sampling_error': repr(exc)}
        observer_cpu += time.thread_time() - before
        data['observer_cpu_seconds'] = observer_cpu
        stream.write(json.dumps(data, ensure_ascii=False) + '\n')

    def loop():
        while not stop.wait(1):
            sample()

    class Pipe:
        def send(self, message):
            nonlocal started, thread
            if message[0] == 'started':
                started = message[1]
                pipe.send(('cpu_start', time.process_time()))
                pipe.send(message)
                thread = threading.Thread(target=loop, daemon=True)
                thread.start()
            elif message[0] in ('done', 'error'):
                stop.set()
                if thread:
                    thread.join(timeout=1)
                pipe.send(('cpu_end', time.process_time()))
                pipe.send(message)
            else:
                pipe.send(message)
        def recv(self):
            return pipe.recv()
        def close(self):
            stop.set()
            if thread:
                thread.join(timeout=1)
            stream.close()
            pipe.close()
    engine_worker(instance_path, engine, seconds, seed, solver_root, directory, Pipe())


def supervise(target, arguments, directory, seconds, *, startup_timeout=30, hard_deadline=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    parent, child = mp.get_context('spawn').Pipe()
    process = mp.get_context('spawn').Process(target=target, args=(*arguments, str(directory), child))
    wall_start = time.perf_counter()
    process.start()
    child.close()
    started = ready = outcome = error = cpu_start = cpu_end = None
    peak, last, next_sample = 0, {}, 0
    samples = (directory / 'resources.jsonl').open('w', encoding='utf-8', buffering=1)
    try:
        while True:
            now = time.perf_counter()
            while True:
                try:
                    if not parent.poll():
                        break
                    kind, value = parent.recv()
                except (EOFError, OSError):
                    break
                if kind == 'ready':
                    ready = value
                    parent.send('go')
                elif kind == 'started':
                    started = value
                elif kind == 'done':
                    outcome = value
                elif kind == 'error':
                    error = value
                elif kind == 'cpu_start':
                    cpu_start = value
                elif kind == 'cpu_end':
                    cpu_end = value
            expired = (started is not None and now >= started + seconds)
            if now >= next_sample or expired or outcome is not None or error is not None:
                memory = process_memory(process.pid)
                cpu = cpu_seconds(process.pid)
                if memory:
                    last = {**memory, 'cpu_seconds_total': cpu}
                    peak = max(peak, memory['peak_working_set_bytes'])
                samples.write(json.dumps({'elapsed_seconds': None if started is None else now - started,
                                          **last, 'cpu_seconds_measured': None if cpu is None or cpu_start is None else max(0, cpu - cpu_start)}) + '\n')
                next_sample = now + 0.5
            if outcome is not None or error is not None:
                break
            if hard_deadline is not None and now >= hard_deadline:
                error = 'BATCH_HARD_DEADLINE'
                break
            if expired:
                error = 'TIME_LIMIT'
                break
            if started is None and now - wall_start >= startup_timeout:
                error = 'STARTUP_TIMEOUT'
                break
            if not process.is_alive():
                error = 'WORKER_EXITED_WITHOUT_RESULT'
                break
            time.sleep(0.01)
    finally:
        end_wall = time.perf_counter()
        cpu_end = cpu_end if cpu_end is not None else cpu_seconds(process.pid)
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        parent.close()
        samples.close()
    return {'process_status': 'COMPLETED' if outcome is not None else error,
            'startup_seconds': None if started is None else started - wall_start,
            'call_wall_seconds': end_wall - wall_start,
            'measured_elapsed_seconds': None if started is None else min(seconds, end_wall - started),
            'cpu_seconds_measured': None if cpu_start is None or cpu_end is None else max(0, cpu_end - cpu_start),
            'cpu_seconds_total': cpu_end, 'cpu_seconds_startup': cpu_start,
            'peak_working_set_bytes': peak or None, 'last_memory_sample': last,
            'ready': ready, 'outcome': outcome, 'error': error, 'exitcode': process.exitcode,
            'state_measurement': 'read-only 1 Hz native frame counters; sampled counts are lower bounds on peaks; no universal state count'}
