"""Frozen measurement/search loop, with only the run seed parameterized."""
from experiments.window_joint.__main__ import (
    isolate, write_json, os, now, time, threading, process_memory, Path,
    read_json, build, cp_model, math, append, Counter, rows_from, certificate, traceback)

def unlimited(payload, directory, *, cpu=4, seed=11):
    """No deadline. Shared frozen compilation/normalization/validation only."""
    isolate(cpu)
    directory.mkdir(parents=True, exist_ok=True)
    started, cpu_start = time.perf_counter(), time.process_time()
    write_json(directory / 'ready.json', {'pid': os.getpid(), 'utc': now(),
                                         'started_perf': started, 'cpu_start': cpu_start})
    lock, stopped = threading.RLock(), threading.Event()
    state = {'status': 'RUNNING', 'phase': 'BUILDING', 'pid': os.getpid(), 'started_at': now(),
             'time_limit_seconds': None, 'candidate_count': 0, 'best_makespan': None, 'lower_bound': None,
             'first_solution_seconds': None, 'first_optimal_seconds': None,
             'validation_seconds': 0.0, 'persistence_seconds': 0.0, 'rejections': [],
             'branches': None, 'conflicts': None, 'search_stats_sample_seconds': None}

    def save():
        with lock:
            state.update(updated_at=now(), elapsed_seconds=time.perf_counter()-started,
                         cpu_seconds=time.process_time()-cpu_start, memory=process_memory(os.getpid()))
            write_json(directory / 'state.json', state)

    def heartbeat():
        while not stopped.wait(5):
            save()

    save()
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        if isinstance(payload, Path):
            payload = read_json(payload)
        model, scenario, slots, stats = build(payload)
        built = time.perf_counter() - started
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = seed
        assert math.isinf(solver.parameters.max_time_in_seconds)
        assert math.isinf(solver.parameters.max_deterministic_time)
        write_json(directory / 'model.json', {'build_seconds': built, 'stats': stats,
                                              'parameters_text': str(solver.parameters),
                                              'wall_time_limit': 'infinity', 'deterministic_time_limit': 'infinity'})
        with lock:
            state.update(phase='SOLVING', build_seconds=built, model_stats=stats)
        save()

        def accept_bound(value):
            bound = max(0, math.ceil(value - scenario.start_time - 1e-7))
            with lock:
                if state['lower_bound'] is None or bound > state['lower_bound']:
                    state['lower_bound'] = bound
                    append(directory / 'bounds.jsonl', {'seconds': time.perf_counter()-started, 'lower_bound': bound})
                best = state['best_makespan']
                if best is not None and bound >= best:
                    if bound > best:
                        state['rejections'].append({'error': 'Bound exceeds replayed incumbent', 'bound': bound, 'best': best})
                    else:
                        state['first_optimal_seconds'] = state['first_optimal_seconds'] or time.perf_counter()-started
                    solver.stop_search()

        solver.best_bound_callback = accept_bound

        class Observe(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self):
                found = time.perf_counter() - started
                try:
                    raw = rows_from(self, slots)
                    cert = certificate(payload, raw)
                    validated = time.perf_counter() - started
                    bound = max(0, math.ceil(self.best_objective_bound - scenario.start_time - 1e-7))
                    counts = Counter(row['activity_id'] for row in cert['executions'])
                    record = {'discovered_seconds': found, 'validated_seconds': validated,
                              'raw_makespan': self.objective_value - scenario.start_time,
                              'makespan': cert['makespan'], 'lower_bound': bound,
                              'branches': self.num_branches, 'conflicts': self.num_conflicts,
                              'execution_count': cert['execution_count'], 'selected_definitions': cert['selected_definitions'],
                              'repeated_activities': {a: n for a, n in counts.items() if n > 1},
                              'raw_rows': raw,
                              'schedule': [{k: e[k] for k in ('activity_id', 'start_time', 'end_time')} for e in cert['executions']]}
                    persist_started = time.perf_counter()
                    with lock:
                        state['validation_seconds'] += validated - found
                        state['candidate_count'] += 1
                        if state['first_solution_seconds'] is None:
                            state['first_solution_seconds'] = validated
                        if state['best_makespan'] is None or cert['makespan'] < state['best_makespan']:
                            state.update(best_makespan=cert['makespan'], best_candidate_seconds=validated,
                                         best_execution_count=cert['execution_count'], best_repeated_activities=record['repeated_activities'])
                            write_json(directory / 'best.json', {'summary': record, 'certificate': cert})
                        state.update(lower_bound=bound, branches=self.num_branches, conflicts=self.num_conflicts,
                                     search_stats_sample_seconds=found)
                        append(directory / 'candidates.jsonl', record)
                        state['persistence_seconds'] += time.perf_counter() - persist_started
                    accept_bound(self.best_objective_bound)
                    save()
                except Exception:
                    with lock:
                        state['rejections'].append({'seconds': time.perf_counter()-started, 'error': traceback.format_exc()})
                    self.stop_search()

        result = solver.solve(model, Observe())
        with lock:
            state.update(solver_status=solver.status_name(result), branches=solver.num_branches,
                         conflicts=solver.num_conflicts, search_stats_sample_seconds=time.perf_counter()-started)
            bound = max(0, math.ceil(solver.best_objective_bound-scenario.start_time-1e-7))
            state['lower_bound'] = bound
            best = state['best_makespan']
            if best is not None and bound > best:
                state['rejections'].append({'error': 'Final bound exceeds replayed incumbent'})
            proven = best is not None and best == bound and not state['rejections']
            state.update(proven_optimal=proven,
                         status='ENCODING_ERROR' if state['rejections'] else 'PROVEN_OPTIMAL' if proven else
                         'PROVEN_INFEASIBLE' if result == cp_model.INFEASIBLE and best is None else
                         'FEASIBLE' if best is not None else 'UNKNOWN', phase='FINISHED',
                         response_stats=solver.response_stats())
            if proven and state['first_optimal_seconds'] is None:
                state['first_optimal_seconds'] = time.perf_counter()-started
    except BaseException:
        with lock:
            state.update(status='ERROR', phase='FINISHED', error=traceback.format_exc())
        raise
    finally:
        stopped.set()
        thread.join(timeout=6)
        save()
        write_json(directory / 'result.json', state)
    return state
