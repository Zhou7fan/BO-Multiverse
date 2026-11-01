# -*- coding: utf-8 -*-
"""Simple task scheduler for exp4 runs (subprocess-based, resumable)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
EXP4 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP4)
for _p in (ROOT, EXP4, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_LOCK = threading.Lock()


def _update(manifest_path, task_id, **kw):
    with _LOCK:
        data = {}
        if os.path.exists(manifest_path):
            try:
                data = json.load(open(manifest_path, encoding="utf-8"))
            except Exception:
                data = {}
        rec = data.setdefault(task_id, {})
        rec.update(kw)
        tmp = manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, manifest_path)


def _task_id(t):
    return f"{t['paper']}/{t['algo']}/seed{t['seed']}"


def run_task(t, manifest, python):
    tid = _task_id(t)
    out = os.path.join(t["out"], t["paper"], t["algo"], f"seed{t['seed']}")
    os.makedirs(out, exist_ok=True)
    if t.get("resume", True) and os.path.exists(os.path.join(out, "summary.json")):
        try:
            s = json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))
            if s.get("status") in ("success", "cap", "timeout", "not_run", "error"):
                _update(manifest, tid, status="skipped", finished=time.time())
                return "skipped"
        except Exception:
            pass
    log_path = os.path.join(out, "scheduler.log")
    cmd = [python, os.path.join(HERE, "runner.py"),
           "--space", t["space"], "--algo", t["algo"], "--seed", str(t["seed"]),
           "--out", t["out"], "--warmup", str(t.get("warmup", 20)),
           "--budget", str(t.get("budget", 100)),
           "--max-seconds", str(t.get("max_seconds", 3600)),
           "--max-proposals", str(t.get("max_proposals", 500)),
           "--pool-n", str(t.get("pool_n", 500000))]
    if t.get("pool_file"):
        cmd += ["--pool-file", t["pool_file"]]
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    t0 = time.time()
    _update(manifest, tid, status="running", started=t0, cmd=cmd)
    with open(log_path, "w", encoding="utf-8") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    status = "done" if p.returncode == 0 else f"failed:{p.returncode}"
    _update(manifest, tid, status=status, finished=time.time(),
            seconds=round(time.time() - t0, 1), returncode=p.returncode)
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()
    tasks = json.load(open(args.tasks, encoding="utf-8"))
    os.makedirs(args.out, exist_ok=True)
    manifest = args.manifest or os.path.join(args.out, "task_manifest.json")
    print(f"tasks={len(tasks)} workers={args.workers} manifest={manifest}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_task, t, manifest, args.python): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            try:
                st = fut.result()
            except Exception as exc:
                st = f"exception:{type(exc).__name__}:{exc}"
                _update(manifest, _task_id(t), status=st, finished=time.time())
            print(f"[{i}/{len(tasks)}] {_task_id(t)} -> {st} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print("ALL_DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
