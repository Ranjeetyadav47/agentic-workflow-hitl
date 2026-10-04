"""Command-line entry point: `python -m agentflow <command>`.

  generate-data            build the synthetic dataset (seeded, reproducible)
  run --all | --file F     process invoices through the workflow
  review list|approve|reject   the human-in-the-loop queue
  dlq list|replay          dead-letter queue
  show RUN_ID              audit trail for one run
  eval                     score the workflow against the labelled set, write a report
  stats                    ledger and step statistics
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import Settings
from .data import generate, load_invoices
from .eval import run_eval, write_report
from .logging_utils import configure_logging
from .orchestrator import Orchestrator
from .store import Store
from .tools.erp import ERP


def _orch(settings: Settings) -> tuple[Orchestrator, Store]:
    store = Store(settings.db_path)
    erp = ERP(settings.data_dir)
    return Orchestrator(settings, store, erp), store


def _print_result(res) -> None:
    print(json.dumps({
        "run_id": res.run_id, "invoice_id": res.invoice_id, "status": res.status.value,
        "decision": res.decision.value if res.decision else None, "reasons": res.reasons,
        "ledger_entry_id": res.ledger_entry_id, "error": res.error,
    }))


def cmd_generate(args, settings):
    counts = generate(out_dir=args.out, seed=args.seed)
    print(f"wrote {sum(counts.values())} invoices to {args.out}/")
    for k, v in sorted(counts.items()):
        print(f"  {k:<24}{v}")


def cmd_run(args, settings):
    orch, store = _orch(settings)
    try:
        if args.all:
            for inv in load_invoices(settings.data_dir):
                _print_result(orch.process(inv["invoice_id"], inv["text"]))
        elif args.file:
            with open(args.file) as f:
                text = f.read()
            _print_result(orch.process(args.invoice_id or args.file, text))
        else:
            print("run: pass --all or --file PATH", file=sys.stderr)
            sys.exit(2)
    finally:
        store.close()


def cmd_review(args, settings):
    orch, store = _orch(settings)
    try:
        if args.action == "list":
            rows = store.pending_reviews()
            if not rows:
                print("review queue is empty")
            for r in rows:
                print(f"{r['run_id']}  {r['invoice_id']}\n    {r['summary']}\n    reasons: {', '.join(json.loads(r['reasons_json']))}")
        else:
            _print_result(orch.resume(args.run_id, args.action, args.note or ""))
    finally:
        store.close()


def cmd_dlq(args, settings):
    orch, store = _orch(settings)
    try:
        if args.action == "list":
            rows = store.pending_dead_letters()
            if not rows:
                print("dead-letter queue is empty")
            for r in rows:
                print(f"{r['run_id']}  {r['invoice_id']}  step={r['step']}  attempts={r['attempts']}\n    {r['error']}")
        else:
            _print_result(orch.replay(args.run_id))
    finally:
        store.close()


def cmd_show(args, settings):
    store = Store(settings.db_path)
    try:
        row = store.get_run(args.run_id)
        if row is None:
            print(f"run {args.run_id} not found", file=sys.stderr)
            sys.exit(1)
        state = json.loads(row["state_json"])
        print(f"run {row['run_id']}  invoice {row['invoice_id']}  status={row['status']}  decision={row['decision']}")
        if state.get("policy"):
            print("rule trace:")
            for t in state["policy"]["rule_trace"]:
                print(f"  {t}")
        if state.get("note"):
            print(f"reviewer note: {state['note']['summary']}")
        print("steps:")
        for s in store.steps_for(args.run_id):
            err = f"  error={s['error']}" if s["error"] else ""
            print(f"  {s['step']:<8} attempt={s['attempt']} {s['status']:<7} {s['latency_ms'] or 0:>8.2f}ms"
                  f"  tokens={s['prompt_tokens']}+{s['completion_tokens']}{err}")
    finally:
        store.close()


def cmd_eval(args, settings):
    results = []
    if args.compare:
        results.append(run_eval(settings.data_dir, normalise_indian_numbers=False, label="before fix"))
        results.append(run_eval(settings.data_dir, normalise_indian_numbers=True, label="after fix"))
    else:
        results.append(run_eval(settings.data_dir, normalise_indian_numbers=not args.no_indian_numbers,
                                label="no-indian-numbers" if args.no_indian_numbers else "default"))
    path = write_report(results, args.out)
    for r in results:
        print(f"[{r.label}] accuracy={r.accuracy:.1%}  false_auto_approvals={r.false_auto_approvals}  "
              f"missed_rejects={r.missed_rejects}  human_review_rate={r.human_review_rate:.1%}  "
              f"dlq={r.dead_lettered}  n={r.n}")
    print(f"report: {path}")


def cmd_stats(args, settings):
    store = Store(settings.db_path)
    try:
        print(json.dumps({
            "runs_by_status": {s: len(store.list_runs(status=s, limit=100000))
                               for s in ("approved", "rejected", "awaiting_review", "failed", "running", "pending")},
            "ledger_entries": len(store.ledger_entries()),
            "pending_reviews": len(store.pending_reviews()),
            "dead_letters": len(store.pending_dead_letters()),
            "steps": store.step_stats(),
        }, indent=2))
    finally:
        store.close()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="agentflow", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=None, help="SQLite path (default: $AGENTFLOW_DB or agentflow.db)")
    p.add_argument("--data-dir", default=None, help="data directory (default: $AGENTFLOW_DATA_DIR or data)")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--quiet", action="store_true", help="suppress JSON logs on stderr")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate-data"); g.add_argument("--out", default="data"); g.add_argument("--seed", type=int, default=42)
    g.set_defaults(fn=cmd_generate)

    r = sub.add_parser("run"); r.add_argument("--all", action="store_true"); r.add_argument("--file")
    r.add_argument("--invoice-id"); r.set_defaults(fn=cmd_run)

    rv = sub.add_parser("review"); rv.add_argument("action", choices=["list", "approve", "reject"])
    rv.add_argument("run_id", nargs="?"); rv.add_argument("--note"); rv.set_defaults(fn=cmd_review)

    d = sub.add_parser("dlq"); d.add_argument("action", choices=["list", "replay"]); d.add_argument("run_id", nargs="?")
    d.set_defaults(fn=cmd_dlq)

    s = sub.add_parser("show"); s.add_argument("run_id"); s.set_defaults(fn=cmd_show)

    e = sub.add_parser("eval"); e.add_argument("--out", default="reports/eval_report.md")
    e.add_argument("--no-indian-numbers", action="store_true", help="reproduce the amount-parsing bug")
    e.add_argument("--compare", action="store_true", help="run before/after the fix and report both")
    e.set_defaults(fn=cmd_eval)

    st = sub.add_parser("stats"); st.set_defaults(fn=cmd_stats)

    args = p.parse_args(argv)
    overrides = {}
    if args.db:
        overrides["db_path"] = args.db
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    settings = Settings.from_env(**overrides)
    configure_logging(args.log_level, quiet=args.quiet)
    try:
        args.fn(args, settings)
    except BrokenPipeError:
        # `python -m agentflow ... | head` closed the pipe; that is not an error.
        import os
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)


if __name__ == "__main__":
    main()
