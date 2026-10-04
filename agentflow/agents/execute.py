"""Execute agent: post an approved invoice to the ledger.

Idempotent: the ledger has a unique index on (vendor, invoice_number), so a replayed
run cannot double-post. The entry id is derived from the run id for the same reason.
"""

from __future__ import annotations

import sqlite3

from ..models import ExtractedInvoice, PermanentError, TransientError
from ..store import Store


class ExecuteAgent:
    name = "execute"

    def __init__(self, store: Store):
        self.store = store

    def run(self, run_id: str, inv: ExtractedInvoice, approved_by: str) -> str:
        if inv.total_amount is None or not inv.invoice_number or not inv.vendor_name:
            raise PermanentError("execute: cannot post an invoice without number, vendor and total")
        entry_id = f"LED-{run_id[:8]}"
        try:
            self.store.post_ledger(
                entry_id=entry_id,
                run_id=run_id,
                invoice_number=inv.invoice_number,
                vendor_name=inv.vendor_name,
                amount=inv.total_amount,
                currency=inv.currency or "INR",
                approved_by=approved_by,
            )
        except sqlite3.IntegrityError as e:
            existing = self.store.find_ledger_duplicate(inv.vendor_name, inv.invoice_number)
            raise PermanentError(f"execute: already posted as {existing}") from e
        except sqlite3.OperationalError as e:
            # "database is locked" and friends: another writer, not a bad invoice. Retry.
            raise TransientError(f"execute: ledger unavailable: {e}") from e
        return entry_id
