"""S2-03 review madde 9: resolve() dalları + entries_to_apply / audit_entries.

Streamlit'siz saf mantık testleri. `ALLOWED_TRANSFORMS`'un gerçek içeriğine
bağımlı olmamak için `pareto.cleaning.ledger.ALLOWED_TRANSFORMS`
monkeypatch'lenir — LedgerEntry validator'ı testler sırasında bu sahte
kataloğu görür, gerçek transform kayıt defteri değişse bile testler kırılmaz.
"""

from __future__ import annotations

import pytest

import pareto.cleaning.ledger as ledger_module
from pareto.cleaning.agent import (
    Resolution,
    ResolvedDecision,
    audit_entries,
    entries_to_apply,
    resolve,
)
from pareto.cleaning.ledger import LedgerEntry


@pytest.fixture(autouse=True)
def _allow_test_transforms(monkeypatch: pytest.MonkeyPatch) -> None:
    """LedgerEntry._must_be_vetted testlerde gerçek kataloğa değil buna baksın."""
    monkeypatch.setattr(
        ledger_module,
        "ALLOWED_TRANSFORMS",
        {"coerce_numeric", "drop_duplicates", "rename_column"},
    )


def _entry(
    *, transform_name: str = "coerce_numeric", params: dict | None = None, flagged: bool = True
) -> LedgerEntry:
    return LedgerEntry(
        bulgu="test bulgu",
        transform_name=transform_name,
        params=params if params is not None else {"col": "x"},
        gerekce="test gerekçe",
        belirsizlik_bayragi=flagged,
    ).stamped()


# --------------------------------------------------------------------------- #
# resolve() — gatekeeper dalları (madde 7 + 4 + 10)
# --------------------------------------------------------------------------- #
class TestResolve:
    def test_unflagged_entry_auto_approves(self) -> None:
        entry = _entry(flagged=False)
        decision = resolve(entry)
        assert decision.resolution == Resolution.APPROVED

    def test_flagged_entry_default_call_raises(self) -> None:
        """Bug (a): resolution verilmeden default auto_approve=True ile çağrılan
        bayraklı bir karar artık sessizce onaylanmıyor, ValueError fırlatıyor."""
        entry = _entry(flagged=True)
        with pytest.raises(ValueError):
            resolve(entry)

    def test_flagged_entry_without_resolution_and_auto_approve_false_raises(self) -> None:
        entry = _entry(flagged=True)
        with pytest.raises(ValueError):
            resolve(entry, auto_approve=False)

    def test_flagged_entry_explicit_resolution_respected_despite_auto_approve(self) -> None:
        """Bug (b): auto_approve=True iken verilen resolution artık yok sayılıp
        sessizce APPROVED'a düşmüyor; verilen karar (REJECTED) aynen dönüyor."""
        entry = _entry(flagged=True)
        decision = resolve(entry, auto_approve=True, resolution=Resolution.REJECTED)
        assert decision.resolution == Resolution.REJECTED

    def test_modified_without_params_raises(self) -> None:
        entry = _entry(flagged=True)
        with pytest.raises(ValueError):
            resolve(
                entry, auto_approve=False, resolution=Resolution.MODIFIED, modified_params=None
            )

    def test_modified_with_empty_dict_is_allowed(self) -> None:
        """Madde 10: modified_params={} artık geçerli MODIFIED sayılır
        (örn. drop_duplicates(subset=None) gibi meşru bir 'params'ı boşaltma)."""
        entry = _entry(transform_name="drop_duplicates", params={"subset": ["a"]}, flagged=True)
        decision = resolve(
            entry, auto_approve=False, resolution=Resolution.MODIFIED, modified_params={}
        )
        assert decision.resolution == Resolution.MODIFIED
        assert decision.modified_params == {}

    def test_modified_params_invalid_schema_raises_value_error(self) -> None:
        """Madde 4: L3 bypass kapatıldı. Yanlış anahtar ("column" yerine "col"
        olmalıydı) artık ham TypeError değil, ValueError olarak yükseliyor."""
        entry = _entry(transform_name="coerce_numeric", params={"col": "x"}, flagged=True)
        with pytest.raises(ValueError):
            resolve(
                entry,
                auto_approve=False,
                resolution=Resolution.MODIFIED,
                modified_params={"column": "x"},
            )

    def test_modified_params_valid_schema_passes(self) -> None:
        entry = _entry(transform_name="coerce_numeric", params={"col": "x"}, flagged=True)
        decision = resolve(
            entry,
            auto_approve=False,
            resolution=Resolution.MODIFIED,
            modified_params={"col": "y"},
        )
        assert decision.modified_params == {"col": "y"}

    def test_approved_resolution(self) -> None:
        entry = _entry(flagged=True)
        decision = resolve(entry, auto_approve=False, resolution=Resolution.APPROVED)
        assert decision.resolution == Resolution.APPROVED

    def test_rejected_resolution(self) -> None:
        entry = _entry(flagged=True)
        decision = resolve(entry, auto_approve=False, resolution=Resolution.REJECTED)
        assert decision.resolution == Resolution.REJECTED


# --------------------------------------------------------------------------- #
# entries_to_apply() — REJECTED elenir / MODIFIED günceller / APPROVED aynen geçer
# --------------------------------------------------------------------------- #
class TestEntriesToApply:
    def test_rejected_entries_filtered_out(self) -> None:
        entry = _entry(flagged=True)
        resolutions = {0: ResolvedDecision(entry=entry, resolution=Resolution.REJECTED)}
        assert entries_to_apply([entry], resolutions) == []

    def test_unresolved_entries_filtered_out(self) -> None:
        entry = _entry(flagged=True)
        assert entries_to_apply([entry], {}) == []

    def test_approved_entries_pass_through_unchanged(self) -> None:
        entry = _entry(params={"col": "x"})
        resolutions = {0: ResolvedDecision(entry=entry, resolution=Resolution.APPROVED)}
        out = entries_to_apply([entry], resolutions)
        assert len(out) == 1
        assert out[0].params == {"col": "x"}

    def test_modified_entries_get_new_params(self) -> None:
        entry = _entry(params={"col": "x"})
        resolutions = {
            0: ResolvedDecision(
                entry=entry, resolution=Resolution.MODIFIED, modified_params={"col": "y"}
            )
        }
        out = entries_to_apply([entry], resolutions)
        assert len(out) == 1
        assert out[0].params == {"col": "y"}

    def test_mixed_batch_filters_and_updates_correctly(self) -> None:
        e0 = _entry(params={"col": "a"})
        e1 = _entry(params={"col": "b"})
        e2 = _entry(params={"col": "c"})
        resolutions = {
            0: ResolvedDecision(entry=e0, resolution=Resolution.APPROVED),
            1: ResolvedDecision(entry=e1, resolution=Resolution.REJECTED),
            2: ResolvedDecision(
                entry=e2, resolution=Resolution.MODIFIED, modified_params={"col": "z"}
            ),
        }
        out = entries_to_apply([e0, e1, e2], resolutions)
        assert [o.params["col"] for o in out] == ["a", "z"]


# --------------------------------------------------------------------------- #
# audit_entries() — hiçbir karar elenmez, resolution + varsa yeni params yazılır
# --------------------------------------------------------------------------- #
class TestAuditEntries:
    def test_rejected_entries_kept_with_original_params(self) -> None:
        entry = _entry(params={"col": "x"})
        resolutions = {0: ResolvedDecision(entry=entry, resolution=Resolution.REJECTED)}
        out = audit_entries([entry], resolutions)
        assert len(out) == 1
        assert out[0].resolution == "rejected"
        assert out[0].params == {"col": "x"}

    def test_modified_entries_carry_new_params_and_resolution(self) -> None:
        entry = _entry(params={"col": "x"})
        resolutions = {
            0: ResolvedDecision(
                entry=entry, resolution=Resolution.MODIFIED, modified_params={"col": "y"}
            )
        }
        out = audit_entries([entry], resolutions)
        assert out[0].resolution == "modified"
        assert out[0].params == {"col": "y"}

    def test_unresolved_entries_keep_resolution_none(self) -> None:
        entry = _entry()
        out = audit_entries([entry], {})
        assert out[0].resolution is None

    def test_no_record_is_ever_dropped(self) -> None:
        """entries_to_apply'dan fark burada: REJECTED dahil hiçbir karar elenmez —
        diske yazılan denetim izi ile uygulanan liste çelişmesin diye."""
        e0 = _entry()
        e1 = _entry()
        resolutions = {
            0: ResolvedDecision(entry=e0, resolution=Resolution.REJECTED),
            1: ResolvedDecision(entry=e1, resolution=Resolution.APPROVED),
        }
        out = audit_entries([e0, e1], resolutions)
        assert len(out) == 2