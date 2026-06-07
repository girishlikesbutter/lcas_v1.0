# 0006 — `oracle_clean` should distinguish truth-in-search from truth-for-labels

Status: proposed — 2026-06-06

The `oracle_clean` flag (and the `oracle-leak` gate it pairs with) is currently binary on
*any* read of a `truth_*` value: read truth at all → `false`. That is too blunt. It conflates
a real leak (truth used in the **search / ranking / selection** path — the s058/s059
injection trap) with a harmless read (truth used **only for post-hoc error LABELS** —
dir-error, geo-off, "did the truth-near rep survive"). s114 is the motivating case: its
grid → coarse-RMSE rank → windowed-photometry polish → ρ-banding → winner are all scored
against the **observed light curve**, fully blind; truth is read solely to *report* how far
the blind winner landed (2.27°). Under the binary flag that genuinely-blind run is stamped
`oracle_clean: false`, indistinguishable from a run that cheated.

Decision: make the criterion **less dumb**, not work around it. The flag should capture
"was truth in the inference path?" — split into something like `search_blind: true` (no
truth in ranking/selection/objective) + `truth_read_for_labels: true` (post-hoc scoring
only). The fix is to the *criterion*, NOT to author runs that hide their truth reads in a
separate pass to satisfy a binary flag (that defeats the purpose).

Alternatives rejected: (a) split each run into a blind pass + a separate labels pass purely
to earn `oracle_clean: true` — games the flag, adds no epistemic value, the operator's
explicit objection; (b) leave it binary — keeps blind results mislabelled as tainted and
weakens the `oracle-rests` claim gate (a live claim resting on a label-only run is fine; one
resting on a search-leak run is not — the binary flag can't tell them apart).

Open: exact field shape + how `gate_check.py` (`oracle-coherent`, `oracle-rests`) and the
run_record schema encode it; backfill s114 once landed.
