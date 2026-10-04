"""Data quality checks (phase A-half).

One function per check. Each takes a loaded ``Shot`` (or a file path for the
metadata-level checks) and returns a flat dict of metrics with a short prefix per
check. The functions never print or write files; ``report.py`` aggregates the
per-shot rows into statuses, and ``scripts/run_qa.py`` drives the run.

Check ids follow ``docs/QA.md``. Metric prefixes:

====  =======================  =========
Id    Check                    Prefix
====  =======================  =========
0.1   File inventory           ``inv_``
0.14  Column sets              ``col_``
0.2   Ip time axis             ``ip_``
0.4   Current units            ``cur_``
0.3   MAST Thomson alignment   ``ts_``
0.7   Missing Thomson          ``tsm_``
0.9   Coil geometry            ``coil_``
0.10  Grid                     ``grid_``
0.5   Flux map indexing        ``psi_``
0.6   Boundary contour         ``lcfs_``
0.8   magnetics_dsep           ``dsep_``
0.11  Labels and time bases    ``time_``, ``lab_``
0.15  Scorer comparison        ``sc_``
====  =======================  =========
"""
