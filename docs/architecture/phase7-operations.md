# v1 operations and drift gate

The Python serving path exposes `/metrics` backed by `TrafficMonitor`. It
aggregates request/question count, abstain rate, mean confidence, and route
counts. The monitor deliberately receives typed responses, not raw requests.

`compare_snapshots` reports a drift signal when abstain or route rates move
past the configured threshold. The required action is dataset/calibration
review; there is no automatic retraining or model promotion. Promotion still
requires an explicit model registry transition and rollback remains explicit.
