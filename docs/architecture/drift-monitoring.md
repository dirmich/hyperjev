# Drift monitoring

`TrafficMonitor` aggregates only typed response metadata: request/question
counts, abstain rate, mean confidence, and route counts. It does not retain
state, question text, request ids, or raw result payloads.

`compare_snapshots` compares a current window to a baseline and reports route
and abstain-rate shifts. A drift signal requests dataset/calibration review; it
never silently retrains or promotes a model.
