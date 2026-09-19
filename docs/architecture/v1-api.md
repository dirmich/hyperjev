# v1 API surface

The local server now covers the initial public contract:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health/live`, `/health` | liveness |
| GET | `/health/ready` | readiness and active serving phase |
| GET | `/v1/tasks` | versioned task registry |
| POST | `/v1/tasks/validate` | compatibility validation for task references |
| POST | `/v1/decide` | one typed decision request |
| POST | `/v1/batch/decide` | multiple independent typed requests |
| GET | `/v1/models` | local model manifest registry |
| POST | `/v1/models/{id}/activate` | explicit model activation |
| GET | `/metrics` | minimal serving metric surface |

The Python server is still the Phase 0/1/POC runtime. Rust API crates remain
the production runtime migration target, and no trained Student checkpoint is
implicitly activated by these endpoints.
