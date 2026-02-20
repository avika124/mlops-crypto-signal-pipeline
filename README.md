# MLOps Engineering Internship - Technical Assessment 2

This repository contains a deterministic batch-style Python pipeline that reads OHLCV data, computes a rolling-mean-based signal from the `close` column, and writes machine-readable metrics and logs.

## Setup Instructions

```bash
# Install dependencies
pip install -r requirements.txt
```

If your system Python is externally managed, create a virtual environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Local Execution Instructions

```bash
# Run locally
python run.py --input data.csv --config config.yaml \
    --output metrics.json --log-file run.log
```

## Docker Instructions

```bash
# Build the Docker image
docker build -t mlops-task .

# Run the container
docker run --rm mlops-task
```

The container includes `data.csv` and `config.yaml`, writes `metrics.json` and `run.log`, prints final metrics to stdout, exits with code `0` on success, and exits non-zero on failure.

## Expected Output

The `metrics.json` file has the following structure:

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.5011,
  "latency_ms": 24,
  "seed": 42,
  "status": "success"
}
```

If an error occurs, output format is:

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Description of what went wrong"
}
```

## Dependencies

- `pandas`
- `numpy`
- `pyyaml`

## Notes on Reproducibility and Observability

- Configuration is loaded only from `config.yaml` (`seed`, `window`, `version`).
- The random seed is set using `numpy.random.seed(seed)`.
- All calculation logic uses the `close` column only.
- Logs in `run.log` include job start, config verification, ingestion summary, processing steps, metrics summary, completion status, and errors.
