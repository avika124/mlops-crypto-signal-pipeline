import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml


def setup_logger(log_file: str) -> logging.Logger:
    """Create a file logger for full job traceability."""
    logger = logging.getLogger("mlops_task")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_config(config_path: str) -> dict:
    """Load and validate required config values."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML config format: {exc}") from exc

    if not isinstance(cfg, dict):
        raise ValueError("Invalid configuration file structure: expected a YAML mapping.")

    required_keys = {"seed", "window", "version"}
    missing = required_keys - set(cfg.keys())
    if missing:
        raise ValueError(f"Invalid configuration file structure: missing keys {sorted(missing)}")

    seed = cfg["seed"]
    window = cfg["window"]
    version = cfg["version"]

    if not isinstance(seed, int):
        raise ValueError("Invalid configuration file structure: 'seed' must be an integer.")
    if not isinstance(window, int) or window <= 0:
        raise ValueError("Invalid configuration file structure: 'window' must be a positive integer.")
    if not isinstance(version, str) or not version:
        raise ValueError("Invalid configuration file structure: 'version' must be a non-empty string.")

    return {"seed": seed, "window": window, "version": version}


def load_data(input_path: str) -> pd.DataFrame:
    """Load CSV and enforce required schema."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Empty input file.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Invalid CSV file format: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Invalid CSV file format: {exc}") from exc

    if df.empty:
        raise ValueError("Empty input file.")

    if "close" not in df.columns:
        raise ValueError("Missing required columns in dataset: ['close']")

    return df


def generate_signals(df: pd.DataFrame, window: int) -> pd.Series:
    """Generate binary signal based on close vs rolling mean(close)."""
    rolling_mean = df["close"].rolling(window=window, min_periods=window).mean()
    signal = (df["close"] > rolling_mean).astype(int)
    return signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini MLOps technical assessment pipeline.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--output", required=True, help="Output metrics JSON path")
    parser.add_argument("--log-file", required=True, help="Log file path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger(args.log_file)
    start_time = time.perf_counter()
    version_for_error = "v1"

    try:
        logger.info("Job started")

        config = load_config(args.config)
        seed = config["seed"]
        window = config["window"]
        version = config["version"]
        version_for_error = version

        np.random.seed(seed)
        logger.info("Config loaded: seed=%s, window=%s, version=%s", seed, window, version)
        logger.info("Configuration verified")

        df = load_data(args.input)
        rows_processed = int(len(df))
        logger.info("Data loaded: %s rows", rows_processed)

        signal = generate_signals(df, window)
        logger.info("Rolling mean calculated with window=%s", window)
        logger.info("Signals generated")

        signal_rate = float(signal.mean())
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        metrics = {
            "version": version,
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": round(signal_rate, 4),
            "latency_ms": latency_ms,
            "seed": seed,
            "status": "success",
        }

        write_json(args.output, metrics)
        logger.info("Metrics: signal_rate=%.4f, rows_processed=%s", metrics["value"], rows_processed)
        logger.info("Job completed successfully in %sms", latency_ms)
        print(json.dumps(metrics, indent=2))
        return 0

    except Exception as exc:
        logger.error("Error encountered: %s", str(exc), exc_info=True)
        error_payload = {
            "version": version_for_error,
            "status": "error",
            "error_message": str(exc),
        }
        try:
            write_json(args.output, error_payload)
        except Exception as write_exc:
            logger.error("Failed to write error output JSON: %s", str(write_exc), exc_info=True)
        print(json.dumps(error_payload, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
