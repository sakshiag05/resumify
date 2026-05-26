"""
SecureRAG — Logging & Observability
=====================================
Logs every query, guardrail result, and response to:
  - Console (INFO level)
  - securerag.log file (DEBUG level)
"""
import logging, os, datetime

LOG_FILE = "securerag.log"

def setup_logging():
    logger = logging.getLogger("securerag")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(LOG_FILE)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(fh)

    return logger

def read_logs(n=50):
    """Read last n lines from log file for display in UI."""
    if not os.path.exists(LOG_FILE):
        return ["No logs yet."]
    with open(LOG_FILE) as f:
        lines = f.readlines()
    return lines[-n:] if len(lines) > n else lines
