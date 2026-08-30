import os


def timeout() -> int:
    return int(os.environ.get("TIMEOUT", "30"))
