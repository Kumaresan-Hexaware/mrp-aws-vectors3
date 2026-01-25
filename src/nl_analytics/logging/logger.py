import logging
import os
import glob
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

_INITIALIZED = False


class SizeTimestampRotatingFileHandler(RotatingFileHandler):
    """Rotate a single log file once it reaches maxBytes.

    - Current log always stays at the configured log_file path (e.g. logs/app.log)
    - When rotation happens, the previous file is renamed with a timestamp, e.g.:
        logs/app_20260124_153012.log
    - By default, we keep *all* rotated logs (backupCount=0). If backupCount > 0,
      we keep only the most recent N rotated logs.
    """

    def doRollover(self) -> None:
        if self.stream:
            try:
                self.stream.close()
            finally:
                self.stream = None

        base = self.baseFilename
        base_path = Path(base)
        log_dir = str(base_path.parent)
        stem = base_path.stem
        suffix = base_path.suffix or ".log"

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated = os.path.join(log_dir, f"{stem}_{ts}{suffix}")
        i = 1
        while os.path.exists(rotated):
            rotated = os.path.join(log_dir, f"{stem}_{ts}_{i}{suffix}")
            i += 1

        # Rename current -> timestamped
        if os.path.exists(base):
            try:
                os.replace(base, rotated)
            except Exception:
                # If rename fails for any reason, don't block the app
                pass

        # Optional retention (keep newest backupCount rotated logs)
        if self.backupCount and self.backupCount > 0:
            pattern = os.path.join(log_dir, f"{stem}_*{suffix}")
            files = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p), reverse=True)
            for f in files[self.backupCount:]:
                try:
                    os.remove(f)
                except Exception:
                    pass

        # Re-open the base file for new logs
        if not self.delay:
            self.stream = self._open()


def init_logging(
    log_level: str = "INFO",
    log_file: str = "logs/app.log",
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 0,             # 0 = keep all rotated logs
) -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    file_handler = SizeTimestampRotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    stream_handler = logging.StreamHandler()

    logging.basicConfig(level=level, format=fmt, handlers=[file_handler, stream_handler])
    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
