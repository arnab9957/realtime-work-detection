# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""Logging formatters used by ``multihmr2`` (timedelta and ANSI color)."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import ClassVar

logging.captureWarnings(True)


class TimedeltaFormatter(logging.Formatter):
    """Logging formatter for adding a timedelta field to log records."""

    def has_fmt_field(self, field: str) -> bool:
        """Check if the formatter's format string contains a specific field.

        Parameters:
            field (str): The name of the field to check for in the format string.

        Returns:
            out (bool): :py:`True` if the field is present in the format string,
                :py:`False` otherwise.
        """
        # StrFormatStyle and StringTemplateStyle inherit from PercentStyle, so
        # they must be checked first.
        if isinstance(self._style, logging.StrFormatStyle):
            search = "{" + field
        elif isinstance(self._style, logging.StringTemplateStyle):
            search = "${" + field + "}"
        elif isinstance(self._style, logging.PercentStyle):
            search = "%(" + field + ")"
        return search in self._fmt

    def uses_timedelta(self) -> bool:
        """Check if the formatter's format string contains the timedelta field.

        Returns:
            out (bool): :py:`True` if ``asctimedelta`` field is present in the format
            string, :py:`False` otherwise
        """
        return self.has_fmt_field("asctimedelta")

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with the timedelta field if present.

        Parameters:
            record (logging.LogRecord): The log record to format.

        Returns:
            out (str): The formatted log record with the timedelta field if present.
        """
        if self.uses_timedelta():
            record.asctimedelta = str(timedelta(milliseconds=record.relativeCreated))
        return super().format(record)


class ColorFormatter(TimedeltaFormatter):
    """Add ANSI color codes to log records based on the log level."""

    COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: "\x1b[34m",  # blue
        logging.INFO: "\x1b[32m",  # green
        logging.WARNING: "\x1b[33m",  # yellow
        logging.ERROR: "\x1b[91m",  # red
        logging.CRITICAL: "\x1b[1;41m",  # bold white on red
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with ANSI color codes based on the log level.

        Parameters:
            record (logging.LogRecord): The log record to format.

        Returns:
            out (str): The formatted log record with ANSI color codes based on the log
                level.
        """
        fmt_msg = super().format(record)
        if getattr(record, "no_color", False):
            return fmt_msg
        else:
            c = self.COLORS[record.levelno]
            return f"{c}{fmt_msg}\x1b[0m"


_handler = logging.StreamHandler()
_handler.setFormatter(
    ColorFormatter(
        "[{asctimedelta}] {levelname: >7} {name} "
        + "({filename}:{lineno} / {processName}) : {message}",
        style="{",
    )
)

logger = logging.getLogger("multihmr2")
logger.setLevel(logging.INFO)
logger.addHandler(_handler)
# Do not propagate to root logger to avoid duplicate logs
logger.propagate = False
