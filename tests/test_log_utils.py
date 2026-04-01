import logging
import unittest

from jnlr.utils.log_utils import configure_logging, LevelColorFilter


class LogUtilsTests(unittest.TestCase):
    def test_configure_logging_returns_named_logger(self):
        logger = configure_logging("jnlr_test_log_utils", level=logging.DEBUG)
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "jnlr_test_log_utils")

    def test_logger_emits_info(self):
        logger = configure_logging("jnlr_test_log_utils_info", level=logging.INFO)
        with self.assertLogs("jnlr_test_log_utils_info", level="INFO") as cm:
            logger.info("test message")
        self.assertTrue(any("test message" in r.getMessage() for r in cm.records))

    def test_level_color_filter_sets_attributes(self):
        filt = LevelColorFilter()
        record = logging.LogRecord(
            name="x", level=logging.INFO, pathname="", lineno=0, msg="m", args=(), exc_info=None
        )
        self.assertTrue(filt.filter(record))
        self.assertTrue(hasattr(record, "color_start_message"))


if __name__ == "__main__":
    unittest.main()
