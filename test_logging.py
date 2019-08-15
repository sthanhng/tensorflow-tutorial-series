# Test Logging in Python

import logging
import logging.config


logging.config.fileConfig(fname='log.conf', disable_existing_loggers=False)

# Get the logger specified in the file
logger = logging.getLogger(__name__)

def main():
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')


if __name__ == "__main__":
    main()
