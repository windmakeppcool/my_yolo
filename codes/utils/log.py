import logging

LOGGER_NAME = 'YoloV5'
def get_logger(path):
    # logging.basicConfig(filename=path, filemode='w', level=logging.DEBUG)
    # create logger
    logger = logging.getLogger('yolov5')
    
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler(filename=path, mode='w')
    fh.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


if __name__ == "__main__":
    logger = get_logger("tmp")
    # 'application' code
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical message')

