import logging


def get_logger(file_path):
    """
    只需传入 txt 文件的路径文件名，就可利用 logger.info("文本内容") 在 txt 文件内追加文本内容，同时也在控制台输出文本内容
    txt 文件如果不存在，会自动创建

    :param file_path:
    :return:
    """
    logger = logging.getLogger("自定义的日志器名字")
    logger.setLevel("INFO")
    console_handler = logging.StreamHandler()
    file_hadler = logging.FileHandler(file_path, mode="a", encoding="UTF-8")
    console_handler.setFormatter(logging.Formatter("%(asctime)s   %(message)s"))
    file_hadler.setFormatter(logging.Formatter("%(asctime)s   %(message)s"))
    logger.addHandler(console_handler)  # 传入控制台处理器
    logger.addHandler(file_hadler)
    return logger


if __name__ == "__main__":
    # 创建日志器
    logger = logging.getLogger("自定义的日志器名字")
    logger.setLevel("INFO")  # 设置什么级别以上的信息要记录

    # 定义处理器，即往哪里写日志
    console_handler = logging.StreamHandler()  # 往控制台写日志的处理器
    file_hadler = logging.FileHandler("文本文件的保存路径", mode="a")  # 往文本文件写日志的处理器

    # 设置日志的形式
    console_handler.setFormatter(logging.Formatter("设置控制台处理器的日志格式"))
    file_hadler.setFormatter(logging.Formatter("设置文本处理器的日志格式"))
    """
    日志格式常用信息
    %(name)s         日志器名字
    %(asctime)s      时间
    %(levelname)s    当前日志的等级
    %(message)s      日志的内容
    """

    # 将处理器设置传入日志器
    logger.addHandler(console_handler)  # 传入控制台处理器
    logger.addHandler(file_hadler)  # 传入文本处理器

    # 开始写不同级别日志
    logger.info("info级别的日志内容")
