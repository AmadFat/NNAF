import enum

class LogMode(enum.Enum):
    DEBUG = 1 << 0
    NORMAL = 1 << 1
    IMPORTANT = 1 << 2

class LogFmt(enum.Enum):
    