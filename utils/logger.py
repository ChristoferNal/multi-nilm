import sys

DEBUG: bool = True
TIMING: bool = True
TRACE_MEMORY: bool = True
INFO: bool = True
MB: int = 1024 * 1024


def debug(d):
    if DEBUG:
        print('DEBUG: ' + d)


def info(i):
    if INFO:
        print('INFO: ' + i)


def timing(t):
    if TIMING:
        print('TIMING: ' + t)


def debug_mem(message, obj):
    if TRACE_MEMORY:
        print('MEMORY: {}'.format(message.format(sys.getsizeof(obj) / MB)))


def trace_mem(o):
    return sys.getsizeof(o) / MB
