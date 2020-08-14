"""
Provides file-based interprocess lock
"""
#import fcntl


class FileLocker:
    def __init__(self, lock_file) -> None:
        self.lock_file = lock_file

    def __enter__(self):
        self.fp = open(self.lock_file, "wb+")
        #fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__(self, _type, value, tb):
        #fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()
