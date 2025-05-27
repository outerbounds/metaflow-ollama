from metaflow.exception import MetaflowException


class UnspecifiedRemoteStorageRootException(MetaflowException):
    headline = "Storage root not specified."

    def __init__(self, message):
        super(UnspecifiedRemoteStorageRootException, self).__init__(message)


class EmptyOllamaManifestCacheException(MetaflowException):
    headline = "Model not found."

    def __init__(self, message):
        super(EmptyOllamaManifestCacheException, self).__init__(message)


class EmptyOllamaBlobCacheException(MetaflowException):
    headline = "Blob not found."

    def __init__(self, message):
        super(EmptyOllamaBlobCacheException, self).__init__(message)
