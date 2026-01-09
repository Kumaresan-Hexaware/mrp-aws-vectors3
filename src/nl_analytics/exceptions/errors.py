class BaseException(Exception):
    """Base exception for nl_analytics."""

class DataIngestionError(BaseException):
    pass

class SchemaValidationError(BaseException):
    pass

class RetrievalError(BaseException):
    pass

class AgentExecutionError(BaseException):
    pass

class PlotlyRenderError(BaseException):
    pass

class ExportError(BaseException):
    pass
