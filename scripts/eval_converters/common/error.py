class AdapterError(Exception):
    """Base exception for adapter errors"""
    pass

class TransformationError(AdapterError):
    """Raised when transformation logic fails"""
    pass