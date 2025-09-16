import sys
import logging
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass


class ErrorCategory(Enum):
    """Categories of errors that can occur in the system."""
    NETWORK = "network"
    VALIDATION = "validation" 
    PROCESSING = "processing"
    CONFIGURATION = "configuration"
    EXTERNAL_API = "external_api"
    FILE_IO = "file_io"
    DEPENDENCY = "dependency"


@dataclass
class ErrorContext:
    """Context information for an error."""
    url: Optional[str] = None
    metric_name: Optional[str] = None
    file_path: Optional[str] = None
    operation: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


class ACMEError(Exception):
    """Base exception class for ACME tool errors."""
    
    def __init__(
        self, 
        message: str, 
        category: ErrorCategory, 
        context: Optional[ErrorContext] = None,
        recoverable: bool = False,
        exit_code: int = 1
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.context = context or ErrorContext()
        self.recoverable = recoverable
        self.exit_code = exit_code
    
    def get_user_message(self) -> str:
        """Generate a user-friendly error message."""
        base_msg = self.message
        
        # Add context if available
        if self.context.url:
            base_msg += f" (URL: {self.context.url})"
        if self.context.metric_name:
            base_msg += f" (Metric: {self.context.metric_name})"
        if self.context.operation:
            base_msg += f" (Operation: {self.context.operation})"
            
        return base_msg
    
    def get_log_message(self) -> str:
        """Generate a detailed log message."""
        log_msg = f"[{self.category.value.upper()}] {self.message}"
        
        if self.context.additional_info:
            details = ", ".join(f"{k}={v}" for k, v in self.context.additional_info.items())
            log_msg += f" | Details: {details}"
            
        return log_msg


class NetworkError(ACMEError):
    """Network-related errors (API calls, downloads, etc.)."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message, 
            ErrorCategory.NETWORK, 
            context, 
            recoverable=True
        )


class ValidationError(ACMEError):
    """Input validation errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message, 
            ErrorCategory.VALIDATION, 
            context, 
            recoverable=False
        )


class ProcessingError(ACMEError):
    """Errors during model/data processing."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message, 
            ErrorCategory.PROCESSING, 
            context, 
            recoverable=True
        )


class ConfigurationError(ACMEError):
    """Configuration and setup errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message, 
            ErrorCategory.CONFIGURATION, 
            context, 
            recoverable=False
        )


def handle_error(error: Exception, logger: logging.Logger) -> None:
    """
    Central error handler that logs appropriately and exits with proper code.
    
    Args:
        error: The exception that occurred
        logger: Logger instance for recording the error
    """
    if isinstance(error, ACMEError):
        # Log the detailed message
        logger.error(error.get_log_message())
        
        # Print user-friendly message to stderr
        print(f"Error: {error.get_user_message()}", file=sys.stderr)
        
        # Exit with appropriate code
        sys.exit(error.exit_code)
    else:
        # Handle unexpected errors
        logger.error(f"Unexpected error: {str(error)}", exc_info=True)
        print(f"Error: An unexpected error occurred. Check the log file for details.", file=sys.stderr)
        sys.exit(1)


def retry_on_network_error(func, max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry operations that might fail due to network issues.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    import time
    
    def wrapper(*args, **kwargs):
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except NetworkError as e:
                last_error = e
                if attempt < max_retries:
                    logging.warning(f"Network error on attempt {attempt + 1}, retrying in {delay}s: {e.message}")
                    time.sleep(delay)
                    delay *= 1.5  # Exponential backoff
                else:
                    logging.error(f"Network error after {max_retries + 1} attempts: {e.message}")
            except Exception as e:
                # Don't retry non-network errors
                raise e
        
        raise last_error
    
    return wrapper


def safe_metric_calculation(metric_name: str, url: str):
    """
    Decorator for metric calculations that handles errors gracefully.
    Returns 0.0 score and logs the error if calculation fails.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ACMEError as e:
                # Add metric context
                e.context.metric_name = metric_name
                e.context.url = url
                logging.warning(f"Metric calculation failed for {metric_name}: {e.message}")
                
                # Return default score for failed metric
                return {
                    f"{metric_name}": 0.0,
                    f"{metric_name}_latency": 0
                }
            except Exception as e:
                # Wrap unexpected errors
                context = ErrorContext(metric_name=metric_name, url=url)
                wrapped_error = ProcessingError(
                    f"Unexpected error calculating {metric_name}: {str(e)}", 
                    context
                )
                logging.warning(wrapped_error.get_log_message())
                
                return {
                    f"{metric_name}": 0.0,
                    f"{metric_name}_latency": 0
                }
        
        return wrapper
    return decorator


class ErrorReporter:
    """Collects and reports errors encountered during processing."""
    
    def __init__(self):
        self.errors: list[ACMEError] = []
        self.warnings: list[str] = []
    
    def add_error(self, error: ACMEError) -> None:
        """Add an error to the collection."""
        self.errors.append(error)
        logging.error(error.get_log_message())
    
    def add_warning(self, message: str, context: Optional[ErrorContext] = None) -> None:
        """Add a warning message."""
        warning_msg = message
        if context and context.url:
            warning_msg += f" (URL: {context.url})"
        
        self.warnings.append(warning_msg)
        logging.warning(warning_msg)
    
    def has_critical_errors(self) -> bool:
        """Check if there are any non-recoverable errors."""
        return any(not error.recoverable for error in self.errors)
    
    def get_summary(self) -> str:
        """Get a summary of all errors and warnings."""
        summary = []
        
        if self.errors:
            summary.append(f"Encountered {len(self.errors)} error(s)")
            for error in self.errors:
                summary.append(f"  - {error.get_user_message()}")
        
        if self.warnings:
            summary.append(f"Encountered {len(self.warnings)} warning(s)")
            for warning in self.warnings[:5]:  # Limit to first 5 warnings
                summary.append(f"  - {warning}")
            
            if len(self.warnings) > 5:
                summary.append(f"  ... and {len(self.warnings) - 5} more warnings")
        
        return "\n".join(summary)


# Example usage patterns for the project
def example_url_processing(url: str) -> Dict[str, Any]:
    """Example of how to use error handling in URL processing."""
    try:
        # Validate URL format
        if not url.startswith(("http://", "https://")):
            raise ValidationError(
                "Invalid URL format - must start with http:// or https://",
                ErrorContext(url=url, operation="url_validation")
            )
        
        # Process URL (this would be your actual logic)
        result = process_model_url(url)
        return result
        
    except ACMEError:
        # Re-raise ACME errors as-is
        raise
    except Exception as e:
        # Wrap unexpected errors
        context = ErrorContext(url=url, operation="url_processing")
        raise ProcessingError(f"Failed to process URL: {str(e)}", context)


@retry_on_network_error
def fetch_huggingface_data(url: str) -> Dict[str, Any]:
    """Example of network operation with retry logic."""
    try:
        # Your HuggingFace API call here
        response = make_api_call(url)
        return response
    except requests.RequestException as e:
        context = ErrorContext(url=url, operation="huggingface_api")
        raise NetworkError(f"Failed to fetch data from HuggingFace: {str(e)}", context)


# This would be imported and used in your main application
def process_model_url(url: str) -> Dict[str, Any]:
    """Placeholder for actual URL processing logic."""
    pass

def make_api_call(url: str) -> Dict[str, Any]:
    """Placeholder for actual API call logic.""" 
    pass