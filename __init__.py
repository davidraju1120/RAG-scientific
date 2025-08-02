import logging
import sys
import os

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('SciQAgent')
    logger.info("Logger initialized")
    return logger

# Add the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import and configure DSPy
try:
    import dspy
    # Configure DSPy with a default model
    # This will be overridden by the UI when the API key is provided
    # dspy.configure(lm=dspy.OpenAI(model='gpt-3.5-turbo', api_key='dummy-key'))
    pass
except ImportError:
    print("Warning: DSPy not installed. Some features may not work.")
    # Disable logging from libraries
    logging.getLogger('external_library').setLevel(logging.WARNING)
