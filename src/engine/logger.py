import os
from datetime import datetime
from typing import Union

class Logger:
    def __init__(self, log_dir):
        """Initialize the logger with a directory to save logs"""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'training_log.txt')
        self._create_log_file()
        
    def _create_log_file(self):
        """Create a new log file with timestamp"""
        with open(self.log_file, 'w') as f:
            f.write(f"Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")
    
    def log(self, message, print_to_console=True):
        """Log a message to file and optionally print to console"""
        if print_to_console:
            print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
    
    def log_metrics(self, metrics: dict, model_id: Union[str, None]):
        """Log metrics in a formatted way with optional best score tracking"""
        
        message = f"Evaluation Results"
        if model_id is not None:
            message += f" at Epoch {model_id}"
        message += "\n"

        headers = metrics.keys()
        
        # Create the header line
        message += "| " + " | ".join(headers) + " |\n"

        # Create the metrics line
        metric_values = []
        for metric in headers:
            metric_values.append(f"{metrics[metric]:.4f}")

        # Append the metrics line
        message += "| " + " | ".join(metric_values) + " |\n"
        # Log the message
        self.log(message)

def setup_logger(save_dir):
    """Convenience function to create a logger instance"""
    return Logger(save_dir)

def log_message(logger, message, print_to_console=True):
    """Log a message using the logger instance"""
    if logger is not None:
        logger.log(message, print_to_console)
    elif print_to_console:
        print(message)
