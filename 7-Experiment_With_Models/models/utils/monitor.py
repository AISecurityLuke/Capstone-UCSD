import time
import psutil
import os
import gc
import logging
from datetime import datetime
import threading
from functools import wraps

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/experiment.log'),
        logging.StreamHandler()
    ]
)

class ResourceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.memory_warnings = 0
        self.disk_warnings = 0
        
    def check_memory(self):
        """Monitor memory usage and warn if high"""
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            self.memory_warnings += 1
            logging.warning(f"High memory usage: {memory.percent}% ({self.memory_warnings} warnings)")
            gc.collect()
        return memory.percent
    
    def check_disk(self):
        """Monitor disk space"""
        disk = psutil.disk_usage('.')
        if disk.percent > 90:
            self.disk_warnings += 1
            logging.warning(f"Low disk space: {disk.percent}% used ({self.disk_warnings} warnings)")
        return disk.percent
    
    def estimate_time_remaining(self, current_iter, total_iters, elapsed_time):
        """Estimate time remaining based on current progress"""
        if current_iter == 0:
            return "Unknown"
        
        avg_time_per_iter = elapsed_time / current_iter
        remaining_iters = total_iters - current_iter
        remaining_time = avg_time_per_iter * remaining_iters
        
        if remaining_time > 3600:
            return f"{remaining_time/3600:.1f} hours"
        elif remaining_time > 60:
            return f"{remaining_time/60:.1f} minutes"
        else:
            return f"{remaining_time:.0f} seconds"
    
    def cleanup_old_models(self, keep_recent=5):
        """Clean up old model files to save disk space"""
        models_dir = 'models'
        if not os.path.exists(models_dir):
            return
            
        model_files = []
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.endswith(('.pkl', '.h5', '.pb')):
                    filepath = os.path.join(root, file)
                    model_files.append((filepath, os.path.getmtime(filepath)))
        
        # Sort by modification time and keep only recent ones
        model_files.sort(key=lambda x: x[1], reverse=True)
        
        for filepath, _ in model_files[keep_recent:]:
            try:
                os.remove(filepath)
                logging.info(f"🧹 Cleaned up old model: {filepath}")
            except Exception as e:
                logging.warning(f"Failed to clean up {filepath}: {e}")

def timeout_handler(timeout_seconds=1800):  # 30 minutes default
    """Decorator to add timeout to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)
            
            if thread.is_alive():
                logging.error(f"⏰ Timeout after {timeout_seconds}s for {func.__name__}")
                return None
            elif exception[0]:
                raise exception[0]
            else:
                return result[0]
        return wrapper
    return decorator

def monitor_resources(func):
    """Decorator to monitor resources during function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = ResourceMonitor()
        start_time = time.time()
        
        logging.info(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            
            elapsed = time.time() - start_time
            memory_used = monitor.check_memory()
            disk_used = monitor.check_disk()
            
            logging.info(f"Completed {func.__name__} in {elapsed:.1f}s (Memory: {memory_used}%, Disk: {disk_used}%)")
            
            # Cleanup after successful completion
            monitor.cleanup_old_models()
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logging.error(f"Failed {func.__name__} after {elapsed:.1f}s: {e}")
            raise
    
    return wrapper

# Global monitor instance
monitor = ResourceMonitor()
