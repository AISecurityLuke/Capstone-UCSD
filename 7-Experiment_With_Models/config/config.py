import json
import os

def load_config():
    """Load configuration from config.json"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Add missing keys with defaults
    if 'dl_models' not in config:
        config['dl_models'] = ['bert', 'distilbert']
    
    # Alias enable_bert to enable_dl for consistency
    if 'enable_dl' not in config:
        config['enable_dl'] = config.get('enable_bert', False)
    
    return config

CONFIG = load_config() 