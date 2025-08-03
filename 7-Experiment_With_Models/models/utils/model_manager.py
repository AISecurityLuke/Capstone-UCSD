import os
import time
import logging
from transformers import (
    BertTokenizerFast, DistilBertTokenizerFast,
    BertForSequenceClassification, DistilBertForSequenceClassification,
    RobertaTokenizerFast, RobertaForSequenceClassification,
    AlbertTokenizerFast, AlbertForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification
)
import requests
from utils.monitor import timeout_handler

class ModelManager:
    def __init__(self):
        self.models = {
            'bert': {
                'name': 'bert-base-uncased',
                'tokenizer_cls': BertTokenizerFast,
                'model_cls': BertForSequenceClassification,
                'size_mb': 440
            },
            'distilbert': {
                'name': 'distilbert-base-uncased',
                'tokenizer_cls': DistilBertTokenizerFast,
                'model_cls': DistilBertForSequenceClassification,
                'size_mb': 260
            },
            'roberta': {
                'name': 'roberta-base',
                'tokenizer_cls': RobertaTokenizerFast,
                'model_cls': RobertaForSequenceClassification,
                'size_mb': 500
            },
            'albert': {
                'name': 'albert-base-v2',
                'tokenizer_cls': AlbertTokenizerFast,
                'model_cls': AlbertForSequenceClassification,
                'size_mb': 45
            },
            'distilroberta': {
                'name': 'distilroberta-base',
                'tokenizer_cls': RobertaTokenizerFast,
                'model_cls': RobertaForSequenceClassification,
                'size_mb': 300
            },
            'bert_large': {
                'name': 'bert-large-uncased',
                'tokenizer_cls': BertTokenizerFast,
                'model_cls': BertForSequenceClassification,
                'size_mb': 1320
            },
            'roberta_large': {
                'name': 'roberta-large',
                'tokenizer_cls': RobertaTokenizerFast,
                'model_cls': RobertaForSequenceClassification,
                'size_mb': 1350
            }
        }
        self.download_timeout = 600  # 10 minutes
        self.cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
        
    def check_model_availability(self, model_key):
        """Check if model is available locally"""
        if model_key not in self.models:
            return False
            
        model_info = self.models[model_key]
        model_name = model_info['name']
        
        # Check if model files exist in cache
        model_path = os.path.join(self.cache_dir, 'models--' + model_name.replace('/', '--'))
        return os.path.exists(model_path)
    
    @timeout_handler(600)  # 10 minute timeout
    def download_model(self, model_key):
        """Download model with timeout and progress tracking"""
        if model_key not in self.models:
            raise ValueError(f"Unknown model: {model_key}")
            
        model_info = self.models[model_key]
        model_name = model_info['name']
        
        logging.info(f"Successfully downloaded {model_key} ({model_name}) - {model_info['size_mb']}MB")
        
        try:
            # Download tokenizer
            tokenizer = model_info['tokenizer_cls'].from_pretrained(
                model_name, 
                cache_dir=self.cache_dir,
                local_files_only=False
            )
            
            # Download model
            model = model_info['model_cls'].from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                local_files_only=False
            )
            
            logging.info(f"✅ Successfully downloaded {model_key}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to download {model_key}: {e}")
            return False
    
    def ensure_models_available(self, model_keys):
        """Ensure all required models are available, download if needed"""
        for model_key in model_keys:
            if not self.check_model_availability(model_key):
                logging.info(f"Model {model_key} not found locally, attempting download...")
                success = self.download_model(model_key)
                if not success:
                    logging.error(f"Failed to ensure {model_key} is available")
                    return False
            else:
                logging.info(f"Model {model_key} already available")
        
        return True
    
    def get_model_components(self, model_key):
        """Get tokenizer and model classes for a given model key"""
        if model_key not in self.models:
            raise ValueError(f"Unknown model: {model_key}")
            
        model_info = self.models[model_key]
        return (
            model_info['name'],
            model_info['tokenizer_cls'],
            model_info['model_cls']
        )

# Global model manager instance
model_manager = ModelManager()
