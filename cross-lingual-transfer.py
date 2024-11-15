import torch
from transformers import pipeline
import pandas as pd
import json
from sklearn.metrics import classification_report, accuracy_score, f1_score
from typing import List, Dict
import logging
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path

class CrossLingualZeroShot:
    def __init__(self, model_name: str = "xlm-roberta-large"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.classifier = None
        self.supported_languages = [
            #'en', 'es', 'de', 'ar', 'hi', 'vi', 'zh', 'th', 'tr', 'ru', 'el'
            'en', 'es', 'de', 'ar', 'ru'
        ]
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def _evaluate_predictions(self, predictions: List[str], data: List[Dict]) -> Dict:
        try:
            ground_truth = []
            for item in data:
                answer_text = item['answers'][0]['text'] if item['answers'] else "unknown"
                if any(word in answer_text.lower() for word in ['who', 'person', 'people', 'name']):
                    ground_truth.append('person')
                elif any(word in answer_text.lower() for word in ['where', 'city', 'country', 'place']):
                    ground_truth.append('location')
                elif any(word in answer_text.lower() for word in ['when', 'year', 'month', 'time']):
                    ground_truth.append('date')
                elif answer_text.replace('.','').replace(',','').isdigit():
                    ground_truth.append('number')
                elif any(word in answer_text.lower() for word in ['company', 'organization', 'institution']):
                    ground_truth.append('organization')
                else:
                    ground_truth.append('description')

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(ground_truth, predictions),
                'f1_macro': f1_score(ground_truth, predictions, average='macro'),
                'f1_weighted': f1_score(ground_truth, predictions, average='weighted'),
                'classification_report': classification_report(ground_truth, predictions, output_dict=True),
                'predictions': predictions,
                'ground_truth': ground_truth
            }
            
            self.logger.info(f"Evaluation completed. Accuracy: {metrics['accuracy']:.3f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in evaluation: {str(e)}")
            # Return basic metrics if evaluation fails
            return {
                'error': str(e),
                'predictions': predictions,
                'ground_truth': ['unknown'] * len(predictions)
            }

    def load_xquad_multilingual(self, base_path: str) -> Dict[str, List]:
        data_by_language = {}
        base_path = Path(base_path)
        
        # Create sample data if no files are found
        if not base_path.exists() or not any(base_path.glob("xquad.*")):
            self.logger.warning(f"No data files found in {base_path}. Creating sample data...")
            return self.create_sample_data()

        for lang in self.supported_languages:
            try:
                # Try different file patterns
                possible_files = [
                    base_path / f"xquad.{lang}.json"
                ]
                
                file_found = None
                for file_path in possible_files:
                    if file_path.exists():
                        file_found = file_path
                        break
                
                if file_found:
                    with open(file_found, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        processed_data = self._process_xquad_data(data)
                        data_by_language[lang] = processed_data
                        self.logger.info(f"Loaded {len(processed_data)} samples for {lang}")
                else:
                    self.logger.warning(f"No data file found for language: {lang}")
                    
            except Exception as e:
                self.logger.error(f"Error loading data for {lang}: {str(e)}")
                continue
        
        if not data_by_language:
            self.logger.warning("No data files could be loaded. Using sample data...")
            return self.create_sample_data()
            
        return data_by_language

    def create_sample_data(self) -> Dict[str, List]:
        sample_data = {
            'en': [
                {
                    'context': 'The quick brown fox jumps over the lazy dog.',
                    'question': 'What animal jumps over the dog?',
                    'answers': [{'text': 'fox'}],
                    'id': 'en_1'
                },
                {
                    'context': 'Paris is the capital of France.',
                    'question': 'What is the capital of France?',
                    'answers': [{'text': 'Paris'}],
                    'id': 'en_2'
                }
            ],
            'es': [
                {
                    'context': 'El rápido zorro marrón salta sobre el perro perezoso.',
                    'question': '¿Qué animal salta sobre el perro?',
                    'answers': [{'text': 'zorro'}],
                    'id': 'es_1'
                }
            ]
        }
        self.logger.info("Created sample data for testing")
        return sample_data

    def _process_xquad_data(self, raw_data: Dict) -> List[Dict]:
        processed_data = []
        try:
            if 'data' in raw_data:
                for article in raw_data['data']:
                    for paragraph in article['paragraphs']:
                        context = paragraph['context']
                        for qa in paragraph['qas']:
                            processed_data.append({
                                'context': context,
                                'question': qa['question'],
                                'answers': qa['answers'],
                                'id': qa['id']
                            })
            else:
                self.logger.warning("Unexpected data format. Using raw data as is.")
                processed_data = raw_data
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            return []
            
        return processed_data

    def perform_cross_lingual_classification(
        self,
        data_by_language: Dict[str, List],
        source_lang: str,
        target_langs: List[str],
        sample_size: int = 100
    ) -> Dict:
        if not self.classifier:
            self.load_model()

        results = {}
        
        # Verify source language data exists
        if source_lang not in data_by_language:
            raise ValueError(f"Source language '{source_lang}' not found in data. Available languages: {list(data_by_language.keys())}")
        
        # Prepare labels
        labels = ["person", "location", "organization", "date", "number", "description"]
        
        # Process source language
        self.logger.info(f"Processing source language: {source_lang}")
        source_data = data_by_language[source_lang][:sample_size]
        source_predictions = self._classify_batch(source_data, labels)
        results[source_lang] = self._evaluate_predictions(source_predictions, source_data)

        # Process target languages
        for target_lang in target_langs:
            if target_lang in data_by_language:
                self.logger.info(f"Processing target language: {target_lang}")
                target_data = data_by_language[target_lang][:sample_size]
                target_predictions = self._classify_batch(target_data, labels)
                results[target_lang] = self._evaluate_predictions(target_predictions, target_data)
            else:
                self.logger.warning(f"Target language '{target_lang}' not found in data. Skipping.")

        return results

    def _classify_batch(self, data: List[Dict], labels: List[str]) -> List[str]:
        predictions = []
        for item in tqdm(data):
            try:
                result = self.classifier(
                    item['question'],
                    labels,
                    hypothesis_template="This question is asking about {}."
                )
                predictions.append(result['labels'][0])
            except Exception as e:
                self.logger.error(f"Error classifying item: {str(e)}")
                predictions.append(labels[0])  # Default to first label on error
        return predictions

def main():
    classifier = CrossLingualZeroShot()
    
    source_lang = 'en'
    target_langs = ['es', 'de'] 
    
    try:
        data_path = "./xquad_data"  
        data_by_language = classifier.load_xquad_multilingual(data_path)
        
        if not data_by_language:
            raise ValueError("No data was loaded. Please check your data directory and file names.")
            
        results = classifier.perform_cross_lingual_classification(
            data_by_language,
            source_lang,
            target_langs,
            sample_size=100
        )
        
        for lang, lang_results in results.items():
            print(f"\nResults for {lang}:")
            if 'classification_report' in lang_results:
                metrics = lang_results['classification_report']
                if isinstance(metrics, dict) and 'weighted avg' in metrics:
                    print(f"Weighted F1-score: {metrics['weighted avg']['f1-score']:.3f}")
                else:
                    print("Metrics format unexpected")
            else:
                print("No classification report available")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Using sample data instead...")
        
        # Fall back to sample data
        data_by_language = classifier.create_sample_data()
        results = classifier.perform_cross_lingual_classification(
            data_by_language,
            source_lang,
            target_langs,
            sample_size=2
        )

if __name__ == "__main__":
    main()