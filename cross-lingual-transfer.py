import os
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import pipeline
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class CrossLingualZeroShot:
    def __init__(self, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", models: Optional[List[str]] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.available_models = {
            "mDeBERTa": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            "Multilingual_BERT": "emrecan/bert-base-multilingual-cased-snli_tr",
            "XLM-Roberta" : "xlm-roberta-large",
            "Info-XLM" : "microsoft/infoxlm-large",
            "BART" : "facebook/bart-large-mnli"
        }
        self.model_name = model_name
        self.models_to_use = models or list(self.available_models.values())
        self.classifier = None
        self.supported_languages = [
            'en', 'es', 'de', 'ar', 'ru', 'zh', 'tr'
        ]
        self.setup_logging()

    def load_model(self):
        try:
            for model_path in self.models_to_use:
                self.logger.info(f"Loading model: {model_path}")
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model=model_path,
                    device=0 if torch.cuda.is_available() else -1
                )
            
            self.logger.info(f"Loaded {len(self.classifier)} models successfully")
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
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
                'precision': precision_score(ground_truth, predictions, average='weighted'),
                'recall': recall_score(ground_truth,predictions, average='weighted'),
                'classification_report': classification_report(ground_truth, predictions, output_dict=True),
                'predictions': predictions,
                'ground_truth': ground_truth
            }
            
            self.logger.info(f"Evaluation completed. Accuracy: {metrics['accuracy']:.3f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in evaluation: {str(e)}")
            return {
                'error': str(e),
                'predictions': predictions,
                'ground_truth': ['unknown'] * len(predictions)
            }

    def load_xquad_multilingual(self, base_path: str) -> Dict[str, List]:
        data_by_language = {}
        base_path = Path(base_path)
        
        # Create sample data if no files are found
        if not base_path.exists() or not any(base_path.glob("xquad*")):
            self.logger.warning(f"No data files found in {base_path}. Creating sample data...")
            return self.create_sample_data()

        for lang in self.supported_languages:
            try:
                possible_files = [
                    base_path / f"xquad-{lang}.json"
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
                    'context': 'Neil Armstrong was the first person to walk on the moon in 1969.',
                    'question': 'Who was the first person to walk on the moon?',
                    'answers': [{'text': 'Neil Armstrong'}],
                    'id': 'en_2'
                },
                {
                    'context': 'The Nile is the longest river in the world, flowing through 11 countries in Africa.',
                    'question': 'What is the longest river in the world?',
                    'answers': [{'text': 'The Nile'}],
                    'id': 'en_3'
                },
                {
                    'context': 'Mount Everest is the tallest mountain in the world, standing at 8,848 meters.',
                    'question': 'What is the tallest mountain in the world?',
                    'answers': [{'text': 'Mount Everest'}],
                    'id': 'en_4'
                },
                {
                    'context': 'The Great Wall of China is one of the most famous landmarks in the world.',
                    'question': 'What is one of the most famous landmarks in China?',
                    'answers': [{'text': 'The Great Wall of China'}],
                    'id': 'en_5'
                },
                {
                    'context': 'The capital city of France is Paris, known for its art, fashion, and culture.',
                    'question': 'What is the capital city of France?',
                    'answers': [{'text': 'Paris'}],
                    'id': 'en_6'
                },
                {
                    'context': 'Shakespeare wrote the famous play Romeo and Juliet in the late 16th century.',
                    'question': 'Who wrote Romeo and Juliet?',
                    'answers': [{'text': 'Shakespeare'}],
                    'id': 'en_7'
                },
                {
                    'context': 'The Pacific Ocean is the largest and deepest ocean on Earth.',
                    'question': 'What is the largest ocean on Earth?',
                    'answers': [{'text': 'The Pacific Ocean'}],
                    'id': 'en_8'
                },
                {
                    'context': 'Thomas Edison invented the electric light bulb in 1879.',
                    'question': 'Who invented the electric light bulb?',
                    'answers': [{'text': 'Thomas Edison'}],
                    'id': 'en_9'
                },
                {
                    'context': 'The Mona Lisa, painted by Leonardo da Vinci, is one of the most famous artworks in history.',
                    'question': 'Who painted the Mona Lisa?',
                    'answers': [{'text': 'Leonardo da Vinci'}],
                    'id': 'en_10'
                },
                {
                    'context': 'The Sahara Desert is the largest hot desert in the world, located in Africa.',
                    'question': 'What is the largest hot desert in the world?',
                    'answers': [{'text': 'The Sahara Desert'}],
                    'id': 'en_11'
                }
            ],
            'es': [
                {
                    'context': 'El rápido zorro marrón salta sobre el perro perezoso.',
                    'question': '¿Qué animal salta sobre el perro?',
                    'answers': [{'text': 'zorro'}],
                    'id': 'es_1'
                },
                {
                    "context": "Cristóbal Colón zarpó en 1492 desde España y llegó al continente americano.",
                    "question": "¿En qué año llegó Cristóbal Colón a América?",
                    "answers": [{"text": "1492"}],
                    "id": "es_2"
                },
                {
                    "context": "La Torre Eiffel, uno de los monumentos más famosos del mundo, está ubicada en París.",
                    "question": "¿Dónde está ubicada la Torre Eiffel?",
                    "answers": [{"text": "París"}],
                    "id": "es_3"
                },
                {
                    "context": "El Amazonas es el río más largo del mundo y atraviesa varios países de América del Sur.",
                    "question": "¿Cuál es el río más largo del mundo?",
                    "answers": [{"text": "El Amazonas"}],
                    "id": "es_4"
                },
                {
                    "context": "La capital de Argentina es Buenos Aires, conocida por su rica cultura y tango.",
                    "question": "¿Cuál es la capital de Argentina?",
                    "answers": [{"text": "Buenos Aires"}],
                    "id": "es_5"
                },
                {
                    "context": "Miguel de Cervantes escribió Don Quijote, una de las obras más importantes de la literatura española.",
                    "question": "¿Quién escribió Don Quijote?",
                    "answers": [{"text": "Miguel de Cervantes"}],
                    "id": "es_6"
                },
                {
                    "context": "El Everest, con una altura de 8,848 metros, es la montaña más alta del mundo.",
                    "question": "¿Cuál es la montaña más alta del mundo?",
                    "answers": [{"text": "El Everest"}],
                    "id": "es_7"
                },
                {
                    "context": "El Real Madrid es uno de los equipos de fútbol más exitosos de Europa.",
                    "question": "¿Qué equipo de fútbol es uno de los más exitosos de Europa?",
                    "answers": [{"text": "Real Madrid"}],
                    "id": "es_8"
                },
                {
                    "context": "La Revolución Francesa comenzó en 1789 y marcó un cambio significativo en la historia de Europa.",
                    "question": "¿En qué año comenzó la Revolución Francesa?",
                    "answers": [{"text": "1789"}],
                    "id": "es_9"
                },
                {
                    "context": "Pablo Picasso fue un pintor y escultor español conocido por cofundar el movimiento cubista.",
                    "question": "¿Quién cofundó el movimiento cubista?",
                    "answers": [{"text": "Pablo Picasso"}],
                    "id": "es_10"
                },
                {
                    "context": "La Ópera de Sídney es uno de los edificios más emblemáticos de Australia.",
                    "question": "¿Cuál es uno de los edificios más emblemáticos de Australia?",
                    "answers": [{"text": "La Ópera de Sídney"}],
                    "id": "es_11"
                },
                {
                    'context': 'La Alhambra es un palacio histórico situado en Granada, España.',
                    'question': '¿Dónde se encuentra la Alhambra?',
                    'answers': [{'text': 'en Granada, España'}],
                    'id': 'es_12'
                },
            ],
            'ru': [
                {
                "context": "Москва — столица России и крупнейший город страны.",
                "question": "Какой город является столицей России?",
                "answers": [{"text": "Москва"}],
                "id": "ru_1"
                },
                {
                "context": "Лев Толстой — известный русский писатель, автор «Войны и мира».",
                "question": "Кто написал роман «Война и мир»?",
                "answers": [{"text": "Лев Толстой"}],
                "id": "ru_2"
                },
                {
                "context": "Байкал — самое глубокое озеро в мире и расположено в России.",
                "question": "Как называется самое глубокое озеро в мире?",
                "answers": [{"text": "Байкал"}],
                "id": "ru_3"
                },
                {
                "context": "Юрий Гагарин стал первым человеком, полетевшим в космос в 1961 году.",
                "question": "Кто был первым человеком в космосе?",
                "answers": [{"text": "Юрий Гагарин"}],
                "id": "ru_4"
                },
                {
                "context": "Санкт-Петербург — второй по величине город в России, известный своими дворцами и каналами.",
                "question": "Какой город в России известен своими дворцами и каналами?",
                "answers": [{"text": "Санкт-Петербург"}],
                "id": "ru_5"
                },
                {
                "context": "Красная площадь — это главная площадь Москвы и всей России.",
                "question": "Как называется главная площадь Москвы?",
                "answers": [{"text": "Красная площадь"}],
                "id": "ru_6"
                },
                {
                "context": "Фёдор Достоевский написал роман «Преступление и наказание», который считается шедевром мировой литературы.",
                "question": "Кто написал роман «Преступление и наказание»?",
                "answers": [{"text": "Фёдор Достоевский"}],
                "id": "ru_7"
                },
                {
                "context": "Транссибирская магистраль — самая длинная железная дорога в мире, соединяющая Москву и Владивосток.",
                "question": "Как называется самая длинная железная дорога в мире?",
                "answers": [{"text": "Транссибирская магистраль"}],
                "id": "ru_8"
                },
                {
                "context": "Эрмитаж в Санкт-Петербурге — один из крупнейших и старейших музеев мира.",
                "question": "Какой музей в Санкт-Петербурге является одним из крупнейших в мире?",
                "answers": [{"text": "Эрмитаж"}],
                "id": "ru_9"
                },
                {
                "context": "Кремль в Москве — это резиденция президента России и символ страны.",
                "question": "Что является резиденцией президента России?",
                "answers": [{"text": "Кремль"}],
                "id": "ru_10"
                },
                {
                    'context': 'Московский Кремль расположен в самом сердце Москвы.',
                    'question': 'Где расположен Московский Кремль?',
                    'answers': [{'text': 'в самом сердце Москвы'}],
                    'id': 'ru_11'
                }
            ],
            'ar': [
                {
                "context": "الهرم الأكبر في الجيزة هو أحد عجائب الدنيا السبع القديمة.",
                "question": "ما هو أحد عجائب الدنيا السبع القديمة في مصر؟",
                "answers": [{"text": "الهرم الأكبر"}],
                "id": "ar_1"
                },
                {
                "context": "البتراء في الأردن مشهورة بمعمارها المنحوت في الصخور.",
                "question": "ما هي المدينة الأردنية المشهورة بمعمارها المنحوت في الصخور؟",
                "answers": [{"text": "البتراء"}],
                "id": "ar_2"
                },
                {
                "context": "قناة السويس تربط البحر الأحمر بالبحر الأبيض المتوسط.",
                "question": "ما هي القناة التي تربط البحر الأحمر بالبحر الأبيض المتوسط؟",
                "answers": [{"text": "قناة السويس"}],
                "id": "ar_3"
                },
                {
                "context": "ابن الهيثم كان عالماً مسلماً بارزاً في مجالات البصريات والرياضيات.",
                "question": "من هو العالم المسلم البارز في مجال البصريات؟",
                "answers": [{"text": "ابن الهيثم"}],
                "id": "ar_4"
                },
                {
                "context": "برج خليفة في دبي هو أطول مبنى في العالم.",
                "question": "ما هو أطول مبنى في العالم؟",
                "answers": [{"text": "برج خليفة"}],
                "id": "ar_5"
                },
                {
                "context": "صلاح الدين الأيوبي كان قائداً عسكرياً إسلامياً شهيراً.",
                "question": "من هو القائد الإسلامي الشهير الذي حرر القدس؟",
                "answers": [{"text": "صلاح الدين الأيوبي"}],
                "id": "ar_6"
                },
                {
                "context": "نهر النيل هو أطول نهر في العالم ويمر بعدة دول أفريقية.",
                "question": "ما هو أطول نهر في العالم؟",
                "answers": [{"text": "نهر النيل"}],
                "id": "ar_7"
                },
                {
                "context": "الكعبة في مكة المكرمة هي قبلة المسلمين.",
                "question": "ما هي القبلة التي يتجه إليها المسلمون في صلاتهم؟",
                "answers": [{"text": "الكعبة"}],
                "id": "ar_8"
                },
                {
                "context": "اللغة العربية هي واحدة من أكثر اللغات تحدثاً في العالم.",
                "question": "ما هي اللغة التي تُعتبر من أكثر اللغات تحدثاً في العالم؟",
                "answers": [{"text": "اللغة العربية"}],
                "id": "ar_9"
                },
                {
                "context": "الأهرامات في مصر تعد من أهم معالم السياحة في العالم.",
                "question": "ما هي المعالم التي تعد من أهم معالم السياحة في مصر؟",
                "answers": [{"text": "الأهرامات"}],
                "id": "ar_10"
                },
                {
                    'context': 'تقع الأهرامات في مصر.',
                    'question': 'أين تقع الأهرامات؟',
                    'answers': [{'text': 'في مصر'}],
                    'id': 'ar_11'
                }
            ],
            "de": [
                {
                "context": "Albert Einstein entwickelte die allgemeine Relativitätstheorie, die das Verständnis von Raum, Zeit und Gravitation revolutionierte.",
                "question": "Wer entwickelte die allgemeine Relativitätstheorie?",
                "answers": [{"text": "Albert Einstein"}],
                "id": "de_2"
                },
                {
                "context": "Die Berliner Mauer fiel am 9. November 1989 und markierte das Ende der Teilung Deutschlands.",
                "question": "Wann fiel die Berliner Mauer?",
                "answers": [{"text": "9. November 1989"}],
                "id": "de_3"
                },
                {
                "context": "Die Donau ist der zweitlängste Fluss Europas und fließt durch zehn Länder.",
                "question": "Wie viele Länder durchfließt die Donau?",
                "answers": [{"text": "zehn"}],
                "id": "de_4"
                },
                {
                "context": "Die Bundeskanzlerin von Deutschland im Jahr 2021 war Angela Merkel.",
                "question": "Wer war die Bundeskanzlerin Deutschlands im Jahr 2021?",
                "answers": [{"text": "Angela Merkel"}],
                "id": "de_5"
                },
                {
                "context": "Das Brandenburger Tor ist eines der bekanntesten Wahrzeichen Berlins.",
                "question": "Was ist eines der bekanntesten Wahrzeichen Berlins?",
                "answers": [{"text": "Das Brandenburger Tor"}],
                "id": "de_6"
                },
                {
                "context": "Johann Wolfgang von Goethe schrieb Faust, eines der bedeutendsten Werke der deutschen Literatur.",
                "question": "Wer schrieb Faust?",
                "answers": [{"text": "Johann Wolfgang von Goethe"}],
                "id": "de_7"
                },
                {
                "context": "Der Schwarzwald ist eine Region im Südwesten Deutschlands, bekannt für seine dichten Wälder und Kuckucksuhren.",
                "question": "Wofür ist der Schwarzwald bekannt?",
                "answers": [{"text": "dichte Wälder und Kuckucksuhren"}],
                "id": "de_8"
                },
                {
                "context": "Die deutsche Wiedervereinigung fand am 3. Oktober 1990 statt.",
                "question": "Wann fand die deutsche Wiedervereinigung statt?",
                "answers": [{"text": "3. Oktober 1990"}],
                "id": "de_9"
                },
                {
                "context": "Die Zugspitze ist der höchste Berg Deutschlands mit einer Höhe von 2.962 Metern.",
                "question": "Wie hoch ist die Zugspitze?",
                "answers": [{"text": "2.962 Meter"}],
                "id": "de_10"
                },
                {
                "context": "Die erste Buchdruckmaschine wurde von Johannes Gutenberg im 15. Jahrhundert entwickelt.",
                "question": "Wer entwickelte die erste Buchdruckmaschine?",
                "answers": [{"text": "Johannes Gutenberg"}],
                "id": "de_11"
                }
            ],
            "zh": [
                {
                "context": "长城是中国最著名的历史建筑之一，全长超过2万公里。",
                "question": "中国最著名的历史建筑是什么？",
                "answers": [{"text": "长城"}],
                "id": "zh_1"
                },
                {
                "context": "孔子是中国历史上著名的哲学家和教育家，被称为至圣先师。",
                "question": "谁被称为至圣先师？",
                "answers": [{"text": "孔子"}],
                "id": "zh_2"
                },
                {
                "context": "长江是中国最长的河流，也是世界第三长的河流。",
                "question": "中国最长的河流是什么？",
                "answers": [{"text": "长江"}],
                "id": "zh_3"
                },
                {
                "context": "故宫位于北京市中心，是明清两代的皇家宫殿。",
                "question": "故宫位于哪里？",
                "answers": [{"text": "北京市中心"}],
                "id": "zh_4"
                },
                {
                "context": "中国的国庆节是每年的10月1日。",
                "question": "中国的国庆节是哪一天？",
                "answers": [{"text": "10月1日"}],
                "id": "zh_5"
                },
                {
                "context": "四大发明是中国古代对世界文明的重要贡献，包括造纸术、印刷术、指南针和火药。",
                "question": "四大发明包括什么？",
                "answers": [{"text": "造纸术、印刷术、指南针和火药"}],
                "id": "zh_6"
                },
                {
                "context": "兵马俑被认为是秦始皇陵的陪葬品，是世界八大奇迹之一。",
                "question": "兵马俑属于哪个历史人物的陪葬品？",
                "answers": [{"text": "秦始皇"}],
                "id": "zh_7"
                },
                {
                "context": "北京奥运会于2008年在中国举行，是一场盛大的国际体育盛会。",
                "question": "2008年奥运会在哪个国家举行？",
                "answers": [{"text": "中国"}],
                "id": "zh_8"
                },
                {
                "context": "熊猫是中国的国宝，主要栖息在四川、陕西和甘肃的山区。",
                "question": "熊猫主要栖息在哪里？",
                "answers": [{"text": "四川、陕西和甘肃"}],
                "id": "zh_9"
                },
                {
                "context": "张艺谋是一位著名的中国导演，他执导了许多经典的电影。",
                "question": "谁是著名的中国导演，执导了许多经典的电影？",
                "answers": [{"text": "张艺谋"}],
                "id": "zh_10"
                }
            ],
            "tr": [
                {
                "context": "Ankara, Türkiye'nin başkenti ve ikinci en kalabalık şehridir.",
                "question": "Türkiye'nin başkenti neresidir?",
                "answers": [{"text": "Ankara"}],
                "id": "tr_1"
                },
                {
                "context": "Türkiye'nin en uzun nehri olan Kızılırmak, 1,355 kilometre uzunluğundadır.",
                "question": "Türkiye'nin en uzun nehri hangisidir?",
                "answers": [{"text": "Kızılırmak"}],
                "id": "tr_2"
                },
                {
                "context": "Atatürk, Türkiye Cumhuriyeti'nin kurucusudur ve ilk cumhurbaşkanıdır.",
                "question": "Türkiye Cumhuriyeti'ni kim kurmuştur?",
                "answers": [{"text": "Atatürk"}],
                "id": "tr_3"
                },
                {
                "context": "Pamukkale, travertenleri ve termal sularıyla ünlü bir doğa harikasıdır.",
                "question": "Pamukkale neden ünlüdür?",
                "answers": [{"text": "travertenleri ve termal sularıyla"}],
                "id": "tr_4"
                },
                {
                "context": "Türkiye'nin en büyük gölü olan Van Gölü, Doğu Anadolu Bölgesi'nde bulunur.",
                "question": "Türkiye'nin en büyük gölü hangisidir?",
                "answers": [{"text": "Van Gölü"}],
                "id": "tr_5"
                },
                {
                "context": "İstanbul Boğazı, Asya ve Avrupa kıtalarını birbirine bağlar.",
                "question": "Asya ve Avrupa kıtalarını hangi boğaz birleştirir?",
                "answers": [{"text": "İstanbul Boğazı"}],
                "id": "tr_6"
                },
                {
                "context": "Kapadokya, peri bacaları ve balon turlarıyla dünya çapında ünlüdür.",
                "question": "Kapadokya neden ünlüdür?",
                "answers": [{"text": "peri bacaları ve balon turlarıyla"}],
                "id": "tr_7"
                },
                {
                "context": "Efes Antik Kenti, UNESCO Dünya Mirası Listesi'nde yer alır ve Türkiye'nin önemli bir turistik alanıdır.",
                "question": "Efes Antik Kenti nerede listelenmiştir?",
                "answers": [{"text": "UNESCO Dünya Mirası Listesi"}],
                "id": "tr_8"
                },
                {
                "context": "Nemrut Dağı, devasa heykelleri ve tarihi kalıntılarıyla ünlüdür.",
                "question": "Nemrut Dağı neden ünlüdür?",
                "answers": [{"text": "devasa heykelleri ve tarihi kalıntılarıyla"}],
                "id": "tr_9"
                },
                {
                "context": "Mevlana, Konya'da yaşamış bir mutasavvıf ve Mevlevilik tarikatının kurucusudur.",
                "question": "Mevlana kimdir?",
                "answers": [{"text": "mutasavvıf ve Mevlevilik tarikatının kurucusu"}],
                "id": "tr_10"
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
        labels = ["person", "location", "organization", "date", "number", "description" , "sports", "politics", 
                  "technology", "science", "health", "history", "geography", "arts", "entertainment", "culture", "society", "economy", "business"]
        
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
                predictions.append(labels[0])  
        return predictions

def main():
    classifier = CrossLingualZeroShot()
    
    source_lang = 'en'
    target_langs = ['es','de','ru','ar','zh','tr']  
    
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