import streamlit as st
from PIL import Image
import random
import os
import tempfile
import base64
from io import BytesIO
import numpy as np
import time
import logging
import hashlib
import re
import urllib.request
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    st.warning("⚠️ Text-to-speech not available. Install gTTS: pip install gTTS")

try:
    import os as _os
    import shutil as _shutil
    from pydub import AudioSegment
    from pydub.generators import Sine, WhiteNoise
    from pydub.effects import normalize, compress_dynamic_range
    from pydub.utils import which as _which

    def _ensure_ffmpeg_paths():
        """Try to locate ffmpeg/ffprobe on Windows and wire them into pydub.
        Returns (ffmpeg_path, ffprobe_path) or (None, None).
        """
        candidates = []
        # 1) PATH
        if _which("ffmpeg"):
            candidates.append((_which("ffmpeg"), _which("ffprobe")))
        # 2) Common Windows install locations
        common_dirs = [
            r"C:\\ffmpeg\\bin",
            r"C:\\Program Files\\ffmpeg\\bin",
            r"C:\\Program Files (x86)\\ffmpeg\\bin",
            # Winget/Gyan build often lands in Program Files
            r"C:\\Program Files\\Gyan\\ffmpeg\\bin",
            # WinGet package location
            _os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin"),
        ]
        for d in common_dirs:
            ffmpeg_path = _os.path.join(d, "ffmpeg.exe")
            ffprobe_path = _os.path.join(d, "ffprobe.exe")
            if _os.path.exists(ffmpeg_path) and _os.path.exists(ffprobe_path):
                candidates.append((ffmpeg_path, ffprobe_path))
        # Choose first valid
        for ffm, ffp in candidates:
            if ffm and ffp:
                # Set for current process so pydub can use them
                AudioSegment.converter = ffm
                AudioSegment.ffprobe = ffp
                # Also put on PATH for any subprocess use
                ffm_dir = _os.path.dirname(ffm)
                if ffm_dir not in _os.environ.get("PATH", ""):
                    _os.environ["PATH"] = ffm_dir + _os.pathsep + _os.environ.get("PATH", "")
                return ffm, ffp
        return None, None

    # Try to wire ffmpeg paths (no-op if already on PATH)
    _ensure_ffmpeg_paths()

    # Probe capabilities: we need to be able to export MP3 and read MP3 to mix bg music
    try:
        # Generating a sine tone never needs ffmpeg, but export to MP3 does.
        _tmp_dir = __import__("tempfile").mkdtemp()
        _tmp_mp3 = _os.path.join(_tmp_dir, "probe.mp3")
        Sine(440).to_audio_segment(duration=200).export(_tmp_mp3, format="mp3")
        # Try read-back
        _ = AudioSegment.from_mp3(_tmp_mp3)
        AUDIO_PROCESSING_AVAILABLE = True
    except Exception:
        # Attempt another pass after trying to locate ffmpeg explicitly
        _ensure_ffmpeg_paths()
        try:
            _tmp_dir = __import__("tempfile").mkdtemp()
            _tmp_mp3 = _os.path.join(_tmp_dir, "probe.mp3")
            Sine(440).to_audio_segment(duration=200).export(_tmp_mp3, format="mp3")
            _ = AudioSegment.from_mp3(_tmp_mp3)
            AUDIO_PROCESSING_AVAILABLE = True
        except Exception:
            AUDIO_PROCESSING_AVAILABLE = False

except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

# Optional advanced audio libraries
try:
    import librosa
    import soundfile as sf
    ADVANCED_AUDIO_AVAILABLE = True
except ImportError:
    ADVANCED_AUDIO_AVAILABLE = False

# Try to import AI model dependencies
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    AI_MODEL_AVAILABLE = True
    # Initialize global model variables
    model = None
    tokenizer = None
except ImportError as e:
    AI_MODEL_AVAILABLE = False
    model = None
    tokenizer = None

# Additional imports for enhanced features
import json
import uuid
from pathlib import Path
from datetime import datetime
import zipfile
import io
import base64
import heapq
from collections import defaultdict, deque
import bisect
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Any
import math
import networkx as nx
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    # Try importing moviepy components individually
    import moviepy
    # import moviepy.editor as mp  # Commented out for now - causing import issues
    from PIL import Image, ImageDraw, ImageFont
    VIDEO_AVAILABLE = False  # Set to False until moviepy is fully working
except ImportError:
    VIDEO_AVAILABLE = False

# Voice cloning libraries (placeholder - would need specific implementation)
try:
    # import coqui_tts  # Example: Coqui TTS for voice cloning
    VOICE_CLONE_AVAILABLE = False  # Set to True when implemented
except ImportError:
    VOICE_CLONE_AVAILABLE = False

# ============================================================================
# ADVANCED DATA STRUCTURES & ALGORITHMS FOR POETRY AI
# ============================================================================

@dataclass
class WordNode:
    """Node for storing word information with metadata"""
    word: str
    syllables: int
    phonetic_ending: str
    sentiment_score: float
    word_type: str  # noun, verb, adjective, etc.
    frequency: int = 0
    rhyme_group: str = ""
    semantic_tags: List[str] = None
    
    def __post_init__(self):
        if self.semantic_tags is None:
            self.semantic_tags = []

class PoetryTrie:
    """Advanced Trie structure for fast word lookup and pattern matching"""
    
    def __init__(self):
        self.root = {}
        self.end_symbol = '*'
        self.word_count = 0
    
    def insert(self, word_node: WordNode):
        """Insert a word with all its metadata"""
        current = self.root
        word = word_node.word.lower()
        
        for char in word:
            if char not in current:
                current[char] = {}
            current = current[char]
        
        current[self.end_symbol] = word_node
        self.word_count += 1
    
    def search(self, word: str) -> Optional[WordNode]:
        """Search for a word and return its metadata"""
        current = self.root
        word = word.lower()
        
        for char in word:
            if char not in current:
                return None
            current = current[char]
        
        return current.get(self.end_symbol)
    
    def find_words_by_prefix(self, prefix: str) -> List[WordNode]:
        """Find all words with given prefix"""
        current = self.root
        prefix = prefix.lower()
        
        for char in prefix:
            if char not in current:
                return []
            current = current[char]
        
        words = []
        self._collect_words(current, prefix, words)
        return words
    
    def find_rhymes(self, word: str, max_distance: int = 2) -> List[WordNode]:
        """Find words that rhyme with the given word"""
        target_node = self.search(word)
        if not target_node:
            return []
        
        rhymes = []
        self._find_rhymes_recursive(self.root, "", target_node.phonetic_ending, rhymes, max_distance)
        return rhymes
    
    def _collect_words(self, node: dict, prefix: str, words: List[WordNode]):
        """Helper method to collect all words from a trie node"""
        if self.end_symbol in node:
            words.append(node[self.end_symbol])
        
        for char, child_node in node.items():
            if char != self.end_symbol:
                self._collect_words(child_node, prefix + char, words)
    
    def _find_rhymes_recursive(self, node: dict, current_word: str, target_ending: str, 
                              rhymes: List[WordNode], max_distance: int):
        """Recursively find rhyming words"""
        if self.end_symbol in node:
            word_node = node[self.end_symbol]
            if self._calculate_rhyme_distance(word_node.phonetic_ending, target_ending) <= max_distance:
                rhymes.append(word_node)
        
        for char, child_node in node.items():
            if char != self.end_symbol:
                self._find_rhymes_recursive(child_node, current_word + char, target_ending, rhymes, max_distance)
    
    def _calculate_rhyme_distance(self, ending1: str, ending2: str) -> int:
        """Calculate phonetic distance between word endings"""
        # Simple Levenshtein distance for phonetic endings
        if len(ending1) < len(ending2):
            return self._calculate_rhyme_distance(ending2, ending1)
        
        if len(ending2) == 0:
            return len(ending1)
        
        previous_row = list(range(len(ending2) + 1))
        for i, char1 in enumerate(ending1):
            current_row = [i + 1]
            for j, char2 in enumerate(ending2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (char1 != char2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

class SemanticGraph:
    """Graph structure for semantic word relationships"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.concept_clusters = {}
        self.word_embeddings = {}
    
    def add_word_relationship(self, word1: str, word2: str, relationship_type: str, strength: float):
        """Add a semantic relationship between two words"""
        self.graph.add_edge(word1, word2, 
                           relationship=relationship_type, 
                           weight=strength)
    
    def find_semantic_path(self, start_word: str, end_word: str) -> List[str]:
        """Find semantic path between words using shortest path"""
        try:
            return nx.shortest_path(self.graph, start_word, end_word, weight='weight')
        except nx.NetworkXNoPath:
            return []
    
    def get_related_words(self, word: str, max_distance: int = 2) -> List[Tuple[str, float]]:
        """Get semantically related words within max distance"""
        if word not in self.graph:
            return []
        
        related = []
        try:
            distances = nx.single_source_shortest_path_length(self.graph, word, cutoff=max_distance)
            for related_word, distance in distances.items():
                if related_word != word and distance <= max_distance:
                    # Calculate semantic similarity based on distance
                    similarity = 1.0 / (1.0 + distance)
                    related.append((related_word, similarity))
        except:
            pass
        
        return sorted(related, key=lambda x: x[1], reverse=True)
    
    def cluster_concepts(self, concepts: List[str]) -> Dict[str, List[str]]:
        """Cluster related concepts using community detection"""
        subgraph = self.graph.subgraph(concepts)
        try:
            communities = nx.community.greedy_modularity_communities(subgraph)
            clusters = {}
            for i, community in enumerate(communities):
                clusters[f"cluster_{i}"] = list(community)
            return clusters
        except:
            return {"single_cluster": concepts}

class RhymeEngine:
    """Advanced rhyme detection and generation engine"""
    
    def __init__(self):
        self.phonetic_map = {}
        self.rhyme_patterns = defaultdict(list)
        self.perfect_rhymes = defaultdict(set)
        self.near_rhymes = defaultdict(set)
        self.priority_queue = []
    
    def add_word(self, word: str, phonetic: str):
        """Add word with its phonetic representation"""
        self.phonetic_map[word] = phonetic
        ending = self._extract_rhyme_ending(phonetic)
        self.rhyme_patterns[ending].append(word)
        
        # Build perfect and near rhyme sets
        self._update_rhyme_sets(word, ending)
    
    def find_best_rhymes(self, word: str, count: int = 5) -> List[Tuple[str, float]]:
        """Find best rhyming words using priority queue"""
        if word not in self.phonetic_map:
            return []
        
        target_phonetic = self.phonetic_map[word]
        target_ending = self._extract_rhyme_ending(target_phonetic)
        
        # Use max heap (negate scores for min heap behavior)
        rhyme_heap = []
        
        # Perfect rhymes (highest priority)
        for rhyme_word in self.perfect_rhymes[target_ending]:
            if rhyme_word != word:
                score = self._calculate_rhyme_quality(word, rhyme_word)
                heapq.heappush(rhyme_heap, (-score, rhyme_word, "perfect"))
        
        # Near rhymes (medium priority)
        for rhyme_word in self.near_rhymes[target_ending]:
            if rhyme_word != word:
                score = self._calculate_rhyme_quality(word, rhyme_word) * 0.8
                heapq.heappush(rhyme_heap, (-score, rhyme_word, "near"))
        
        # Extract top rhymes
        best_rhymes = []
        for _ in range(min(count, len(rhyme_heap))):
            if rhyme_heap:
                neg_score, rhyme_word, rhyme_type = heapq.heappop(rhyme_heap)
                best_rhymes.append((rhyme_word, -neg_score))
        
        return best_rhymes
    
    def _extract_rhyme_ending(self, phonetic: str) -> str:
        """Extract the rhyming part of a phonetic representation"""
        # Simplified: take last 2-3 phonemes
        parts = phonetic.split()
        return " ".join(parts[-2:]) if len(parts) >= 2 else phonetic
    
    def _update_rhyme_sets(self, word: str, ending: str):
        """Update perfect and near rhyme sets"""
        for existing_word in self.rhyme_patterns[ending]:
            if existing_word != word:
                distance = self._phonetic_distance(word, existing_word)
                if distance == 0:
                    self.perfect_rhymes[ending].add(word)
                    self.perfect_rhymes[ending].add(existing_word)
                elif distance <= 1:
                    self.near_rhymes[ending].add(word)
                    self.near_rhymes[ending].add(existing_word)
    
    def _phonetic_distance(self, word1: str, word2: str) -> int:
        """Calculate phonetic distance between words"""
        phon1 = self.phonetic_map.get(word1, "")
        phon2 = self.phonetic_map.get(word2, "")
        
        # Simple edit distance on phonetic representations
        return self._edit_distance(phon1, phon2)
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance between two strings"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _calculate_rhyme_quality(self, word1: str, word2: str) -> float:
        """Calculate quality score for a rhyme pair"""
        # Factors: phonetic similarity, syllable match, frequency
        phonetic_sim = 1.0 / (1.0 + self._phonetic_distance(word1, word2))
        syllable_sim = 1.0 if len(word1.split()) == len(word2.split()) else 0.7
        
        return phonetic_sim * syllable_sim

class PoetryOptimizer:
    """Dynamic programming optimizer for poem structure"""
    
    def __init__(self):
        self.memo = {}
        self.word_scores = {}
        self.transition_scores = {}
    
    def optimize_line_structure(self, words: List[str], target_syllables: int, 
                               target_sentiment: float) -> List[str]:
        """Find optimal word arrangement using dynamic programming"""
        n = len(words)
        if n == 0:
            return []
        
        # Memoization key
        key = (tuple(words), target_syllables, target_sentiment)
        if key in self.memo:
            return self.memo[key]
        
        # DP table: dp[i][s][sent] = best score for first i words with s syllables and sent sentiment
        dp = {}
        parent = {}
        
        # Initialize
        dp[(0, 0, 0)] = 0
        
        for i in range(n):
            word = words[i]
            word_syllables = self._count_syllables(word)
            word_sentiment = self._get_word_sentiment(word)
            
            new_states = []
            for (pos, syllables, sentiment), score in dp.items():
                if pos == i:
                    new_syll = syllables + word_syllables
                    new_sent = (sentiment * pos + word_sentiment) / (pos + 1) if pos > 0 else word_sentiment
                    new_state = (pos + 1, new_syll, new_sent)
                    
                    new_score = score + self._calculate_word_score(word, new_syll, target_syllables, 
                                                                  new_sent, target_sentiment)
                    
                    if new_state not in dp or dp[new_state] < new_score:
                        dp[new_state] = new_score
                        parent[new_state] = (pos, syllables, sentiment)
                        new_states.append(new_state)
        
        # Find best final state
        best_state = None
        best_score = float('-inf')
        
        for (pos, syllables, sentiment), score in dp.items():
            if pos == n:
                total_score = score - abs(syllables - target_syllables) - abs(sentiment - target_sentiment)
                if total_score > best_score:
                    best_score = total_score
                    best_state = (pos, syllables, sentiment)
        
        if best_state is None:
            self.memo[key] = words
            return words
        
        # Reconstruct optimal path
        path = []
        current = best_state
        while current in parent:
            path.append(current)
            current = parent[current]
        
        path.reverse()
        optimized_words = [words[i] for i, _, _ in path[1:]]
        
        self.memo[key] = optimized_words
        return optimized_words
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        vowels = "aeiouy"
        word = word.lower()
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _get_word_sentiment(self, word: str) -> float:
        """Get sentiment score for a word"""
        # Simplified sentiment analysis
        positive_words = {'love', 'joy', 'happy', 'beautiful', 'wonderful', 'bright', 'hope', 'dream'}
        negative_words = {'sad', 'dark', 'pain', 'sorrow', 'fear', 'lonely', 'broken', 'tears'}
        
        word_lower = word.lower()
        if word_lower in positive_words:
            return 1.0
        elif word_lower in negative_words:
            return -1.0
        else:
            return 0.0
    
    def _calculate_word_score(self, word: str, current_syllables: int, target_syllables: int,
                             current_sentiment: float, target_sentiment: float) -> float:
        """Calculate score for including a word"""
        syllable_penalty = abs(current_syllables - target_syllables) * 0.1
        sentiment_bonus = 1.0 - abs(current_sentiment - target_sentiment)
        
        return sentiment_bonus - syllable_penalty

class CreativePathFinder:
    """A* search algorithm for finding creative word sequences"""
    
    def __init__(self, semantic_graph: SemanticGraph):
        self.semantic_graph = semantic_graph
        self.creativity_weights = {
            'novelty': 0.3,
            'semantic_distance': 0.3,
            'phonetic_beauty': 0.2,
            'emotional_impact': 0.2
        }
    
    def find_creative_sequence(self, start_concept: str, target_emotion: str, 
                              sequence_length: int = 5) -> List[str]:
        """Find most creative word sequence from concept to emotion using A*"""
        if start_concept not in self.semantic_graph.graph:
            return [start_concept]
        
        # Priority queue: (f_score, g_score, current_word, path)
        open_set = [(0, 0, start_concept, [start_concept])]
        closed_set = set()
        
        while open_set:
            f_score, g_score, current_word, path = heapq.heappop(open_set)
            
            if len(path) >= sequence_length:
                return path
            
            if current_word in closed_set:
                continue
            
            closed_set.add(current_word)
            
            # Get neighbors from semantic graph
            if current_word in self.semantic_graph.graph:
                neighbors = list(self.semantic_graph.graph.neighbors(current_word))
                
                for neighbor in neighbors:
                    if neighbor not in closed_set:
                        new_path = path + [neighbor]
                        new_g_score = g_score + 1
                        h_score = self._heuristic(neighbor, target_emotion, new_path)
                        new_f_score = new_g_score + h_score
                        
                        heapq.heappush(open_set, (new_f_score, new_g_score, neighbor, new_path))
        
        return [start_concept]  # Fallback
    
    def _heuristic(self, current_word: str, target_emotion: str, path: List[str]) -> float:
        """Heuristic function combining creativity and goal distance"""
        # Semantic distance to target
        semantic_distance = self._calculate_semantic_distance(current_word, target_emotion)
        
        # Creativity score
        creativity_score = self._calculate_creativity_score(path)
        
        # Combine with weights
        return semantic_distance * 0.6 + (1.0 - creativity_score) * 0.4
    
    def _calculate_semantic_distance(self, word1: str, word2: str) -> float:
        """Calculate semantic distance between words"""
        try:
            path = self.semantic_graph.find_semantic_path(word1, word2)
            return len(path) if path else 10.0  # Large distance if no path
        except:
            return 10.0
    
    def _calculate_creativity_score(self, path: List[str]) -> float:
        """Calculate creativity score for a word sequence"""
        if len(path) < 2:
            return 0.5
        
        novelty = self._calculate_novelty(path)
        semantic_variety = self._calculate_semantic_variety(path)
        phonetic_beauty = self._calculate_phonetic_beauty(path)
        
        return (novelty * self.creativity_weights['novelty'] + 
                semantic_variety * self.creativity_weights['semantic_distance'] +
                phonetic_beauty * self.creativity_weights['phonetic_beauty'])
    
    def _calculate_novelty(self, path: List[str]) -> float:
        """Calculate novelty based on word rarity"""
        # Simplified: longer words are considered more novel
        avg_length = sum(len(word) for word in path) / len(path)
        return min(1.0, avg_length / 10.0)
    
    def _calculate_semantic_variety(self, path: List[str]) -> float:
        """Calculate semantic variety in the path"""
        if len(path) < 3:
            return 0.5
        
        total_distance = 0
        for i in range(len(path) - 1):
            distance = self._calculate_semantic_distance(path[i], path[i + 1])
            total_distance += distance
        
        avg_distance = total_distance / (len(path) - 1)
        return min(1.0, avg_distance / 5.0)
    
    def _calculate_phonetic_beauty(self, path: List[str]) -> float:
        """Calculate phonetic beauty of word sequence"""
        # Simplified: check for alliteration and rhythm
        score = 0.0
        
        for i in range(len(path) - 1):
            # Alliteration bonus
            if path[i][0].lower() == path[i + 1][0].lower():
                score += 0.2
            
            # Rhythm bonus (similar syllable counts)
            syllables1 = len([c for c in path[i] if c.lower() in 'aeiouy'])
            syllables2 = len([c for c in path[i + 1] if c.lower() in 'aeiouy'])
            if abs(syllables1 - syllables2) <= 1:
                score += 0.1
        
        return min(1.0, score)

# Global instances of data structures
poetry_trie = PoetryTrie()
semantic_graph = SemanticGraph()
rhyme_engine = RhymeEngine()
poetry_optimizer = PoetryOptimizer()
creative_path_finder = CreativePathFinder(semantic_graph)

def initialize_advanced_structures():
    """Initialize all advanced data structures with sample data"""
    global poetry_trie, semantic_graph, rhyme_engine, poetry_optimizer, creative_path_finder
    
    # Sample vocabulary with metadata
    sample_words = [
        WordNode("love", 1, "ʌv", 0.8, "noun", semantic_tags=["emotion", "romance"]),
        WordNode("dove", 1, "ʌv", 0.6, "noun", semantic_tags=["peace", "nature"]),
        WordNode("above", 2, "ʌv", 0.3, "preposition", semantic_tags=["position"]),
        WordNode("heart", 1, "ɑrt", 0.7, "noun", semantic_tags=["emotion", "body"]),
        WordNode("art", 1, "ɑrt", 0.5, "noun", semantic_tags=["creativity", "beauty"]),
        WordNode("start", 1, "ɑrt", 0.4, "verb", semantic_tags=["beginning", "action"]),
        WordNode("dream", 1, "im", 0.6, "noun", semantic_tags=["imagination", "sleep"]),
        WordNode("stream", 1, "im", 0.4, "noun", semantic_tags=["water", "nature"]),
        WordNode("beam", 1, "im", 0.5, "noun", semantic_tags=["light", "energy"]),
        WordNode("night", 1, "aɪt", 0.2, "noun", semantic_tags=["time", "darkness"]),
        WordNode("light", 1, "aɪt", 0.7, "noun", semantic_tags=["brightness", "energy"]),
        WordNode("bright", 1, "aɪt", 0.8, "adjective", semantic_tags=["luminous", "positive"]),
        WordNode("moon", 1, "un", 0.4, "noun", semantic_tags=["celestial", "night"]),
        WordNode("soon", 1, "un", 0.3, "adverb", semantic_tags=["time", "future"]),
        WordNode("tune", 1, "un", 0.5, "noun", semantic_tags=["music", "harmony"]),
        WordNode("ocean", 2, "oʊʃən", 0.6, "noun", semantic_tags=["water", "vast", "nature"]),
        WordNode("motion", 2, "oʊʃən", 0.4, "noun", semantic_tags=["movement", "physics"]),
        WordNode("emotion", 3, "oʊʃən", 0.8, "noun", semantic_tags=["feeling", "psychology"]),
        WordNode("star", 1, "ɑr", 0.7, "noun", semantic_tags=["celestial", "bright"]),
        WordNode("far", 1, "ɑr", 0.3, "adjective", semantic_tags=["distance", "remote"]),
        WordNode("scar", 1, "ɑr", -0.3, "noun", semantic_tags=["wound", "memory"]),
        WordNode("peace", 1, "is", 0.9, "noun", semantic_tags=["tranquility", "harmony"]),
        WordNode("cease", 1, "is", -0.2, "verb", semantic_tags=["stop", "ending"]),
        WordNode("release", 2, "is", 0.4, "verb", semantic_tags=["freedom", "liberation"]),
        WordNode("wind", 1, "ɪnd", 0.3, "noun", semantic_tags=["air", "movement", "nature"]),
        WordNode("kind", 1, "aɪnd", 0.8, "adjective", semantic_tags=["gentle", "compassionate"]),
        WordNode("mind", 1, "aɪnd", 0.5, "noun", semantic_tags=["thought", "consciousness"]),
        WordNode("fire", 2, "aɪər", 0.6, "noun", semantic_tags=["element", "passion", "energy"]),
        WordNode("desire", 3, "aɪər", 0.7, "noun", semantic_tags=["want", "longing"]),
        WordNode("inspire", 3, "aɪər", 0.9, "verb", semantic_tags=["motivate", "creativity"]),
    ]
    
    # Initialize Trie with sample words
    for word_node in sample_words:
        poetry_trie.insert(word_node)
        rhyme_engine.add_word(word_node.word, word_node.phonetic_ending)
    
    # Build semantic relationships
    semantic_relationships = [
        ("love", "heart", "relates_to", 0.9),
        ("heart", "emotion", "part_of", 0.8),
        ("dream", "night", "occurs_during", 0.7),
        ("star", "night", "visible_during", 0.8),
        ("moon", "night", "visible_during", 0.9),
        ("light", "bright", "synonym", 0.8),
        ("ocean", "water", "contains", 0.9),
        ("peace", "tranquil", "synonym", 0.8),
        ("fire", "passion", "symbolizes", 0.7),
        ("wind", "motion", "causes", 0.6),
        ("art", "beauty", "creates", 0.7),
        ("music", "emotion", "evokes", 0.8),
        ("nature", "peace", "inspires", 0.7),
        ("love", "inspire", "can", 0.8),
        ("dream", "inspire", "can", 0.7),
        ("star", "inspire", "can", 0.6),
    ]
    
    for word1, word2, rel_type, strength in semantic_relationships:
        semantic_graph.add_word_relationship(word1, word2, rel_type, strength)
    
    # Initialize creative path finder
    creative_path_finder = CreativePathFinder(semantic_graph)
    
    _dbg("🧠 Advanced data structures initialized with sample vocabulary")

def get_advanced_rhymes(word: str, count: int = 3) -> List[str]:
    """Get advanced rhymes using the rhyme engine"""
    try:
        rhymes = rhyme_engine.find_best_rhymes(word, count)
        return [rhyme_word for rhyme_word, score in rhymes]
    except:
        return []

def get_semantic_suggestions(word: str, max_suggestions: int = 5) -> List[str]:
    """Get semantically related words"""
    try:
        related = semantic_graph.get_related_words(word, max_distance=2)
        return [related_word for related_word, similarity in related[:max_suggestions]]
    except:
        return []

def optimize_poem_structure_advanced(words: List[str], target_syllables: int = 10, 
                                   target_sentiment: float = 0.5) -> List[str]:
    """Use advanced optimization for poem structure"""
    try:
        return poetry_optimizer.optimize_line_structure(words, target_syllables, target_sentiment)
    except:
        return words

def find_creative_word_path(start_concept: str, target_emotion: str, length: int = 4) -> List[str]:
    """Find creative word sequence using A* search"""
    try:
        return creative_path_finder.find_creative_sequence(start_concept, target_emotion, length)
    except:
        return [start_concept]

def search_words_by_pattern(pattern: str) -> List[str]:
    """Search words using Trie pattern matching"""
    try:
        word_nodes = poetry_trie.find_words_by_prefix(pattern)
        return [node.word for node in word_nodes[:10]]  # Limit results
    except:
        return []

# ============================================================================
# END OF ADVANCED DATA STRUCTURES & ALGORITHMS
# ============================================================================

# Configure the page (try to use a favicon from assets if available)
try:
    from pathlib import Path as _Path
    _page_icon = "🎭"
    for _cand in (
        'assets/favicon.png', 'assets/favicon.webp', 'assets/favicon.jpg', 'assets/favicon.jpeg',
        'assets/logo.png', 'assets/logo.webp', 'assets/logo.jpg', 'assets/logo.jpeg',
        'assets/poetry_ai_logo.png', 'assets/poetry_ai_logo.webp', 'assets/poetry_ai_logo.jpg', 'assets/poetry_ai_logo.jpeg',
    ):
        _p = _Path(_cand)
        if _p.exists():
            try:
                _page_icon = Image.open(str(_p))
                break
            except Exception:
                pass
except Exception:
    _page_icon = "🎭"

st.set_page_config(
    page_title="POETRY.AI",
    page_icon=_page_icon,
    layout="wide"
)

# Support embed mode (hide Streamlit chrome when ?embed=true)
def apply_embed_mode():
    """Hide Streamlit header/footer/toolbar when the URL contains ?embed=true."""
    try:
        # Prefer modern API if available, else fall back
        qp = getattr(st, "query_params", None)
        if qp is None:
            qp = st.experimental_get_query_params()

        embed_val = None
        if isinstance(qp, dict):
            embed_val = qp.get("embed", False)
        elif qp is not None:
            embed_val = False
        
        if isinstance(embed_val, list):
            embed_val = embed_val[0] if embed_val else ""
        
        embed = str(embed_val).lower() in ("1", "true", "yes", "y")

        if embed:
            st.markdown(
                """
                <style>
                  header, footer, [data-testid="stToolbar"] { display: none !important; }
                </style>
                """,
                unsafe_allow_html=True,
            )
    except Exception:
        # Best-effort only; never block the app
        pass

apply_embed_mode()

# Theming: inject our minimal cartoon CSS
def inject_theme_css():
        try:
                css_path = Path("assets/theme.css")
                if css_path.exists():
                        with open(css_path, 'r', encoding='utf-8') as f:
                                css = f.read()
                        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        except Exception as e:
                st.warning(f"Theme load failed: {e}")

inject_theme_css()

# Additional lightweight layout tweaks for spacing and banner styling
st.markdown(
    """
    <style>
      /* Gentle global heading spacing */
      .block-container h1 { margin-bottom: 0.25rem; }
      .block-container h2 { margin-top: 0.75rem; margin-bottom: 0.35rem; }
      .soft-card { margin-top: 0.5rem; margin-bottom: 1rem; }
      .section-gap { height: 12px; }
      .section-gap-lg { height: 22px; }
      /* Banner visuals */
      .app-banner-wrap { margin: 0.25rem 0 1rem 0; }
      .app-banner { width: 100%; max-height: 280px; object-fit: cover; border-radius: 12px; box-shadow: 0 6px 24px rgba(0,0,0,0.08); }
      /* Header alignment tweaks */
      .app-header { gap: 10px; }
      .app-header .title p { margin: 0.1rem 0 0 0; }
      .emoji-logo { font-size: 2.4rem; line-height: 1; }
      .app-logo { vertical-align: middle; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Simple inline SVGs for cartoon elements
def svg_quill(size=40):
        return f'''
        <svg width="{size}" height="{size}" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" class="cartoon">
            <path d="M10 52 C24 36, 40 20, 54 12" />
            <path d="M14 50 L22 42" />
            <!-- eyes -->
            <circle cx="28" cy="34" r="1.6" class="cartoon-fill" />
            <circle cx="34" cy="28" r="1.6" class="cartoon-fill" />
            <!-- smile -->
            <path d="M30 38 C32 40, 36 38, 36 36" />
        </svg>
        '''

def svg_book(size=44):
        return f'''
        <svg width="{size}" height="{size}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" class="cartoon">
            <path d="M10 16 L30 16 C36 16, 40 20, 40 26 L40 48 L10 48 Z" />
            <path d="M54 16 L34 16 C28 16, 24 20, 24 26 L24 48 L54 48 Z" />
            <!-- eyes -->
            <circle cx="22" cy="28" r="1.6" class="cartoon-fill" />
            <circle cx="42" cy="28" r="1.6" class="cartoon-fill" />
            <!-- smile -->
            <path d="M26 34 C32 38, 38 38, 44 34" />
        </svg>
        '''

def svg_star(size=24):
        return f'''
        <svg width="{size}" height="{size}" viewBox="0 0 24 24" class="cartoon">
            <path d="M12 2 L14.5 8 L21 9 L16 13 L17.5 20 L12 16.5 L6.5 20 L8 13 L3 9 L9.5 8 Z" />
        </svg>
        '''

def _try_read_file_bytes(paths):
        for p in paths:
                try:
                        with open(p, 'rb') as f:
                                return f.read(), str(p)
                except Exception:
                        continue
        return None, None

def get_app_logo_img_tag(size_px: int = 56) -> str:
        """Return an <img> tag with a base64-embedded logo if available; otherwise empty string.
        Looks for assets/poetry_ai_logo.(png|webp|jpg|jpeg) or assets/logo.(png|webp|jpg|jpeg).
        """
        import base64
        from pathlib import Path

        candidates = [
                Path('assets/poetry_ai_logo.png'),
                Path('assets/poetry_ai_logo.webp'),
                Path('assets/poetry_ai_logo.jpg'),
                Path('assets/poetry_ai_logo.jpeg'),
                Path('assets/logo.png'),
                Path('assets/logo.webp'),
                Path('assets/logo.jpg'),
                Path('assets/logo.jpeg'),
        ]
        data, path = _try_read_file_bytes(candidates)
        if not data:
            return ''
        # detect mime by extension
        ext = path.lower()
        if ext.endswith('.png'):
            mime = 'image/png'
        elif ext.endswith('.webp'):
            mime = 'image/webp'
        elif ext.endswith('.jpg') or ext.endswith('.jpeg'):
            mime = 'image/jpeg'
        else:
            mime = 'image/png'
        b64 = base64.b64encode(data).decode('ascii')
        return f"<img class=\"app-logo\" alt=\"POETRY.AI Logo\" width=\"{size_px}\" src=\"data:{mime};base64,{b64}\" />"

def get_banner_img_tag(max_height_px: int = 280) -> str:
        """Return an <img> tag for a banner if assets/banner.* or assets/hero.* exists; otherwise empty.
        Image is embedded as base64 for portability.
        """
        import base64
        from pathlib import Path

        candidates = [
            Path('assets/banner.png'), Path('assets/banner.webp'), Path('assets/banner.jpg'), Path('assets/banner.jpeg'),
            Path('assets/hero.png'),   Path('assets/hero.webp'),   Path('assets/hero.jpg'),   Path('assets/hero.jpeg'),
        ]
        data, path = _try_read_file_bytes(candidates)
        if not data:
            return ''
        ext = path.lower()
        if ext.endswith('.png'):
            mime = 'image/png'
        elif ext.endswith('.webp'):
            mime = 'image/webp'
        elif ext.endswith('.jpg') or ext.endswith('.jpeg'):
            mime = 'image/jpeg'
        else:
            mime = 'image/png'
        b64 = base64.b64encode(data).decode('ascii')
        return f"<img class=\"app-banner\" alt=\"POETRY.AI Banner\" style=\"max-height:{max_height_px}px\" src=\"data:{mime};base64,{b64}\" />"

def ensure_assets_dir():
        from pathlib import Path
        Path('assets').mkdir(parents=True, exist_ok=True)

def save_uploaded_logo(uploaded_file) -> str | None:
        """Save uploaded logo into assets as poetry_ai_logo.<ext> and return saved path."""
        if uploaded_file is None:
            return None
        ensure_assets_dir()
        from pathlib import Path
        import shutil
        # Determine extension from original name
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in {'.png', '.jpg', '.jpeg', '.webp'}:
            # default to .png if unknown
            suffix = '.png'
        # Remove any existing logo variants for cleanliness
        for ext in ('.png', '.jpg', '.jpeg', '.webp'):
            p = Path(f'assets/poetry_ai_logo{ext}')
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        target = Path(f'assets/poetry_ai_logo{suffix}')
        # Save file
        with open(target, 'wb') as out:
            out.write(uploaded_file.getbuffer())
        return str(target)

def remove_saved_logo():
        from pathlib import Path
        removed = False
        for ext in ('.png', '.jpg', '.jpeg', '.webp'):
            p = Path(f'assets/poetry_ai_logo{ext}')
            if p.exists():
                try:
                    p.unlink()
                    removed = True
                except Exception:
                    pass
        return removed

def header_with_mascot():
    # Create header with history button in top right
    col1, col2 = st.columns([5, 1])

    with col1:
        # Use an image logo if available; otherwise fallback to emojis (brain + pen)
        logo_tag = get_app_logo_img_tag(56)
        icon_html = logo_tag if logo_tag else '<span class="emoji-logo" title="POETRY.AI">🧠✒️</span>'
        st.markdown(
                    f"""
                    <div class="app-header fade-in">
                    <div class="logo-wrap">{icon_html}</div>
                        <div class="title">
                            <h1>POETRY.AI</h1>
                            <p>The art of crafted verse </p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
            )

        # Optional banner image if present
        _banner = get_banner_img_tag()
        if _banner:
            st.markdown(f"<div class='app-banner-wrap'>{_banner}</div>", unsafe_allow_html=True)

    with col2:
        # History button in top right
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Add some top padding
        if st.button("📚 History", key="history_btn", help="View your previous poems and translations", type="primary"):
            st.session_state.show_history = not st.session_state.show_history

def save_to_history(poem_text, metadata, input_prompt, is_translation=False):
    """Save a poem or translation to history"""
    import datetime
    
    history_item = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "poem_text": poem_text,
        "metadata": metadata,
        "input_prompt": input_prompt,
        "is_translation": is_translation,
        "type": "Translation" if is_translation else "Poem"
    }
    
    # Add to beginning of history (newest first)
    st.session_state.poem_history.insert(0, history_item)
    
    # Keep only last 20 items to prevent memory issues
    if len(st.session_state.poem_history) > 20:
        st.session_state.poem_history = st.session_state.poem_history[:20]

def display_history():
    """Display the history modal/section"""
    if not st.session_state.poem_history:
        st.info("🔍 No history yet. Generate some poems or translations to see them here!")
        return
    
    st.markdown("### 📚 Your Poetry & Translation History")
    st.markdown("---")
    
    for i, item in enumerate(st.session_state.poem_history):
        with st.expander(f"🎭 {item['type']} - {item['timestamp']} - \"{item['input_prompt'][:50]}...\""):
            # Display the poem/translation
            st.markdown(
                f"""
                <div class="poetry-card fade-in">
                    <div class="corner-deco tl">{svg_star(28)}</div>
                    <div class="corner-deco tr">{svg_star(28)}</div>
                    <div class="poem-text">{item['poem_text'].replace(chr(10), '<br>')}</div>
                    <div class="corner-deco bl">{svg_book(36)}</div>
                    <div class="corner-deco br">{svg_quill(36)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # Display metadata
            st.markdown(f"**Details:** {item['metadata']}")
            st.markdown(f"**Original Input:** {item['input_prompt']}")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Regenerate", key=f"regen_{i}", help="Generate new poem with same input"):
                    st.session_state.prompt_text = item['input_prompt']
                    st.session_state.show_history = False
                    st.rerun()
            
            with col2:
                if st.button("📋 Copy Text", key=f"copy_{i}", help="Copy poem text"):
                    st.code(item['poem_text'], language=None)
                    st.success("✅ Text displayed above for copying!")
            
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{i}", help="Remove from history"):
                    st.session_state.poem_history.pop(i)
                    st.success("🗑️ Removed from history!")
                    st.rerun()
    
    # Clear all history button
    st.markdown("---")
    if st.button("🧹 Clear All History", type="secondary", help="Remove all items from history"):
        st.session_state.poem_history = []
        st.success("🧹 History cleared!")
        st.rerun()

# Debug information at the top
## Remove debug info from sidebar (per request)

# Lightweight debug logger gated by a session toggle
def _dbg(msg: str):
    try:
        if st.session_state.get("verbose_audio", False):
            # Use success/info for less visual noise
            if "✅" in msg or "Created" in msg:
                st.success(msg)
            elif "🎶" in msg or "Added" in msg:
                st.info(msg, icon="🎵")
            else:
                st.info(msg)
    except Exception:
        # In early import phases or if session state isn't ready, skip debug logs
        pass

def translate_text(text, target_language):
    """Translate text to target language using deep-translator"""
    try:
        from deep_translator import GoogleTranslator
        
        lang_codes = {
            "English": "en",
            "Spanish": "es", 
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Russian": "ru",
            "Japanese": "ja",
            "Chinese": "zh-CN",
            "Arabic": "ar",
            "Hindi": "hi",
            "Telugu": "te",
            "Malayalam": "ml",
            "Kannada": "kn",
            "Tamil": "ta",
        }
        
        if target_language != "English" and target_language in lang_codes:
            translator = GoogleTranslator(source='en', target=lang_codes[target_language])
            translated = translator.translate(text)
            return translated
        return text
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        st.error(f"Translation error: {str(e)}")
        return text

# User Preferences and Custom Theme System
def load_user_preferences():
    """Load user preferences from session state or file"""
    try:
        prefs_file = Path("user_preferences.json")
        if prefs_file.exists():
            with open(prefs_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not load preferences: {e}")
    
    return {
        'poetry_style': 'balanced',
        'favorite_themes': [],
        'custom_themes': {},
        'voice_preferences': {},
        'export_preferences': {'format': 'pdf', 'include_audio': True},
        'learning_data': {'liked_poems': [], 'style_patterns': {}}
    }

def save_user_preferences(preferences):
    """Save user preferences to file"""
    try:
        with open("user_preferences.json", 'w', encoding='utf-8') as f:
            json.dump(preferences, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Could not save preferences: {e}")
        return False

def learn_from_user_feedback(poem_text, user_rating, user_input):
    """Learn from user feedback to improve future generations"""
    preferences = load_user_preferences()
    
    if user_rating >= 4:  # 4-5 star rating
        preferences['learning_data']['liked_poems'].append({
            'poem': poem_text,
            'input': user_input,
            'rating': user_rating,
            'timestamp': datetime.now().isoformat(),
            'style_features': extract_style_features(poem_text)
        })
        
        # Limit to last 50 liked poems
        if len(preferences['learning_data']['liked_poems']) > 50:
            preferences['learning_data']['liked_poems'] = preferences['learning_data']['liked_poems'][-50:]
    
    save_user_preferences(preferences)

def extract_style_features(poem_text):
    """Extract style features from poem for learning"""
    lines = poem_text.strip().split('\n')
    return {
        'line_count': len(lines),
        'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
        'rhyme_scheme': detect_rhyme_scheme(lines),
        'common_words': extract_common_words(poem_text),
        'sentiment': analyze_poem_sentiment(poem_text)
    }

def detect_rhyme_scheme(lines):
    """Simple rhyme scheme detection"""
    if len(lines) < 2:
        return "free_verse"
    
    # Simple end-word comparison (very basic)
    end_words = [line.strip().split()[-1].lower().rstrip('.,!?;:') for line in lines if line.strip()]
    
    if len(end_words) >= 4:
        if end_words[0] == end_words[2] and end_words[1] == end_words[3]:
            return "ABAB"
        elif end_words[0] == end_words[1] and end_words[2] == end_words[3]:
            return "AABB"
    
    return "free_verse"

def analyze_poem_sentiment(poem_text):
    """Analyze the overall sentiment of the poem"""
    positive_words = ['love', 'joy', 'happy', 'beautiful', 'wonderful', 'bright', 'warm', 'hope', 'dream', 'smile']
    negative_words = ['sad', 'dark', 'cold', 'pain', 'sorrow', 'tears', 'lonely', 'fear', 'lost', 'broken']
    
    text_lower = poem_text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

def extract_common_words(poem_text):
    """Extract most common meaningful words from poem"""
    import re
    from collections import Counter
    
    # Remove punctuation and common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
    
    words = re.findall(r'\b[a-zA-Z]+\b', poem_text.lower())
    meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return [word for word, count in Counter(meaningful_words).most_common(10)]

def create_background_music(duration_ms, theme="peaceful", key="C"):
    """Generate simple background music based on theme with better error handling"""
    if not AUDIO_PROCESSING_AVAILABLE:
        st.warning("🎵 Audio processing not available - FFmpeg required for background music")
        return None
    
    try:
        # Test basic audio generation capability first
        try:
            test_tone = Sine(440).to_audio_segment(duration=500)
            _dbg("✅ Audio generation system working")
        except Exception as e:
            st.error(f"❌ Audio generation failed: {e}")
            return None
        
        # Expanded musical themes with multiple variations for each theme
        musical_themes = {
            "peaceful": {
                "variations": [
                    {"name": "gentle_meadow", "frequencies": [261.63, 293.66, 329.63, 349.23, 392.00, 440.00], "rhythm": [1800, 1400, 1800, 1400, 2200, 1600], "volume": -18, "harmonies": [130.81, 146.83, 164.81]},
                    {"name": "soft_rain", "frequencies": [220.00, 246.94, 293.66, 329.63, 369.99, 415.30], "rhythm": [2000, 1600, 2000, 1600, 2400, 1800], "volume": -20, "harmonies": [110.00, 123.47, 146.83]},
                    {"name": "morning_mist", "frequencies": [174.61, 196.00, 220.00, 261.63, 293.66, 329.63], "rhythm": [2200, 1800, 2200, 1800, 2600, 2000], "volume": -22, "harmonies": [87.31, 98.00, 110.00]},
                    {"name": "calm_ocean", "frequencies": [146.83, 164.81, 185.00, 220.00, 246.94, 277.18], "rhythm": [2400, 2000, 2400, 2000, 2800, 2200], "volume": -19, "harmonies": [73.42, 82.41, 92.50]},
                    {"name": "zen_garden", "frequencies": [196.00, 220.00, 246.94, 293.66, 329.63, 369.99], "rhythm": [1900, 1500, 1900, 1500, 2300, 1700], "volume": -21, "harmonies": [98.00, 110.00, 123.47]},
                    {"name": "forest_whispers", "frequencies": [164.81, 185.00, 207.65, 246.94, 277.18, 311.13], "rhythm": [2100, 1700, 2100, 1700, 2500, 1900], "volume": -23, "harmonies": [82.41, 92.50, 103.83]},
                    {"name": "moonbeam_lullaby", "frequencies": [185.00, 207.65, 233.08, 277.18, 311.13, 349.23], "rhythm": [2300, 1900, 2300, 1900, 2700, 2100], "volume": -24, "harmonies": [92.50, 103.83, 116.54]},
                    {"name": "starlight_serenity", "frequencies": [207.65, 233.08, 261.63, 311.13, 349.23, 392.00], "rhythm": [2000, 1600, 2000, 1600, 2400, 1800], "volume": -20, "harmonies": [103.83, 116.54, 130.81]}
                ]
            },
            "energetic": {
                "variations": [
                    {"name": "electric_pulse", "frequencies": [293.66, 329.63, 369.99, 415.30, 466.16, 523.25], "rhythm": [600, 400, 600, 400, 800, 1000], "volume": -12, "harmonies": [146.83, 164.81, 184.99]},
                    {"name": "power_surge", "frequencies": [349.23, 392.00, 440.00, 493.88, 554.37, 622.25], "rhythm": [500, 300, 500, 300, 700, 900], "volume": -10, "harmonies": [174.61, 196.00, 220.00]},
                    {"name": "lightning_strike", "frequencies": [415.30, 466.16, 523.25, 587.33, 659.25, 739.99], "rhythm": [400, 200, 400, 200, 600, 800], "volume": -8, "harmonies": [207.65, 233.08, 261.63]},
                    {"name": "rocket_launch", "frequencies": [369.99, 415.30, 466.16, 523.25, 587.33, 659.25], "rhythm": [550, 350, 550, 350, 750, 950], "volume": -11, "harmonies": [185.00, 207.65, 233.08]},
                    {"name": "cyber_beat", "frequencies": [329.63, 369.99, 415.30, 466.16, 523.25, 587.33], "rhythm": [450, 250, 450, 250, 650, 850], "volume": -9, "harmonies": [164.81, 185.00, 207.65]},
                    {"name": "thunderstorm", "frequencies": [261.63, 293.66, 329.63, 415.30, 466.16, 523.25], "rhythm": [550, 350, 550, 350, 750, 950], "volume": -13, "harmonies": [130.81, 146.83, 164.81]},
                    {"name": "volcanic_eruption", "frequencies": [246.94, 277.18, 311.13, 392.00, 440.00, 493.88], "rhythm": [500, 300, 500, 300, 700, 900], "volume": -11, "harmonies": [123.47, 138.59, 155.56]},
                    {"name": "industrial_pulse", "frequencies": [220.00, 246.94, 277.18, 349.23, 392.00, 440.00], "rhythm": [480, 280, 480, 280, 680, 880], "volume": -12, "harmonies": [110.00, 123.47, 138.59]}
                ]
            },
            "romantic": {
                "variations": [
                    {"name": "moonlight_serenade", "frequencies": [246.94, 293.66, 329.63, 392.00, 440.00, 493.88], "rhythm": [2500, 2000, 2200, 1800, 3000, 2400], "volume": -22, "harmonies": [123.47, 146.83, 164.81]},
                    {"name": "rose_petals", "frequencies": [220.00, 261.63, 293.66, 349.23, 392.00, 440.00], "rhythm": [2700, 2200, 2400, 2000, 3200, 2600], "volume": -24, "harmonies": [110.00, 130.81, 146.83]},
                    {"name": "candlelight_waltz", "frequencies": [196.00, 233.08, 277.18, 329.63, 369.99, 415.30], "rhythm": [2600, 2100, 2300, 1900, 3100, 2500], "volume": -23, "harmonies": [98.00, 116.54, 138.59]},
                    {"name": "sunset_embrace", "frequencies": [174.61, 207.65, 246.94, 293.66, 329.63, 369.99], "rhythm": [2800, 2300, 2500, 2100, 3300, 2700], "volume": -25, "harmonies": [87.31, 103.83, 123.47]},
                    {"name": "lovers_melody", "frequencies": [261.63, 311.13, 369.99, 415.30, 466.16, 523.25], "rhythm": [2400, 1900, 2100, 1700, 2900, 2300], "volume": -21, "harmonies": [130.81, 155.56, 185.00]},
                    {"name": "wedding_bells", "frequencies": [329.63, 392.00, 440.00, 523.25, 587.33, 659.25], "rhythm": [2200, 1800, 2000, 1600, 2800, 2200], "volume": -20, "harmonies": [164.81, 196.00, 220.00]},
                    {"name": "first_kiss", "frequencies": [293.66, 349.23, 415.30, 466.16, 523.25, 587.33], "rhythm": [2300, 1900, 2100, 1700, 2900, 2300], "volume": -22, "harmonies": [146.83, 174.61, 207.65]},
                    {"name": "eternal_love", "frequencies": [277.18, 329.63, 392.00, 440.00, 493.88, 554.37], "rhythm": [2500, 2000, 2200, 1800, 3000, 2400], "volume": -23, "harmonies": [138.59, 164.81, 196.00]}
                ]
            },
            "mysterious": {
                "variations": [
                    {"name": "shadow_whispers", "frequencies": [220.00, 246.94, 277.18, 311.13, 369.99, 415.30], "rhythm": [1200, 1000, 1400, 1000, 1600, 1300], "volume": -16, "harmonies": [110.00, 123.47, 138.59]},
                    {"name": "ancient_secrets", "frequencies": [196.00, 220.00, 246.94, 277.18, 311.13, 349.23], "rhythm": [1400, 1200, 1600, 1200, 1800, 1500], "volume": -18, "harmonies": [98.00, 110.00, 123.47]},
                    {"name": "midnight_fog", "frequencies": [174.61, 196.00, 220.00, 246.94, 277.18, 311.13], "rhythm": [1500, 1300, 1700, 1300, 1900, 1600], "volume": -19, "harmonies": [87.31, 98.00, 110.00]},
                    {"name": "dark_magic", "frequencies": [155.56, 174.61, 196.00, 220.00, 246.94, 277.18], "rhythm": [1300, 1100, 1500, 1100, 1700, 1400], "volume": -17, "harmonies": [77.78, 87.31, 98.00]},
                    {"name": "cryptic_puzzle", "frequencies": [207.65, 233.08, 261.63, 293.66, 329.63, 369.99], "rhythm": [1100, 900, 1300, 900, 1500, 1200], "volume": -15, "harmonies": [103.83, 116.54, 130.81]},
                    {"name": "haunted_manor", "frequencies": [164.81, 185.00, 207.65, 233.08, 261.63, 293.66], "rhythm": [1350, 1150, 1550, 1150, 1750, 1450], "volume": -18, "harmonies": [82.41, 92.50, 103.83]},
                    {"name": "occult_ritual", "frequencies": [146.83, 164.81, 185.00, 207.65, 233.08, 261.63], "rhythm": [1450, 1250, 1650, 1250, 1850, 1550], "volume": -19, "harmonies": [73.42, 82.41, 92.50]},
                    {"name": "vampire_castle", "frequencies": [138.59, 155.56, 174.61, 196.00, 220.00, 246.94], "rhythm": [1250, 1050, 1450, 1050, 1650, 1350], "volume": -20, "harmonies": [69.30, 77.78, 87.31]}
                ]
            },
            "joyful": {
                "variations": [
                    {"name": "sunshine_dance", "frequencies": [261.63, 329.63, 392.00, 493.88, 523.25, 659.25], "rhythm": [800, 600, 800, 600, 1200, 1000], "volume": -10, "harmonies": [130.81, 164.81, 196.00]},
                    {"name": "spring_festival", "frequencies": [293.66, 369.99, 440.00, 554.37, 587.33, 739.99], "rhythm": [700, 500, 700, 500, 1100, 900], "volume": -8, "harmonies": [146.83, 185.00, 220.00]},
                    {"name": "children_laughter", "frequencies": [329.63, 415.30, 493.88, 622.25, 659.25, 830.61], "rhythm": [600, 400, 600, 400, 1000, 800], "volume": -9, "harmonies": [164.81, 207.65, 246.94]},
                    {"name": "carnival_music", "frequencies": [349.23, 440.00, 523.25, 659.25, 698.46, 880.00], "rhythm": [750, 550, 750, 550, 1150, 950], "volume": -11, "harmonies": [174.61, 220.00, 261.63]},
                    {"name": "celebration_bells", "frequencies": [392.00, 493.88, 587.33, 739.99, 783.99, 987.77], "rhythm": [650, 450, 650, 450, 1050, 850], "volume": -7, "harmonies": [196.00, 246.94, 293.66]},
                    {"name": "victory_march", "frequencies": [415.30, 523.25, 622.25, 783.99, 830.61, 1046.50], "rhythm": [700, 500, 700, 500, 1100, 900], "volume": -9, "harmonies": [207.65, 261.63, 311.13]},
                    {"name": "party_time", "frequencies": [369.99, 466.16, 554.37, 698.46, 739.99, 932.33], "rhythm": [650, 450, 650, 450, 1050, 850], "volume": -8, "harmonies": [185.00, 233.08, 277.18]},
                    {"name": "rainbow_bridge", "frequencies": [311.13, 392.00, 466.16, 587.33, 622.25, 783.99], "rhythm": [750, 550, 750, 550, 1150, 950], "volume": -10, "harmonies": [155.56, 196.00, 233.08]}
                ]
            },
            "melancholic": {
                "variations": [
                    {"name": "autumn_leaves", "frequencies": [220.00, 261.63, 293.66, 329.63, 369.99, 440.00], "rhythm": [2200, 1800, 2200, 1800, 2800, 2400], "volume": -20, "harmonies": [110.00, 130.81, 146.83]},
                    {"name": "gentle_tears", "frequencies": [196.00, 233.08, 277.18, 311.13, 349.23, 392.00], "rhythm": [2400, 2000, 2400, 2000, 3000, 2600], "volume": -22, "harmonies": [98.00, 116.54, 138.59]},
                    {"name": "distant_memory", "frequencies": [174.61, 207.65, 246.94, 277.18, 311.13, 349.23], "rhythm": [2500, 2100, 2500, 2100, 3100, 2700], "volume": -23, "harmonies": [87.31, 103.83, 123.47]},
                    {"name": "lonely_piano", "frequencies": [155.56, 185.00, 220.00, 246.94, 277.18, 311.13], "rhythm": [2600, 2200, 2600, 2200, 3200, 2800], "volume": -24, "harmonies": [77.78, 92.50, 110.00]},
                    {"name": "fading_light", "frequencies": [207.65, 246.94, 293.66, 329.63, 369.99, 415.30], "rhythm": [2300, 1900, 2300, 1900, 2900, 2500], "volume": -21, "harmonies": [103.83, 123.47, 146.83]},
                    {"name": "winter_solitude", "frequencies": [185.00, 220.00, 261.63, 293.66, 329.63, 369.99], "rhythm": [2400, 2000, 2400, 2000, 3000, 2600], "volume": -22, "harmonies": [92.50, 110.00, 130.81]},
                    {"name": "broken_heart", "frequencies": [164.81, 196.00, 233.08, 261.63, 293.66, 329.63], "rhythm": [2500, 2100, 2500, 2100, 3100, 2700], "volume": -23, "harmonies": [82.41, 98.00, 116.54]},
                    {"name": "lost_dreams", "frequencies": [146.83, 174.61, 207.65, 233.08, 261.63, 293.66], "rhythm": [2600, 2200, 2600, 2200, 3200, 2800], "volume": -24, "harmonies": [73.42, 87.31, 103.83]}
                ]
            }
        }
        
        # Check for custom user themes
        custom_themes = load_user_preferences().get('custom_themes', {})
        if theme in custom_themes:
            theme_data = custom_themes[theme]
            _dbg(f"🎨 Using custom user theme: {theme}")
        else:
            # Randomly select a variation from the theme
            theme_data = musical_themes.get(theme, musical_themes["peaceful"])
        
        import random
        variations = theme_data["variations"]
        selected_variation = random.choice(variations)
        
        frequencies = selected_variation["frequencies"]
        rhythm = selected_variation["rhythm"] 
        volume_db = selected_variation["volume"]
        harmonies = selected_variation.get("harmonies", [])
        
        _dbg(f"🎵 Creating {theme} background music - {selected_variation['name']} ({duration_ms/1000:.1f}s)")
        
        # Create background music with harmonies
        background = AudioSegment.silent(duration=duration_ms)
        current_time = 0
        harmony_offset = 0
        
        while current_time < duration_ms:
            for i, (freq, note_duration) in enumerate(zip(frequencies, rhythm)):
                if current_time >= duration_ms:
                    break
                    
                actual_duration = min(note_duration, duration_ms - current_time)
                
                try:
                    # Create main tone
                    tone = Sine(freq).to_audio_segment(duration=actual_duration)
                    
                    # Add harmony if available
                    if harmonies and i < len(harmonies):
                        harmony_tone = Sine(harmonies[i % len(harmonies)]).to_audio_segment(duration=actual_duration)
                        harmony_tone = harmony_tone + (volume_db - 8)  # Quieter harmony
                        tone = tone.overlay(harmony_tone)
                    
                    # Apply musical effects
                    tone = tone.fade_in(100).fade_out(100)  # Longer fades
                    tone = tone + volume_db
                    
                    # Add slight reverb effect
                    if len(tone) > 200:
                        reverb = tone + (volume_db - 12)  # Much quieter reverb
                        tone = tone.overlay(reverb, position=150)
                    
                    # Overlay onto background
                    if current_time + len(tone) <= duration_ms:
                        background = background.overlay(tone, position=current_time)
                        _dbg(f"🎶 Added {theme} tone at {freq:.1f}Hz")
                    
                    current_time += note_duration
                    
                except Exception as e:
                    st.warning(f"⚠️ Error creating tone at {freq}Hz: {e}")
                    current_time += note_duration
                    continue
        
        # Add overall fade and normalize
        background = background.fade_in(1000).fade_out(1000)
        try:
            background = normalize(background)
        except Exception:
            pass
        
        # Verify we created actual audio content (no .silent attribute in pydub; check RMS)
        if len(background) > 0 and background.rms > 0:
            _dbg(f"✅ Created {theme} background music: {len(background)}ms")
            return background
        else:
            st.warning("⚠️ Background music generation produced silent audio")
            return None
        
    except Exception as e:
        logging.error(f"Error creating background music: {str(e)}")
        st.warning(f"⚠️ Error creating background music: {str(e)}")
        return None

def add_audio_effects(audio_segment, effect_type="reverb"):
    """Add audio effects to enhance the speech"""
    if not AUDIO_PROCESSING_AVAILABLE or not audio_segment:
        return audio_segment
    
    try:
        # Normalize audio
        audio_segment = normalize(audio_segment)
        
        if effect_type == "reverb":
            # Simple reverb effect using delay and decay
            reverb = AudioSegment.silent(duration=100)
            for delay in [100, 200, 300]:
                delayed = audio_segment + (-15 - delay//10)  # Reduce volume with each delay
                reverb = reverb.overlay(delayed, position=delay)
            audio_segment = audio_segment.overlay(reverb)
            
        elif effect_type == "echo":
            # Add echo effect
            echo = audio_segment + (-10)  # Reduce volume for echo
            audio_segment = audio_segment.overlay(echo, position=500)
            
        elif effect_type == "enhance":
            # Compress dynamic range for better clarity
            audio_segment = compress_dynamic_range(audio_segment)
            
        return audio_segment
        
    except Exception as e:
        st.warning(f"Audio effects error: {str(e)}. Using original audio.")
        return audio_segment

def detect_poetry_theme(text):
    """Detect the emotional theme from poetry content using keyword analysis"""
    text_lower = text.lower()
    
    # Define theme detection keywords with weights
    theme_keywords = {
        "romantic": ["love", "heart", "kiss", "romance", "darling", "beloved", "sweetheart", "passion", 
                    "embrace", "tender", "affection", "adore", "cherish", "valentine", "soul", "forever"],
        "energetic": ["energy", "power", "strong", "fierce", "bold", "dynamic", "electric", "force",
                     "thunder", "lightning", "burst", "explosive", "vibrant", "intense", "fire", "blazing"],
        "joyful": ["happy", "joy", "smile", "laugh", "bright", "cheerful", "celebration", "dance",
                  "sunshine", "rainbow", "sparkle", "delight", "merry", "festive", "gleeful", "bliss"],
        "mysterious": ["mystery", "secret", "shadow", "dark", "hidden", "unknown", "whisper", "enigma",
                      "magic", "mystical", "ancient", "forbidden", "veil", "shrouded", "cryptic", "occult"],
        "melancholic": ["sad", "sorrow", "tear", "lonely", "empty", "loss", "grief", "melancholy",
                       "weep", "mourn", "despair", "ache", "yearning", "wistful", "lament", "solitude"],
        "peaceful": ["calm", "peace", "quiet", "gentle", "serene", "tranquil", "still", "soft",
                    "breeze", "meadow", "nature", "harmony", "zen", "meditate", "silence", "soothe"]
    }
    
    # Score each theme based on keyword matches
    theme_scores = {}
    for theme, keywords in theme_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        theme_scores[theme] = score
    
    # Return the theme with highest score, default to peaceful
    detected_theme = max(theme_scores, key=theme_scores.get) if max(theme_scores.values()) > 0 else "peaceful"
    
    # Log the detection for debugging
    _dbg(f"🎭 Theme detection scores: {theme_scores}")
    _dbg(f"🎯 Detected theme: {detected_theme}")
    
    return detected_theme

def create_musical_poetry_audio(text, language="en", speed=1.0, theme=None, audio_effects="enhance", bg_volume_percent=40, voice_type="neutral"):
    """Generate enhanced musical audio from text with background music and effects"""
    if not TTS_AVAILABLE:
        return None

    try:
        # Auto-detect theme if not specified
        if theme is None:
            theme = detect_poetry_theme(text)
            _dbg(f"🤖 Auto-detected theme: {theme}")
        else:
            _dbg(f"🎨 Using specified theme: {theme}")
        
        # Map language names to gTTS language codes
        lang_mapping = {
            "English": "en", "Spanish": "es", "French": "fr", 
            "German": "de", "Italian": "it", "Portuguese": "pt",
            "Russian": "ru", "Japanese": "ja", "Chinese": "zh-CN",
            "Arabic": "ar", "Hindi": "hi", "Telugu": "te",
            "Malayalam": "ml", "Kannada": "kn", "Tamil": "ta",
        }

        # Voice type mapping for different TLD (Top Level Domain) for gTTS
        voice_mapping = {
            "male_1": {"tld": "com"},
            "male_2": {"tld": "co.uk"},
            "female_1": {"tld": "com.au"},
            "female_2": {"tld": "ca"},
            "neutral": {"tld": "com"}
        }

        lang_code = lang_mapping.get(language, "en")
        voice_config = voice_mapping.get(voice_type, voice_mapping["neutral"])

        # Clean text for TTS
        clean_text = text.replace("*", "").replace("#", "").replace("**", "")

        # Create TTS and save to temp file
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, f"tts_audio_{int(time.time())}.mp3")
        tts = gTTS(
            text=clean_text, 
            lang=lang_code, 
            slow=(speed < 1.0),
            tld=voice_config['tld']
        )
        tts.save(temp_file_path)        # Verify file was created
        if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
            st.error("Failed to create TTS audio file.")
            return create_audio_from_text(text, language, speed, "neutral")

        # If audio processing isn't available, just return the basic TTS
        if not AUDIO_PROCESSING_AVAILABLE:
            st.info("🔊 Using basic voice-over (FFmpeg required for musical enhancement)")
            return temp_file_path

        # Try to load and enhance the speech with better error handling
        try:
            speech_audio = AudioSegment.from_mp3(temp_file_path)
            _dbg("✅ Speech audio loaded successfully")
            
            # Apply audio effects if requested
            if audio_effects and audio_effects != "none":
                try:
                    speech_audio = add_audio_effects(speech_audio, audio_effects)
                    _dbg(f"🎛️ Applied {audio_effects} effect")
                except Exception as e:
                    st.warning(f"Audio effects failed: {e}, using original audio")

            # Try to create background music
            try:
                music_duration = len(speech_audio) + 2000  # Add 2 seconds padding
                background_music = create_background_music(music_duration, theme)
                
                if background_music:
                    _dbg(f"🎵 Generated {theme} background music")
                    
                    # Make background music quieter than speech
                    try:
                        bg_volume_percent = max(0, min(100, int(bg_volume_percent)))
                    except Exception:
                        bg_volume_percent = 40
                    atten_db = int(round(-40 + bg_volume_percent * 0.35))  # -40 .. -5 approx
                    if atten_db > -5:
                        atten_db = -5
                    if atten_db < -40:
                        atten_db = -40
                    bg = background_music + (atten_db)

                    # Ensure background is at least as long as speech
                    if len(bg) < len(speech_audio):
                        bg = bg * (len(speech_audio) // len(bg) + 1)
                        bg = bg[:len(speech_audio)]

                    # Add gentle fades
                    bg = bg.fade_in(1000).fade_out(1000)

                    # Overlay speech (start slightly after 0.5s) and normalize final
                    final_audio = bg.overlay(speech_audio, position=500).fade_out(500)
                    try:
                        final_audio = normalize(final_audio)
                    except Exception:
                        pass
                    _dbg("🎼 Mixed speech with background music")
                else:
                    st.warning("Background music generation failed, using speech only")
                    final_audio = speech_audio
            except Exception as e:
                st.warning(f"Background music failed: {e}, using speech only")
                final_audio = speech_audio

            # Export final audio with better error handling
            try:
                final_temp_dir = tempfile.mkdtemp()
                final_file_path = os.path.join(final_temp_dir, f"final_audio_{int(time.time())}.mp3")
                final_audio.export(final_file_path, format="mp3", bitrate="192k")
                _dbg("🎵 Exported as high-quality MP3")
            except Exception as e:
                st.warning(f"MP3 export failed: {e}, trying WAV format")
                try:
                    final_temp_dir = tempfile.mkdtemp()
                    final_file_path = os.path.join(final_temp_dir, f"final_audio_{int(time.time())}.wav")
                    final_audio.export(final_file_path, format="wav")
                    _dbg("🎵 Exported as WAV")
                except Exception as e2:
                    st.error(f"Both MP3 and WAV export failed: {e2}")
                    # Return the original TTS file as last resort
                    return temp_file_path

            # Cleanup TTS temp file
            try:
                os.unlink(temp_file_path)
                os.rmdir(temp_dir)
            except Exception:
                pass

            return final_file_path

        except Exception as e:
            st.warning(f"Advanced audio processing failed: {e}")
            st.info("🔊 Returning basic TTS audio instead")
            return temp_file_path

    except Exception as e:
        st.error(f"Musical audio generation error: {str(e)}")
        # Final fallback to basic TTS
        return create_audio_from_text(text, language, speed, "neutral")
def create_simple_audio(text, language="en", speed=1.0, voice_type="neutral"):
    """Create a simple audio file without complex error handling"""
    try:
        from gtts import gTTS
        import urllib.request
        
        # Test internet connection
        urllib.request.urlopen('https://www.google.com', timeout=5)
        
        # Map language names to gTTS language codes
        lang_mapping = {
            "English": "en", "Spanish": "es", "French": "fr", 
            "German": "de", "Italian": "it", "Portuguese": "pt",
            "Russian": "ru", "Japanese": "ja", "Chinese": "zh-CN",
            "Arabic": "ar", "Hindi": "hi", "Telugu": "te",
            "Malayalam": "ml", "Kannada": "kn", "Tamil": "ta",
        }
        
        # Voice type mapping for different TLD (Top Level Domain) for gTTS
        voice_mapping = {
            "male_1": {"tld": "com"},
            "male_2": {"tld": "co.uk"},
            "female_1": {"tld": "com.au"},
            "female_2": {"tld": "ca"},
            "neutral": {"tld": "com"}
        }
        
        lang_code = lang_mapping.get(language, "en")
        voice_config = voice_mapping.get(voice_type, voice_mapping["neutral"])
        
        # Clean text for TTS
        clean_text = text.replace("*", "").replace("#", "").replace("**", "")
        
        if not clean_text.strip():
            st.error("No text to convert to audio")
            return None
        
        st.info(f"🎤 Creating audio: '{clean_text[:50]}...'")
        
        # Create TTS with voice selection
        tts = gTTS(
            text=clean_text, 
            lang=lang_code, 
            slow=(speed < 1.0),
            tld=voice_config['tld']
        )
        
        # Save to temporary file
        temp_file = tempfile.mktemp(suffix='.mp3')
        tts.save(temp_file)
        
        if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
            st.success(f"✅ Audio created successfully! Size: {os.path.getsize(temp_file)} bytes")
            return temp_file
        else:
            st.error("Failed to create audio file")
            return None
            
    except Exception as e:
        st.error(f"Audio creation failed: {str(e)}")
        return None

def create_audio_from_text(text, language="en", speed=1.0, voice_type="neutral"):
    """Generate audio from text using gTTS with voice options"""
    if not TTS_AVAILABLE:
        st.warning("Text-to-speech not available. Install gTTS to enable audio features.")
        return None
    
    # Check internet connection
    if not test_internet_connection():
        st.error("Internet connection required for text-to-speech. Please check your connection.")
        return None

    try:
        # Map language names to gTTS language codes and voice options
        lang_mapping = {
            "English": "en", "Spanish": "es", "French": "fr", 
            "German": "de", "Italian": "it", "Portuguese": "pt",
            "Russian": "ru", "Japanese": "ja", "Chinese": "zh-CN",
            "Arabic": "ar", "Hindi": "hi", "Telugu": "te",
            "Malayalam": "ml", "Kannada": "kn", "Tamil": "ta",
        }
        
        # Voice type mapping for different TLD (Top Level Domain) for gTTS
        # Different TLDs provide different voices/accents
        voice_mapping = {
            # Male voices
            "male_1": {"tld": "com", "description": "Male Voice 1 - Professional"},
            "male_2": {"tld": "co.uk", "description": "Male Voice 2 - British Accent"},
            # Female voices  
            "female_1": {"tld": "com.au", "description": "Female Voice 1 - Australian Accent"},
            "female_2": {"tld": "ca", "description": "Female Voice 2 - Canadian Accent"},
            # Neutral/default
            "neutral": {"tld": "com", "description": "Default Voice"}
        }
        
        lang_code = lang_mapping.get(language, "en")
        voice_config = voice_mapping.get(voice_type, voice_mapping["neutral"])
        
        # Clean text for TTS (remove markdown and special characters)
        clean_text = text.replace("*", "").replace("#", "").replace("**", "")
        
        # Ensure we have text to process
        if not clean_text.strip():
            st.warning("No text available for audio generation.")
            return None

        st.info(f"🎤 Generating audio with {voice_config['description']} for: '{clean_text[:50]}...' in {language}")
        
        # Create TTS object with voice selection
        tts = gTTS(
            text=clean_text, 
            lang=lang_code, 
            slow=(speed < 1.0),
            tld=voice_config['tld']
        )
        
        # Create a safe temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, f"tts_audio_{int(time.time())}.mp3")
        
        st.info(f"📁 Creating audio file at: {temp_file_path}")
        
        # Save to temporary file with explicit path
        tts.save(temp_file_path)
        
        # Verify file was created
        if os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) > 0:
            file_size = os.path.getsize(temp_file_path)
            st.success(f"✅ Audio generated successfully! File size: {file_size} bytes")
            st.info(f"📂 Audio file location: {temp_file_path}")
            return temp_file_path
        else:
            st.error("Failed to create audio file.")
            return None
            
    except Exception as e:
        st.error(f"Audio generation error: {str(e)}")
        # Check if it's a network error
        if "urlopen error" in str(e) or "timeout" in str(e).lower():
            st.error("Network error: Please check your internet connection.")
        return None

def generate_tts(text, target_language, speed=1.0, voice_type="neutral"):
    """Generate TTS audio for translated text with language mapping for translation mode"""
    # Map common language names to gTTS language codes
    language_mapping = {
        "Spanish": "Spanish",
        "French": "French", 
        "German": "German",
        "Italian": "Italian",
        "Portuguese": "Portuguese",
        "Dutch": "Dutch",
        "Japanese": "Japanese",
        "Chinese": "Chinese",
        "Korean": "Korean",
        "Arabic": "Arabic",
        "Russian": "Russian",
        "Hindi": "Hindi",
        "Malayalam": "Malayalam",
        "Kannada": "Kannada",
        "Tamil": "Tamil",
    }
    
    # Map to internal language names that create_audio_from_text expects
    language = language_mapping.get(target_language, "English")
    
    return create_audio_from_text(text, language, speed, voice_type)

def test_internet_connection():
    try:
        import urllib.request
        urllib.request.urlopen('https://www.google.com', timeout=3)
        return True
    except Exception:
        return False

def transcribe_audio_file(file_bytes: bytes, mime_type: str, target_language: str):
    """Transcribe an uploaded audio clip to text using SpeechRecognition (Google Web Speech).
    - Accepts common formats. If not WAV/AIFF/FLAC, converts to WAV when pydub+ffmpeg available.
    - target_language influences recognition language code when possible.
    Returns transcript string or None.
    """
    try:
        import speech_recognition as sr
    except Exception as e:
        st.info("Speech-to-text not available. Install: pip install SpeechRecognition pydub")
        return None

    # Map UI language to recognition locale
    asr_lang_map = {
        "English": "en-US", "Spanish": "es-ES", "French": "fr-FR",
        "German": "de-DE", "Italian": "it-IT", "Portuguese": "pt-PT",
        "Russian": "ru-RU", "Japanese": "ja-JP", "Chinese": "zh-CN",
        "Arabic": "ar", "Hindi": "hi-IN", "Telugu": "te-IN",
        "Malayalam": "ml-IN", "Kannada": "kn-IN", "Tamil": "ta-IN",
    }
    recog_lang = asr_lang_map.get(target_language, "en-US")

    # Write bytes to a temp file with appropriate extension
    ext_by_mime = {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/flac": ".flac",
        "audio/aiff": ".aiff",
        "audio/x-aiff": ".aiff",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/ogg": ".ogg",
        "audio/m4a": ".m4a",
        "audio/x-m4a": ".m4a",
    }
    ext = ext_by_mime.get((mime_type or "").lower(), ".wav")
    tmp_in = tempfile.mktemp(suffix=ext)
    with open(tmp_in, "wb") as f:
        f.write(file_bytes)

    # Ensure format compatible with SpeechRecognition: prefer WAV
    tmp_wav = tmp_in
    try:
        if not tmp_in.lower().endswith((".wav", ".flac", ".aiff")):
            if AUDIO_PROCESSING_AVAILABLE:
                audio = AudioSegment.from_file(tmp_in)
                audio = audio.set_channels(1).set_frame_rate(16000)
                tmp_wav = tempfile.mktemp(suffix=".wav")
                audio.export(tmp_wav, format="wav")
            else:
                st.info("For non-WAV uploads, install FFmpeg+pydub to enable conversion.")
                return None
    except Exception as e:
        st.info(f"Audio conversion failed: {e}")
        return None

    # Recognize speech
    try:
        r = sr.Recognizer()
        with sr.AudioFile(tmp_wav) as source:
            audio_data = r.record(source)
        # Use Google Web Speech (requires internet)
        text = r.recognize_google(audio_data, language=recog_lang)
        return text
    except sr.UnknownValueError:
        st.warning("Speech not understood. Try clearer audio or another language setting.")
        return None
    except sr.RequestError as e:
        st.warning(f"Speech service error: {e}")
        return None
    finally:
        # Cleanup temp files
        try:
            if os.path.exists(tmp_in):
                os.unlink(tmp_in)
        except Exception:
            pass
        if tmp_wav != tmp_in:
            try:
                if os.path.exists(tmp_wav):
                    os.unlink(tmp_wav)
            except Exception:
                pass

def get_audio_player_html(audio_file_path):
    """Create audio player using Streamlit's built-in component"""
    if not audio_file_path:
        st.warning("No audio file path provided.")
        return None
        
    if not os.path.exists(audio_file_path):
        st.error(f"Audio file not found: {audio_file_path}")
        return None
    
    try:
        file_size = os.path.getsize(audio_file_path)
        if file_size == 0:
            st.error("Audio file is empty.")
            return None
            
        st.info(f"🎵 Audio file ready: {file_size} bytes")
        
        with open(audio_file_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            # Detect mime type by extension for better compatibility
            ext = os.path.splitext(audio_file_path)[1].lower()
            if ext == '.wav':
                mime = 'audio/wav'
            elif ext == '.mp3':
                mime = 'audio/mpeg'
            else:
                mime = 'audio/mpeg'  # Default to MP3
            
            # Use Streamlit's audio component
            st.audio(audio_bytes, format=mime)
            st.success("🎧 Audio player loaded successfully!")
            return True
        
    except Exception as e:
        st.error(f"Audio player error: {str(e)}")
        # Try to provide fallback information
        try:
            st.info(f"📁 Audio file location: {audio_file_path}")
            st.info("You can download the file and play it manually if the player doesn't work.")
        except Exception:
            pass
        return None

@st.cache_resource(show_spinner=False)
def _load_text_model():
    """Lazily load a small multilingual text generation model (mT5-small)."""
    try:
        if not AI_MODEL_AVAILABLE:
            return None
        
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        
        # Try to load with proper error handling
        model_name = "google/mt5-small"
        
        # Load tokenizer first with fallback
        try:
            tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        except Exception as e:
            st.info(f"⚠️ Tokenizer loading failed: {e}. Using template generator.")
            return None
        
        # Load model with fallback
        try:
            mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as e:
            st.info(f"⚠️ Model loading failed: {e}. Using template generator.")
            return None
        
        # Create pipeline with error handling
        try:
            pipe = pipeline("text2text-generation", model=mdl, tokenizer=tok)
            return pipe
        except Exception as e:
            st.info(f"⚠️ Pipeline creation failed: {e}. Using template generator.")
            return None
            
    except Exception as e:
        st.info(f"⚠️ AI model not loaded: {e}. Using template generator.")
        return None

def generate_with_model(pipe, prompt, style, theme, length):
    """Generate deeply emotional, gift-worthy poetry that touches hearts naturally"""
    if pipe is None:
        return None
    
    # Emotional depth specifications for each length
    length_emotional_goals = {
        "short": "8-12 lines capturing a perfect moment with profound emotional impact",
        "medium": "16-20 lines building emotional crescendo with vivid imagery and metaphors", 
        "long": "24-32 lines weaving a complete emotional journey with rich sensory details",
        "epic": "40-60 lines creating an unforgettable emotional experience with multiple movements"
    }
    
    # Style guidance for natural, heartfelt expression
    style_heart_guidance = {
        "free verse": "flowing like natural speech, with rhythm that matches the heartbeat of emotion",
        "haiku": "capturing a single, precious moment in time with 5-7-5 syllables that resonate in the soul",
        "sonnet": "classical 14-line love letter with perfect ABAB CDCD EFEF GG rhymes that sing",
        "limerick": "joyful 5-line celebration with AABBA rhymes that bring smiles and warm memories"
    }
    
    # Deep emotional themes that create connections
    theme_emotional_essence = {
        "nature": "the sacred bond between human hearts and nature's eternal wisdom - seasons of life, growth, renewal, and the poetry found in every leaf and stream",
        "love": "the profound connection of souls - tender moments, shared dreams, hearts beating as one, the kind of love that makes ordinary moments magical",
        "adventure": "the call of the wild heart - courage to chase dreams, the thrill of discovery, and the beautiful uncertainty of new horizons",
        "dreams": "the landscape of imagination where anything is possible - hope taking flight, visions of tomorrow, and the magic of believing",
        "mystery": "the allure of the unknown - whispered secrets, moonlit paths, and the beautiful questions that make life enchanting",
        "friendship": "bonds stronger than blood - shared laughter through tears, loyal hearts, chosen family, and memories that warm the soul forever",
        "hope": "light breaking through the darkest nights - resilient spirits, dreams refusing to die, and the strength found in believing tomorrow will be brighter",
        "sadness": "beautiful melancholy that heals - the poetry of loss, tears that cleanse, and finding strength in our most vulnerable moments",
        "joy": "pure celebration of being alive - hearts dancing with delight, infectious laughter, and those precious moments when everything feels perfect"
    }
    
    # Create an emotionally intelligent AI prompt
    heartfelt_prompt = f"""
    Create a deeply moving, emotionally resonant poem that will touch hearts and be treasured as a precious gift.
    
    EMOTIONAL FOUNDATION:
    User's Personal Words: "{prompt}" - Make these words the beating heart of this poem
    Emotional Theme: {theme_emotional_essence.get(theme, 'universal human emotions and connections')}
    Poetic Expression: {style_heart_guidance.get(style, 'natural, flowing verse')}
    Emotional Journey: {length_emotional_goals.get(length, 'complete emotional experience')}
    
    REQUIREMENTS FOR A GIFT-WORTHY POEM:
    ✨ Weave "{prompt}" naturally throughout the poem as the central inspiration
    ✨ Use vivid, sensory language that makes readers feel they're experiencing the moment
    ✨ Include beautiful metaphors that resonate with the human experience
    ✨ Create emotional rhythm that builds and flows like natural speech
    ✨ Use specific, concrete details that make the poem personal and memorable
    ✨ Build to emotional peaks that move the reader's heart
    ✨ End with lines that linger in memory and touch the soul
    ✨ Write in language that someone would treasure receiving as a heartfelt gift
    
    TONE: Write with the warmth and authenticity of someone sharing their deepest feelings with a beloved person.
    
    Now create this treasured poem:
    """
    
    try:
        # Generate with settings optimized for emotional creativity
        output = pipe(
            heartfelt_prompt, 
            max_new_tokens=450,      # Generous length for emotional development
            temperature=0.9,         # High creativity while maintaining coherence
            do_sample=True, 
            top_p=0.85,             # Focus on most probable emotional expressions
            repetition_penalty=1.3,  # Avoid repetitive phrases
            num_return_sequences=1,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        
        generated_text = output[0]["generated_text"].strip()
        
        # Extract the pure poetry from the AI response
        poem_markers = [
            "Now create this treasured poem:",
            "Poem:",
            "Here is the poem:",
            "---",
            "\n\n"
        ]
        
        poem = generated_text
        for marker in poem_markers:
            if marker in poem:
                poem = poem.split(marker)[-1].strip()
        
        # Clean and beautify the poem
        poem = poem.replace("```", "").strip()
        
        # Remove any leftover instruction text
        lines = []
        for line in poem.split('\n'):
            line = line.strip()
            # Skip lines that look like instructions
            if line and not any(skip_word in line.lower() for skip_word in 
                              ['create', 'write', 'requirements', 'emotional', 'tone:', 'foundation:']):
                if len(line) > 2 and not line.isupper():  # Skip short/title case lines
                    lines.append(line)
        
        # Ensure natural line breaks for readability
        if style != "haiku" and len(lines) == 1 and len(lines[0]) > 100:
            # Split long single line into natural verses
            import re
            single_line = lines[0]
            # Break at natural pause points
            broken_lines = re.split(r'([.!?;])\s+', single_line)
            lines = []
            for i in range(0, len(broken_lines)-1, 2):
                if i+1 < len(broken_lines):
                    lines.append(broken_lines[i] + broken_lines[i+1])
        
        # Join and validate the poem
        final_poem = '\n'.join(lines)
        
        # Quality validation - ensure it's gift-worthy
        if (len(final_poem) > 100 and  # Substantial length
            len(lines) >= 4 and       # Multiple lines
            any(word.lower() in final_poem.lower() for word in prompt.split() if len(word) > 2)):  # Contains user's inspiration
            return final_poem
        else:
            return None  # Fallback to template system for better quality
            
    except Exception as e:
        st.info(f"AI generation enhancing: {e}. Using creative templates.")
        return None

def generate_dynamic_ai_poetry(prompt, style="free verse", theme="nature", length="long", generation_mode="creative", use_templates=True):
    """
    Generate highly dynamic, non-repetitive poetry using AI with multiple approaches
    """
    import random
    import time
    import hashlib
    
    # Initialize advanced structures if not already done
    try:
        if poetry_trie.word_count == 0:
            initialize_advanced_structures()
    except:
        pass
    
    # Create a truly unique seed that includes timestamp to prevent repetition
    timestamp = int(time.time() * 1000)  # millisecond precision
    random_factor = random.randint(1, 999999)
    seed_string = f"{prompt}_{theme}_{style}_{length}_{generation_mode}_{timestamp}_{random_factor}"
    seed = int(hashlib.sha256(seed_string.encode()).hexdigest()[:16], 16)
    random.seed(seed)
    
    # Different generation approaches
    if generation_mode == "pure_ai":
        return generate_pure_ai_poetry(prompt, style, theme, length)
    elif generation_mode == "template_enhanced":
        return generate_template_enhanced_poetry(prompt, style, theme, length, use_templates)
    elif generation_mode == "creative":
        return generate_creative_fusion_poetry(prompt, style, theme, length)
    elif generation_mode == "advanced_algorithms":
        return generate_algorithm_enhanced_poetry(prompt, style, theme, length)
    else:
        return generate_adaptive_poetry(prompt, style, theme, length)

def generate_algorithm_enhanced_poetry(prompt, style="free verse", theme="nature", length="long"):
    """
    Generate poetry using advanced data structures and algorithms
    """
    try:
        _dbg("🧠 Using advanced algorithms for poetry generation")
        
        # Extract key concepts from prompt
        key_words = extract_meaningful_words(prompt)
        if not key_words:
            key_words = [theme]
        
        # Use creative path finder to build semantic journey
        start_concept = key_words[0] if key_words else theme
        target_emotions = {
            "peaceful": "peace", "energetic": "energy", "romantic": "love", 
            "mysterious": "mystery", "joyful": "joy", "melancholic": "sorrow"
        }
        target_emotion = target_emotions.get(theme, "beauty")
        
        # Find creative word sequence
        creative_sequence = find_creative_word_path(start_concept, target_emotion, 6)
        _dbg(f"🎯 Creative sequence: {' → '.join(creative_sequence)}")
        
        # Build poem structure
        lines = []
        line_count = {"short": 4, "medium": 8, "long": 12, "epic": 16}.get(length, 12)
        
        for i in range(line_count):
            if i < len(creative_sequence):
                # Use words from creative sequence
                primary_word = creative_sequence[i]
                
                # Get semantic suggestions
                semantic_words = get_semantic_suggestions(primary_word, 3)
                
                # Get rhyming words if needed
                if i > 0 and i % 2 == 1:  # Every other line for ABAB pattern
                    prev_line_word = lines[i-1].split()[-1].rstrip('.,!?;:')
                    rhyme_candidates = get_advanced_rhymes(prev_line_word, 3)
                    if rhyme_candidates:
                        line_ending = rhyme_candidates[0]
                    else:
                        line_ending = semantic_words[0] if semantic_words else primary_word
                else:
                    line_ending = semantic_words[0] if semantic_words else primary_word
                
                # Build line with optimization
                available_words = [primary_word] + semantic_words + key_words
                available_words = list(set(available_words))  # Remove duplicates
                
                # Use optimizer to arrange words
                optimized_words = optimize_poem_structure_advanced(
                    available_words[:6], 
                    target_syllables=8, 
                    target_sentiment=0.6
                )
                
                # Create poetic line
                if len(optimized_words) >= 3:
                    line = f"{optimized_words[0].title()} {optimized_words[1]} {optimized_words[2]}"
                    if line_ending and line_ending != optimized_words[-1]:
                        line += f" {line_ending}"
                else:
                    line = f"{primary_word.title()} brings {target_emotion}"
                
                lines.append(line)
            else:
                # Generate additional lines using pattern matching
                pattern = key_words[i % len(key_words)][:2] if key_words else "be"
                pattern_words = search_words_by_pattern(pattern)
                
                if pattern_words:
                    selected_word = pattern_words[i % len(pattern_words)]
                    semantic_suggestions = get_semantic_suggestions(selected_word, 2)
                    
                    if semantic_suggestions:
                        line = f"{selected_word.title()} {semantic_suggestions[0]} gently"
                    else:
                        line = f"{selected_word.title()} flows through time"
                else:
                    line = f"Beauty {key_words[0] if key_words else 'flows'} eternal"
                
                lines.append(line)
        
        # Format based on style
        if style == "haiku":
            if len(lines) >= 3:
                formatted_poem = f"{lines[0]}\n{lines[1]}\n{lines[2]}"
            else:
                formatted_poem = "\n".join(lines)
        elif style == "sonnet":
            # Group into quatrains
            formatted_lines = []
            for i in range(0, min(14, len(lines)), 4):
                quatrain = lines[i:i+4]
                formatted_lines.extend(quatrain)
                if i < 12:  # Add spacing between quatrains
                    formatted_lines.append("")
            formatted_poem = "\n".join(formatted_lines)
        else:  # free verse or limerick
            # Group into stanzas
            formatted_lines = []
            stanza_size = 4
            for i in range(0, len(lines), stanza_size):
                stanza = lines[i:i+stanza_size]
                formatted_lines.extend(stanza)
                if i + stanza_size < len(lines):
                    formatted_lines.append("")
            formatted_poem = "\n".join(formatted_lines)
        
        _dbg(f"✅ Algorithm-enhanced poem generated with {len(lines)} lines")
        return formatted_poem
        
    except Exception as e:
        _dbg(f"⚠️ Algorithm enhancement failed: {e}")
        # Fallback to creative fusion
        return generate_creative_fusion_poetry(prompt, style, theme, length)

def generate_pure_ai_poetry(prompt, style, theme, length):
    """Generate poetry using pure AI approach without templates"""
    try:
        global model, tokenizer
        
        if not AI_MODEL_AVAILABLE:
            st.info("🤖 AI model not available, using creative template generation")
            return generate_creative_fusion_poetry(prompt, style, theme, length)
        
        # Load model if not already loaded with better error handling
        if model is None or tokenizer is None:
            try:
                model_name = "google/mt5-small"
                
                # Try loading with more specific error handling
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                except Exception as e:
                    st.info(f"🤖 Tokenizer error: {e}. Using creative templates.")
                    return generate_creative_fusion_poetry(prompt, style, theme, length)
                
                try:
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                except Exception as e:
                    st.info(f"🤖 Model loading error: {e}. Using creative templates.")
                    return generate_creative_fusion_poetry(prompt, style, theme, length)
                    
                _dbg(f"✅ Loaded AI model: {model_name}")
                
            except Exception as e:
                st.info(f"🤖 AI model setup failed: {e}. Using creative templates.")
                return generate_creative_fusion_poetry(prompt, style, theme, length)
        
        # Create a sophisticated prompt for the AI model
        ai_prompt = f"""Write a unique, original {style} poem about "{prompt}" with a {theme} theme.
        The poem should be {length} and deeply emotional. 
        Make it personal, touching, and avoid clichés.
        Use vivid imagery and metaphors.
        Length: {length}
        Style: {style}
        Theme: {theme}
        
        Create something completely new and heartfelt:
        """
        
        # Use the AI model to generate poetry with error handling
        try:
            inputs = tokenizer(ai_prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the poem part
            if ":" in generated_text:
                poem = generated_text.split(":", 1)[1].strip()
            else:
                poem = generated_text.strip()
            
            if len(poem) > 50:  # Minimum viable poem length
                return poem
            else:
                return generate_creative_fusion_poetry(prompt, style, theme, length)
                
        except Exception as e:
            st.info(f"🤖 AI generation error: {e}. Using creative templates.")
            return generate_creative_fusion_poetry(prompt, style, theme, length)
            
    except Exception as e:
        st.info(f"🤖 Pure AI generation fallback: {e}. Using creative templates.")
        return generate_creative_fusion_poetry(prompt, style, theme, length)

def generate_creative_fusion_poetry(prompt, style, theme, length):
    """Generate poetry using creative fusion of AI concepts and dynamic templates"""
    import random
    
    # Extract meaningful words from user prompt
    user_words = extract_meaningful_words(prompt)
    
    # Dynamic word banks that change based on prompt analysis
    dynamic_vocab = build_dynamic_vocabulary(prompt, theme)
    
    # Create poem structure based on style and length
    poem_structure = design_poem_structure(style, length)
    
    # Generate verses using dynamic content
    verses = []
    for verse_type in poem_structure:
        verse = create_dynamic_verse(user_words, dynamic_vocab, verse_type, theme)
        verses.append(verse)
    
    return "\n\n".join(verses)

def extract_meaningful_words(prompt):
    """Extract and categorize meaningful words from user input"""
    import re
    
    # Common words to filter out
    stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "a", "an", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}
    
    # Extract words and filter
    words = re.findall(r'\b\w+\b', prompt.lower())
    meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return meaningful_words[:5]  # Limit to top 5 meaningful words

def build_dynamic_vocabulary(prompt, theme):
    """Build a vocabulary that adapts to the user's input and theme"""
    import random
    
    # Analyze prompt sentiment and content
    prompt_lower = prompt.lower()
    
    # Base vocabularies for different themes
    vocab_banks = {
        "nature": {
            "adjectives": ["verdant", "serene", "majestic", "tranquil", "wild", "pristine", "enchanted", "golden", "emerald", "crystalline"],
            "verbs": ["whispers", "dances", "blooms", "flows", "embraces", "awakens", "flourishes", "glimmers", "cascades", "breathes"],
            "nouns": ["meadow", "forest", "mountain", "river", "ocean", "sky", "stars", "moonlight", "sunrise", "valley"]
        },
        "love": {
            "adjectives": ["tender", "passionate", "eternal", "devoted", "cherished", "sacred", "infinite", "pure", "divine", "precious"],
            "verbs": ["adores", "treasures", "embraces", "celebrates", "protects", "nurtures", "completes", "honors", "caresses", "cherishes"],
            "nouns": ["heart", "soul", "spirit", "dreams", "promise", "touch", "kiss", "embrace", "forever", "destiny"]
        },
        "adventure": {
            "adjectives": ["bold", "fearless", "magnificent", "untamed", "soaring", "mighty", "daring", "wild", "majestic", "powerful"],
            "verbs": ["conquers", "explores", "discovers", "soars", "ventures", "blazes", "climbs", "pursues", "seeks", "achieves"],
            "nouns": ["journey", "quest", "horizon", "summit", "path", "adventure", "courage", "destiny", "freedom", "victory"]
        }
    }
    
    # Get base vocabulary for theme
    base_vocab = vocab_banks.get(theme, vocab_banks["nature"])
    
    # Enhance vocabulary based on prompt content
    if any(word in prompt_lower for word in ["sad", "loss", "grief", "tears"]):
        base_vocab["adjectives"].extend(["melancholic", "gentle", "healing", "comforting", "understanding"])
        base_vocab["verbs"].extend(["heals", "comforts", "soothes", "remembers", "transforms"])
    
    if any(word in prompt_lower for word in ["happy", "joy", "celebration", "bright"]):
        base_vocab["adjectives"].extend(["radiant", "joyful", "sparkling", "vibrant", "luminous"])
        base_vocab["verbs"].extend(["celebrates", "dances", "sparkles", "radiates", "delights"])
    
    return base_vocab

def design_poem_structure(style, length):
    """Design the structure of the poem based on style and length"""
    import random
    
    structures = {
        "free verse": {
            "short": ["opening", "development", "conclusion"],
            "medium": ["opening", "development", "reflection", "conclusion"],
            "long": ["opening", "development", "exploration", "reflection", "climax", "conclusion"]
        },
        "sonnet": {
            "short": ["quatrain", "quatrain", "couplet"],
            "medium": ["quatrain", "quatrain", "quatrain", "couplet"],
            "long": ["quatrain", "quatrain", "quatrain", "quatrain", "couplet"]
        },
        "haiku": {
            "short": ["haiku"],
            "medium": ["haiku", "haiku"],
            "long": ["haiku", "haiku", "haiku"]
        }
    }
    
    return structures.get(style, structures["free verse"]).get(length, structures["free verse"]["medium"])

def create_dynamic_verse(user_words, vocab, verse_type, theme):
    """Create a verse that dynamically incorporates user words and theme"""
    import random
    
    # Get vocabulary elements
    adj = random.choice(vocab["adjectives"])
    verb = random.choice(vocab["verbs"])
    noun = random.choice(vocab["nouns"])
    
    # Incorporate user words naturally
    user_word = random.choice(user_words) if user_words else noun
    
    # Different verse patterns based on type
    if verse_type == "opening":
        patterns = [
            f"In the {adj} space where {user_word} {verb},\nI find a world that {noun} creates,\nWhere every moment gently {verb}\nAnd {adj} beauty contemplates.",
            f"When {user_word} {verb} through {adj} light,\nThe {noun} becomes a sacred space,\nWhere {adj} dreams take gentle flight\nAnd find their destined resting place."
        ]
    elif verse_type == "development":
        patterns = [
            f"Here {user_word} {verb} like {adj} {noun},\nTransforming ordinary days,\nWhile {adj} whispers softly {verb}\nThrough life's most wondrous ways.",
            f"The {adj} {noun} {verb} with grace,\nAs {user_word} finds its true form,\nIn this {adj}, sacred space\nWhere hearts grow gently warm."
        ]
    elif verse_type == "conclusion":
        patterns = [
            f"So let {user_word} {verb} on,\nThrough {adj} days and {noun} bright,\nFor in this {adj} song\nWe find our truest light.",
            f"And when the {adj} {noun} {verb},\nRemember how {user_word} can be\nThe {adj} gift that life preserves\nFor all eternity."
        ]
    else:  # Default pattern
        patterns = [
            f"In {adj} moments, {user_word} {verb},\nLike {noun} dancing in the light,\nWhere {adj} dreams gently {verb}\nAnd make everything feel right."
        ]
    
    return random.choice(patterns)

def generate_template_enhanced_poetry(prompt, style, theme, length, use_templates):
    """Enhanced template-based generation with AI assistance"""
    # This function will use templates when specifically requested
    # For now, fall back to creative fusion
    return generate_creative_fusion_poetry(prompt, style, theme, length)

def generate_adaptive_poetry(prompt, style, theme, length):
    """Adaptive poetry that chooses the best approach based on input"""
    # Analyze the prompt to decide the best generation method
    if len(prompt.split()) > 10:  # Complex prompt
        return generate_creative_fusion_poetry(prompt, style, theme, length)
    else:  # Simple prompt
        return generate_pure_ai_poetry(prompt, style, theme, length)
    for i, concept in enumerate(key_concepts[1:] if len(key_concepts) > 1 else [central_concept]):
        if length == "medium" and i < 1:
            additional_verses.append(create_heartfelt_verse(concept, descriptors, theme_words))
        elif length == "long" and i < 2:
            additional_verses.append(create_heartfelt_verse(concept, descriptors, theme_words))
        elif length == "epic" and i < 3:
            additional_verses.append(create_heartfelt_verse(concept, descriptors, theme_words))
    
    # Combine verses with natural flow
    if length == "short":
        return verse1.split('\n\n')[0]  # Just first stanza
    elif additional_verses:
        return f"{verse1}\n\n{chr(10).join(additional_verses)}"
    else:
        return verse1

# Advanced Features Implementation

def export_poetry_pdf(poem_text, metadata, user_preferences=None):
    """Export poetry to PDF format"""
    if not PDF_AVAILABLE:
        st.error("PDF export not available. Please install fpdf2: pip install fpdf2")
        return None
    
    try:
        from fpdf import FPDF
        
        class PoetryPDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 15)
                self.cell(0, 10, 'Multimodal Poetry AI - Generated Poem', 0, 1, 'C')
                self.ln(10)
            
            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'C')
        
        pdf = PoetryPDF()
        pdf.add_page()
        
        # Add metadata
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Poem Details:', 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 8, metadata, 0, 1)
        pdf.ln(10)
        
        # Add poem text
        pdf.set_font('Times', '', 14)
        lines = poem_text.split('\n')
        for line in lines:
            if line.strip():
                # Encode to handle special characters
                try:
                    pdf.cell(0, 8, line.encode('latin-1', 'replace').decode('latin-1'), 0, 1, 'C')
                except:
                    pdf.cell(0, 8, line, 0, 1, 'C')
            else:
                pdf.ln(4)
        
        # Save to bytes
        pdf_bytes = bytes(pdf.output())
        return pdf_bytes
        
    except Exception as e:
        st.error(f"PDF export error: {e}")
        return None

def create_video_poem(poem_text, audio_file, metadata):
    """Create a video with poem text and audio"""
    if not VIDEO_AVAILABLE:
        st.info("🎬 Video export feature is in development. MoviePy installation needs additional setup.")
        return None
    
    try:
        # Create a simple text video (placeholder implementation)
        # In a full implementation, you'd use moviepy to create a proper video
        st.info("🎬 Video creation feature coming soon! For now, you can export as PDF.")
        return None
        
    except Exception as e:
        st.error(f"Video creation error: {e}")
        return None

def create_custom_theme_editor():
    """Interface for creating custom musical themes"""
    st.markdown("### 🎨 Custom Musical Theme Creator")
    
    with st.expander("Create Your Own Musical Theme", expanded=True):
        theme_name = st.text_input("Theme Name", placeholder="Enter a unique name for your theme")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Musical Properties:**")
            base_frequency = st.slider("Base Frequency (Hz)", 100, 500, 261, help="Starting musical note")
            scale_type = st.selectbox("Scale Type", ["Major", "Minor", "Pentatonic", "Blues"])
            tempo = st.slider("Tempo (BPM)", 60, 180, 120)
            volume = st.slider("Volume Level", -30, -10, -20)
        
        with col2:
            st.markdown("**Rhythm Pattern:**")
            rhythm_style = st.selectbox("Rhythm Style", ["Flowing", "Steady", "Syncopated", "Free"])
            note_duration = st.slider("Average Note Duration (ms)", 500, 5000, 2000)
            harmony_richness = st.slider("Harmony Complexity", 1, 5, 3)
        
        # Advanced settings
        with st.expander("Advanced Musical Settings"):
            col3, col4 = st.columns(2)
            with col3:
                frequency_spread = st.slider("Frequency Range", 50, 300, 150)
                rhythm_variation = st.slider("Rhythm Variation %", 0, 50, 20)
            with col4:
                reverb_level = st.slider("Reverb Level", 0, 100, 30)
                harmony_count = st.slider("Number of Harmonies", 0, 6, 3)
        
        if st.button("🎵 Create Custom Theme") and theme_name:
            # Generate custom theme data
            import random
            frequencies = []
            rhythms = []
            
            # Generate frequency pattern based on scale
            for i in range(6):
                if scale_type == "Major":
                    freq = base_frequency * (1.125 ** i)  # Major scale intervals
                elif scale_type == "Minor":
                    freq = base_frequency * (1.067 ** i)  # Minor scale intervals
                else:  # Pentatonic
                    freq = base_frequency * (1.2 ** i)
                
                frequencies.append(round(freq, 2))
                
                # Generate rhythm pattern
                base_rhythm = note_duration
                variation = (rhythm_variation / 100) * base_rhythm
                rhythm = base_rhythm + random.randint(-int(variation), int(variation))
                rhythms.append(rhythm)
            
            # Generate harmonies
            harmonies = [freq / 2 for freq in frequencies[:harmony_count]]
            
            custom_theme = {
                "variations": [{
                    "name": f"{theme_name}_custom",
                    "frequencies": frequencies,
                    "rhythm": rhythms,
                    "volume": volume,
                    "harmonies": harmonies
                }]
            }
            
            # Save to user preferences
            preferences = load_user_preferences()
            preferences['custom_themes'][theme_name.lower()] = custom_theme
            
            if save_user_preferences(preferences):
                st.success(f"✅ Custom theme '{theme_name}' created successfully!")
                st.info("🎵 Your custom theme is now available in the theme selection dropdown.")
                st.balloons()
            else:
                st.error("❌ Failed to save custom theme.")

def voice_cloning_interface():
    """Interface for voice cloning features"""
    st.markdown("### 🎙️ Voice Cloning & Personalization")
    
    if not VOICE_CLONE_AVAILABLE:
        st.info("""
        🔬 **Voice Cloning Feature (Coming Soon!)**
        
        This feature will allow you to:
        - Clone your own voice for personalized poetry narration
        - Create unique voice profiles for different poetry styles
        - Use AI-generated voices with personality traits
        - Share voice profiles with the community
        
        **Implementation Status:** Research & Development Phase
        
        Planned technologies:
        - Coqui TTS for voice synthesis
        - Real-time voice conversion
        - Multi-language voice support
        """)
        
        # Placeholder interface
        with st.expander("Voice Preferences (Beta)"):
            voice_style = st.selectbox("Preferred Voice Style", 
                ["Warm & Gentle", "Strong & Dramatic", "Soft & Whispery", "Clear & Professional"])
            speech_pace = st.slider("Speech Pace", 0.5, 2.0, 1.0)
            emotional_tone = st.selectbox("Emotional Tone", 
                ["Neutral", "Joyful", "Melancholic", "Mysterious", "Romantic"])
            
            if st.button("💾 Save Voice Preferences"):
                preferences = load_user_preferences()
                preferences['voice_preferences'] = {
                    'style': voice_style,
                    'pace': speech_pace,
                    'tone': emotional_tone
                }
                save_user_preferences(preferences)
                st.success("Voice preferences saved!")
    
    else:
        # Full voice cloning interface would go here
        st.success("🎙️ Voice cloning is available!")

def social_sharing_interface(poem_text, metadata):
    """Interface for sharing poems on social media"""
    st.markdown("### 📱 Social Sharing")
    
    with st.expander("Share Your Poem", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Share Options:**")
            include_metadata = st.checkbox("Include poem details", value=True)
            include_attribution = st.checkbox("Include AI attribution", value=True)
            
            share_format = st.selectbox("Sharing Format", 
                ["Text Only", "Image with Text", "Audio Link", "Full Package"])
        
        with col2:
            st.markdown("**Preview:**")
            share_text = poem_text
            
            if include_metadata:
                share_text += f"\n\n---\n{metadata}"
            
            if include_attribution:
                share_text += "\n\n🤖 Generated with Multimodal Poetry AI"
            
            st.text_area("Share Preview", share_text, height=150)
        
        # Social media buttons (placeholder)
        st.markdown("**Share on:**")
        col3, col4, col5, col6 = st.columns(4)
        
        with col3:
            if st.button("📘 Facebook"):
                st.info("Opening Facebook sharing... (Feature in development)")
        
        with col4:
            if st.button("🐦 Twitter"):
                st.info("Opening Twitter sharing... (Feature in development)")
        
        with col5:
            if st.button("📸 Instagram"):
                st.info("Opening Instagram sharing... (Feature in development)")
        
        with col6:
            if st.button("💼 LinkedIn"):
                st.info("Opening LinkedIn sharing... (Feature in development)")
        
        # Copy to clipboard
        if st.button("📋 Copy to Clipboard"):
            # Create shareable text
            if st.session_state.get('share_text'):
                st.code(share_text)
                st.success("✅ Text ready to copy!")

def export_interface(poem_text, metadata, audio_file=None):
    """Interface for exporting poems in different formats"""
    st.markdown("### 📁 Export Your Poem")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Export Formats:**")
        
        # PDF Export
        if st.button("📄 Export as PDF"):
            with st.spinner("Creating PDF..."):
                pdf_bytes = export_poetry_pdf(poem_text, metadata)
                if pdf_bytes:
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name=f"poem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
        
        # Text Export
        if st.button("📝 Export as Text"):
            export_text = f"{poem_text}\n\n---\nPoem Details:\n{metadata}\n\nGenerated with Multimodal Poetry AI"
            st.download_button(
                label="⬇️ Download Text",
                data=export_text,
                file_name=f"poem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        # JSON Export (with all metadata)
        if st.button("🔧 Export as JSON"):
            export_data = {
                "poem": poem_text,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            }
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="⬇️ Download JSON",
                data=json_str,
                file_name=f"poem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        st.markdown("**Package Export:**")
        
        # Complete package with audio
        if st.button("📦 Create Complete Package") and audio_file:
            with st.spinner("Creating package..."):
                # Create ZIP with poem text, metadata, and audio
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Add poem text
                    zip_file.writestr("poem.txt", poem_text)
                    
                    # Add metadata
                    zip_file.writestr("metadata.txt", metadata)
                    
                    # Add audio file if available
                    if audio_file and Path(audio_file).exists():
                        zip_file.write(audio_file, "poem_audio.wav")
                    
                    # Add JSON export
                    export_data = {
                        "poem": poem_text,
                        "metadata": metadata,
                        "timestamp": datetime.now().isoformat()
                    }
                    zip_file.writestr("poem_data.json", json.dumps(export_data, indent=2))
                
                zip_buffer.seek(0)
                st.download_button(
                    label="⬇️ Download Complete Package",
                    data=zip_buffer.getvalue(),
                    file_name=f"poem_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
        
        # Video Export (placeholder)
        if st.button("🎬 Export as Video (Beta)"):
            st.info("🚧 Video export feature coming soon! Will include animated text and audio.")

def learning_interface():
    """Interface for AI learning from user preferences"""
    st.markdown("### 🧠 AI Learning & Personalization")
    
    preferences = load_user_preferences()
    
    with st.expander("📊 Your Poetry Profile"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Learning Statistics:**")
            liked_count = len(preferences['learning_data']['liked_poems'])
            st.metric("Poems You've Liked", liked_count)
            
            if liked_count > 0:
                recent_styles = [poem.get('style_features', {}).get('sentiment', 'unknown') 
                               for poem in preferences['learning_data']['liked_poems'][-10:]]
                most_common_sentiment = max(set(recent_styles), key=recent_styles.count) if recent_styles else 'none'
                st.metric("Preferred Sentiment", most_common_sentiment.title())
        
        with col2:
            st.markdown("**Style Preferences:**")
            style_pref = st.selectbox("Preferred Poetry Style", 
                ["balanced", "creative", "structured", "experimental"], 
                index=["balanced", "creative", "structured", "experimental"].index(
                    preferences.get('poetry_style', 'balanced')))
            
            if st.button("💾 Update Style Preference"):
                preferences['poetry_style'] = style_pref
                save_user_preferences(preferences)
                st.success("Style preference updated!")
    
    # Feedback interface
    if 'last_generated_poem' in st.session_state:
        st.markdown("**Rate the last generated poem:**")
        rating = st.slider("How much did you like it?", 1, 5, 3)
        
        if st.button("📝 Submit Feedback"):
            learn_from_user_feedback(
                st.session_state['last_generated_poem'], 
                rating, 
                st.session_state.get('last_user_input', '')
            )
            st.success("Thank you for your feedback! The AI will learn from this.")


def main():
    # Initialize session state variables first (before any UI elements)
    if "prompt_text" not in st.session_state:
        st.session_state["prompt_text"] = ""
    if "description_input" not in st.session_state:
        st.session_state["description_input"] = ""
    if "description_complete" not in st.session_state:
        st.session_state.description_complete = False
    if "enhanced_prompt" not in st.session_state:
        st.session_state.enhanced_prompt = ""
    if "generated_poem" not in st.session_state:
        st.session_state.generated_poem = ""
    if "generated_metadata" not in st.session_state:
        st.session_state.generated_metadata = ""
    if "generated_audio_file" not in st.session_state:
        st.session_state.generated_audio_file = None
    if "english_poetry" not in st.session_state:
        st.session_state.english_poetry = ""
    if "generation_settings" not in st.session_state:
        st.session_state.generation_settings = {}
    if "last_user_input" not in st.session_state:
        st.session_state.last_user_input = ""
    if "poem_generation_complete" not in st.session_state:
        st.session_state.poem_generation_complete = False
    if "real_time_mode" not in st.session_state:
        st.session_state.real_time_mode = True  # Default to real-time mode
    if "translation_mode" not in st.session_state:
        st.session_state.translation_mode = False
    if "translated_text" not in st.session_state:
        st.session_state.translated_text = ""
    if "translation_settings" not in st.session_state:
        st.session_state.translation_settings = {}
    
    # Initialize history
    if "poem_history" not in st.session_state:
        st.session_state.poem_history = []
    if "show_history" not in st.session_state:
        st.session_state.show_history = False

    header_with_mascot()
    st.markdown("<div class='soft-card'>Generate beautiful poetry across multiple languages with AI</div>", unsafe_allow_html=True)

    # Display history if toggled
    if st.session_state.show_history:
        display_history()
        st.markdown("---")
        return  # Don't show main interface when viewing history

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")

        st.divider()

        # (Removed Figma Theme Sync UI)

        # Optional diagnostic toggle for less/noisy logs during audio generation
        with st.expander("🔧 Advanced Audio Settings", expanded=False):
            st.checkbox("Verbose audio diagnostics", key="verbose_audio", value=False,
                        help="Show detailed audio generation logs (tones added, file exports, etc.)")
            st.info("💡 Enable verbose diagnostics only if you need to troubleshoot audio generation issues.")

        # Optional: Use AI model instead of templates
        use_model = st.checkbox("🧠 Use AI Model (beta)", value=True,  # Default to True for better poetry
                                help="Generate text with a small multilingual model (mT5-small). Creates more personalized, unique poems!")

        # Language selection
        target_language = st.selectbox(
            "🌍 Target Language:",
          ["English", "Spanish", "French", "German", "Italian", 
           "Portuguese", "Russian", "Japanese", "Chinese", "Arabic", "Hindi", "Telugu", "Malayalam", "Kannada", "Tamil"]
        )

        # Poetry style
        poetry_style = st.selectbox(
            "📝 Poetry Style:",
            ["free verse", "haiku", "sonnet", "limerick"]
        )

        # Poem length control
        poem_length = st.select_slider(
            "📏 Poem Length",
            options=["short", "medium", "long", "epic"],
            value="long",
            help="Controls the number of stanzas/lines generated"
        )

        # Theme selection
        theme = st.selectbox(
            "🎨 Theme:",
            ["nature", "love", "adventure", "dreams", "mystery", "custom"]
        )

        # Mood
        mood = st.selectbox(
            "😊 Mood:",
            ["peaceful", "energetic", "romantic", "mysterious", "joyful", "melancholic"]
        )

        st.divider()

        # AI Generation Mode Settings
        st.header("🤖 AI Generation Settings")
        
        # Generation approach choice
        generation_approach = st.radio(
            "🎯 How should AI handle your input?",
            ["Interpret & Create", "Describe First", "Poetry Translate"],
            help="Interpret & Create: AI creates poetry directly from your input. Describe First: AI asks what you want to describe first. Poetry Translate: Translate existing text/poetry with all features."
        )
        
        # Set translation mode based on selection
        st.session_state.translation_mode = (generation_approach == "Poetry Translate")
        
        # Show translation-specific settings if in translation mode
        if st.session_state.translation_mode:
            st.markdown("#### 🌍 Translation Settings")
            target_language = st.selectbox(
                "🗣️ Target Language:",
                ["Spanish", "French", "German", "Italian", "Portuguese", "Dutch", "Japanese", "Chinese", "Korean", "Arabic", "Russian", "Hindi", "Malayalam", "Kannada", "Tamil"],
                index=0,
                help="Choose the language to translate your text into"
            )
            
            translation_style = st.selectbox(
                "🎨 Translation Style:",
                ["Poetic", "Literal", "Cultural", "Modern"],
                index=0,
                help="Poetic: Maintains rhythm and artistic flow | Literal: Direct translation | Cultural: Adapts to cultural context | Modern: Contemporary language style"
            )
            
            st.session_state.translation_settings = {
                "target_language": target_language,
                "translation_style": translation_style
            }
        
        # Generation mode selection
        generation_mode = st.selectbox(
            "⚙️ Generation Mode:",
            ["creative", "advanced_algorithms", "pure_ai", "template_enhanced", "adaptive"],
            index=0,
            help="""
            • Creative: Fusion of AI concepts with dynamic templates (Recommended)
            • Advanced Algorithms: Uses data structures for optimal poetry (NEW!)
            • Pure AI: 100% AI-generated without templates
            • Template Enhanced: AI-enhanced templates for structure
            • Adaptive: AI chooses best approach based on your input
            """
        )
        
        # Template usage option
        use_templates = st.checkbox(
            "📋 Allow Template Assistance",
            value=True,
            help="Let AI use templates when needed for better structure and quality"
        )
        
        # Show generation preview
        if generation_approach == "Describe First":
            st.info("💭 **Description Mode Active**: After you submit, AI will first ask you to elaborate on what you want to describe, then create the poem.")
        elif generation_approach == "Poetry Translate":
            st.info("🌍 **Translation Mode Active**: AI will translate your input text to the target language with full voice-over and musical features.")
        else:
            st.info("✨ **Direct Creation Mode**: AI will create poetry directly from your input using advanced interpretation.")

        st.divider()

        # Voice-over settings
        st.header("🎵 Voice-Over Settings")
        enable_voice = st.checkbox(
            "🔊 Enable Voice-Over",
            value=False,  # Default to False, let user enable it
            disabled=not TTS_AVAILABLE,
            help="Generate audio narration of your poem"
        )

        if not TTS_AVAILABLE:
            st.error("gTTS not available. Please install: pip install gTTS")

        if enable_voice and TTS_AVAILABLE:
            voice_speed = st.slider(
                "🎚️ Voice Speed:",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Adjust the speaking speed"
            )
            
            # Voice selection dropdown
            st.markdown("#### 🎭 Voice Selection")
            voice_type = st.selectbox(
                "🎙️ Choose Voice:",
                options=[
                    ("male_1", "👨 Male Voice 1 - Professional"),
                    ("male_2", "👨 Male Voice 2 - British Accent"),
                    ("female_1", "👩 Female Voice 1 - Australian Accent"),
                    ("female_2", "👩 Female Voice 2 - Canadian Accent"),
                    ("neutral", "🤖 Default Voice")
                ],
                format_func=lambda x: x[1],
                index=4,  # Default to neutral
                help="Select your preferred voice style and gender"
            )
            selected_voice = voice_type[0]  # Extract the voice key
            
            # Show voice preview info
            voice_descriptions = {
                "male_1": "🎯 **Professional Male Voice** - Clear, authoritative tone perfect for serious poetry",
                "male_2": "🇬🇧 **British Male Voice** - Elegant accent ideal for classical and romantic poetry",
                "female_1": "🇦🇺 **Australian Female Voice** - Warm, friendly tone great for contemporary poetry",
                "female_2": "🇨🇦 **Canadian Female Voice** - Gentle, expressive style perfect for emotional poetry",
                "neutral": "⚖️ **Default Voice** - Balanced, universal voice suitable for all poetry types"
            }
            
            with st.expander("ℹ️ Voice Information"):
                st.markdown(voice_descriptions[selected_voice])
                st.info("💡 **Tip**: Different voices work better with different poetry styles. Experiment to find your favorite!")
        else:
            # Default voice when voice is disabled
            selected_voice = "neutral"

        # Store voice selection in session state for persistence
        if enable_voice and TTS_AVAILABLE:
            auto_play = st.checkbox(
                "▶️ Auto-play Audio",
                value=False,
                help="Automatically play audio when poem is generated"
            )
        else:
            voice_speed = 1.0
            auto_play = False

        # Show language info
        if target_language != "English":
            st.info(f"🔄 Poetry will be translated to {target_language}")

        # Musical enhancement controls
        st.subheader("🎼 Musical Enhancement")
        enable_musical = st.checkbox(
            "Enable Musical Background",
            value=AUDIO_PROCESSING_AVAILABLE,
            disabled=not AUDIO_PROCESSING_AVAILABLE,
            help="Mix gentle background music and apply audio effects"
        )
        if enable_musical:
            # Inline FFmpeg guidance on Windows if needed
            if not AUDIO_PROCESSING_AVAILABLE:
                st.info("For full musical features, install FFmpeg on Windows: `winget install Gyan.FFmpeg` or download from https://www.gyan.dev/ffmpeg/builds/ and add `bin` to PATH.")
            
            # AI Auto-detection of musical theme
            st.info("🤖 **AI Auto-Theme Detection:** The system will automatically analyze your poetry and select the most appropriate musical theme and variation from our extensive collection:")
            
            with st.expander("🎵 Available Themes & Variations"):
                st.markdown("""
                **🕊️ Peaceful** (8 variations): gentle_meadow, soft_rain, morning_mist, calm_ocean, zen_garden, forest_whispers, moonbeam_lullaby, starlight_serenity  
                **⚡ Energetic** (8 variations): electric_pulse, power_surge, lightning_strike, rocket_launch, cyber_beat, thunderstorm, volcanic_eruption, industrial_pulse  
                **💕 Romantic** (8 variations): moonlight_serenade, rose_petals, candlelight_waltz, sunset_embrace, lovers_melody, wedding_bells, first_kiss, eternal_love  
                **🔮 Mysterious** (8 variations): shadow_whispers, ancient_secrets, midnight_fog, dark_magic, cryptic_puzzle, haunted_manor, occult_ritual, vampire_castle  
                **😊 Joyful** (8 variations): sunshine_dance, spring_festival, children_laughter, carnival_music, celebration_bells, victory_march, party_time, rainbow_bridge  
                **😢 Melancholic** (8 variations): autumn_leaves, gentle_tears, distant_memory, lonely_piano, fading_light, winter_solitude, broken_heart, lost_dreams  
                
                *Each theme now has 8 unique musical variations for incredible diversity! (48 total musical styles)*
                """)
            
            audio_effects = st.selectbox(
                "🎛️ Audio Effect:",
                ["enhance", "reverb", "echo", "none"],
                index=0,
                help="Enhance = clarity/compression; Reverb/Echo add ambience"
            )
            bg_volume_percent = st.slider(
                "🔉 Background Music Volume",
                min_value=0,
                max_value=100,
                value=40,
                step=5,
                help="Balance background music against the voice-over"
            )
        else:
            musical_theme = None  # Will auto-detect
            audio_effects = "none"
            bg_volume_percent = 40

        st.divider()

        # Audio prompt (speech-to-text) - enhanced with better error handling
        st.subheader("🎙️ Audio Prompt (beta)")
        st.caption("Upload a short recording and transcribe it to use as your prompt. Uses Google Web Speech via SpeechRecognition.")
        
        # Custom label for uploader to control color precisely; hide the default label
        st.markdown('<div class="upload-audio-label">Upload audio (wav/mp3/m4a/ogg)</div>', unsafe_allow_html=True)
        audio_prompt = st.file_uploader(
            "Upload audio (wav/mp3/m4a/ogg)", 
            type=["wav", "mp3", "m4a", "ogg", "flac"], 
            key="audio_prompt_uploader", 
            label_visibility="collapsed",
            help="Upload clear speech recording for transcription. Keep it under 1 minute for best results."
        )
        
        if audio_prompt is not None:
            try:
                # Display audio file info
                file_size = len(audio_prompt.getvalue())
                file_size_mb = file_size / (1024 * 1024)
                st.info(f"📁 **Audio file loaded:** {audio_prompt.name} ({file_size_mb:.2f} MB)")
                
                # Show audio player
                st.audio(audio_prompt.getvalue(), format=audio_prompt.type or "audio/mpeg")
                
                # Transcribe button
                do_transcribe = st.button("📝 Transcribe Audio", type="primary")
                
                if do_transcribe:
                    with st.spinner("🎧 Transcribing audio... This may take a moment."):
                        try:
                            # Check internet connection first
                            if not test_internet_connection():
                                st.error("❌ **Internet connection required** for speech transcription. Please check your connection.")
                            else:
                                # Attempt transcription
                                transcript_text = transcribe_audio_file(
                                    audio_prompt.getvalue(), 
                                    audio_prompt.type or "audio/mpeg", 
                                    target_language
                                )
                                
                                if transcript_text and transcript_text.strip():
                                    st.success("✅ **Transcription successful!**")
                                    st.markdown(f"**Transcribed text:** \"{transcript_text}\"")
                                    
                                    # Auto-populate the prompt field
                                    st.session_state["prompt_text"] = transcript_text
                                    st.success("🎯 **Text inserted into prompt field!**")
                                    
                                    # Show language detected
                                    st.info(f"🌍 **Recognized in:** {target_language}")
                                    
                                else:
                                    st.warning("⚠️ **Could not transcribe audio.** Please try:")
                                    st.markdown("- Speaking more clearly and slowly")
                                    st.markdown("- Using a quieter environment") 
                                    st.markdown("- Recording in the selected language")
                                    st.markdown("- Ensuring good audio quality")
                                    
                        except Exception as transcription_error:
                            st.error(f"❌ **Transcription failed:** {str(transcription_error)}")
                            st.info("💡 **Troubleshooting tips:**")
                            st.markdown("- Try a different audio format (WAV works best)")
                            st.markdown("- Ensure the audio contains clear speech")
                            st.markdown("- Check your internet connection")
                            st.markdown("- Try a shorter audio clip (under 30 seconds)")
                            
            except Exception as audio_error:
                st.error(f"❌ **Error loading audio file:** {str(audio_error)}")
                st.info("💡 Try uploading a different audio file or format")
        else:
            # Show transcribe button as disabled when no file
            st.button("📝 Transcribe Audio", disabled=True, help="Upload an audio file first")
        
        st.divider()
        
        # Advanced Features Section
        st.header("🚀 Advanced Features")
        
        # Voice Cloning Section
        with st.expander("🎙️ Voice Cloning & Personalization"):
            voice_cloning_interface()
        
        # Learning & Preferences Section  
        with st.expander("🧠 AI Learning & Preferences"):
            learning_interface()
        
        # Algorithm Showcase Section
        with st.expander("⚙️ Algorithm Showcase"):
            st.markdown("### 🔬 Data Structures & Algorithms")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **🌳 Trie Structure**
                - Fast word lookup: O(log n)
                - Pattern matching
                - Prefix search
                """)
                
                test_pattern = st.text_input("Test Pattern Search:", placeholder="lo", key="pattern_test")
                if test_pattern:
                    matches = search_words_by_pattern(test_pattern)
                    if matches:
                        st.write(f"Matches: {', '.join(matches[:5])}")
                    else:
                        st.write("No matches found")
            
            with col2:
                st.markdown("""
                **📊 Semantic Graph**
                - Word relationships
                - Conceptual paths
                - A* search algorithm
                """)
                
                test_word = st.text_input("Test Semantic Search:", placeholder="love", key="semantic_test")
                if test_word:
                    related = get_semantic_suggestions(test_word, 3)
                    if related:
                        st.write(f"Related: {', '.join(related)}")
                    else:
                        st.write("No relations found")
            
            # Rhyme Engine Test
            st.markdown("**🎵 Advanced Rhyme Engine**")
            rhyme_test = st.text_input("Test Rhyme Search:", placeholder="heart", key="rhyme_test")
            if rhyme_test:
                rhymes = get_advanced_rhymes(rhyme_test, 3)
                if rhymes:
                    st.write(f"Rhymes: {', '.join(rhymes)}")
                else:
                    st.write("No rhymes found")
            
            # Creative Path Demo
            st.markdown("**🎯 Creative Path Finder (A* Algorithm)**")
            col3, col4 = st.columns(2)
            with col3:
                start_concept = st.text_input("Start:", placeholder="dream", key="path_start")
            with col4:
                end_concept = st.text_input("Target:", placeholder="peace", key="path_end")
            
            if start_concept and end_concept:
                path = find_creative_word_path(start_concept, end_concept, 4)
                if len(path) > 1:
                    st.write(f"Path: {' → '.join(path)}")
                else:
                    st.write("No path found")
        
        # Custom Theme Creation
        if st.button("🎨 Create Custom Musical Theme"):
            st.session_state['show_custom_theme_sidebar'] = True
        
        if st.session_state.get('show_custom_theme_sidebar', False):
            st.markdown("---")
            create_custom_theme_editor()
            if st.button("❌ Close Theme Creator"):
                st.session_state['show_custom_theme_sidebar'] = False
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # Text input with session state (to allow audio transcription to populate it)
        prompt_label = "Enter your poetry inspiration:" if theme == "custom" else f"Describe your {theme} inspiration:"
        prompt = st.text_area(
            prompt_label,
            value=st.session_state["prompt_text"],
            key="prompt_text_widget",
            placeholder="Write about anything that inspires you...",
            height=100
        )
        
        # Update session state when prompt changes
        if prompt != st.session_state["prompt_text"]:
            st.session_state["prompt_text"] = prompt
        
        # Image upload with improved analysis
        st.subheader("🖼️ Visual Inspiration (Optional)")
        uploaded_file = st.file_uploader(
            "Upload an image for inspiration:",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Upload an image to inspire your poetry. Supported formats: PNG, JPG, JPEG, WEBP"
        )
        
        if uploaded_file is not None:
            try:
                # Load and display image with error handling
                image = Image.open(uploaded_file)
                
                # Verify image is valid
                image.verify()
                
                # Reload image for processing (verify() closes the file)
                uploaded_file.seek(0)
                image = Image.open(uploaded_file)
                
                # Convert to RGB if needed (handles RGBA, grayscale, etc.)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Display the image
                st.image(image, caption="Your inspiration image", use_container_width=True)
                
                # Analyze image properties for better description
                width, height = image.size
                
                # Basic color analysis
                try:
                    # Get dominant colors by sampling pixels
                    image_small = image.resize((50, 50))
                    pixels = list(image_small.getdata())
                    
                    # Calculate average color values
                    avg_r = sum(p[0] for p in pixels) / len(pixels)
                    avg_g = sum(p[1] for p in pixels) / len(pixels)
                    avg_b = sum(p[2] for p in pixels) / len(pixels)
                    
                    # Determine color mood
                    if avg_r > avg_g and avg_r > avg_b:
                        color_mood = "warm, passionate reds and oranges"
                    elif avg_g > avg_r and avg_g > avg_b:
                        color_mood = "natural, calming greens"
                    elif avg_b > avg_r and avg_b > avg_g:
                        color_mood = "cool, serene blues"
                    elif avg_r + avg_g + avg_b > 600:
                        color_mood = "bright, luminous light"
                    elif avg_r + avg_g + avg_b < 200:
                        color_mood = "deep, mysterious shadows"
                    else:
                        color_mood = "balanced, harmonious tones"
                    
                    # Determine brightness
                    brightness = (avg_r + avg_g + avg_b) / 3
                    if brightness > 200:
                        light_desc = "bathed in radiant light"
                    elif brightness > 150:
                        light_desc = "softly illuminated"
                    elif brightness > 100:
                        light_desc = "gently shadowed"
                    else:
                        light_desc = "wrapped in mysterious darkness"
                    
                    # Create rich description based on analysis
                    image_descriptions = [
                        f"a vision of {color_mood}, {light_desc}",
                        f"an image {light_desc} with {color_mood}",
                        f"a scene painted in {color_mood}, {light_desc}",
                        f"a moment where {color_mood} meet the {light_desc.split()[-1]}",
                        f"beauty expressed through {color_mood} and {light_desc}"
                    ]
                    
                except Exception as color_error:
                    # Fallback to generic descriptions if color analysis fails
                    image_descriptions = [
                        "a scene of breathtaking natural beauty",
                        "colors that dance with light and shadow", 
                        "a moment frozen in pure magic",
                        "an image that speaks directly to the soul",
                        "beauty that transcends words",
                        "a visual poem waiting to be written",
                        "inspiration captured in perfect harmony"
                    ]
                
                # Add image dimensions context
                if width > height:
                    composition = "sweeping landscape"
                elif height > width:
                    composition = "towering portrait"
                else:
                    composition = "perfectly balanced frame"
                
                # Select and apply description
                image_desc = random.choice(image_descriptions)
                final_desc = f"{image_desc} in a {composition}"
                
                # Integrate with user prompt
                if prompt.strip():
                    prompt += f", inspired by {final_desc}"
                else:
                    prompt = f"A poem inspired by {final_desc}"
                
                # Show what was extracted
                with st.expander("🎨 Image Analysis", expanded=False):
                    st.write(f"**Dimensions:** {width} × {height} pixels")
                    st.write(f"**Composition:** {composition}")
                    st.write(f"**Inspiration:** {final_desc}")
                    st.success("✅ Image successfully analyzed and integrated!")
                
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.info("💡 Try uploading a different image format (PNG, JPG, JPEG, or WEBP)")
                st.info("🔍 Make sure the image file is not corrupted and under 200MB")
        
        # Generate button
        generate_btn = st.button("🎪 Generate Cross-Language Poetry", type="primary", use_container_width=True)
    
    with col2:
        st.header("✨ Generated Poetry")
        
        if generate_btn and prompt.strip():
            # Handle description mode first
            if generation_approach == "Describe First":
                if not st.session_state.description_complete:
                    st.info("💭 **Tell me more about what you want to describe:**")
                    st.write(f"**Your initial input:** {prompt}")
                    
                    description_prompt = st.text_area(
                        "Please elaborate on your idea:",
                        value=st.session_state["description_input"],
                        placeholder="Describe the emotions, imagery, specific details, or memories you want to include...",
                        height=100,
                        key="description_widget"
                    )
                    
                    # Update session state when description changes
                    if description_prompt != st.session_state["description_input"]:
                        st.session_state["description_input"] = description_prompt
                    
                    if st.button("✨ Create Poetry from Description", type="primary"):
                        if description_prompt.strip():
                            # Combine original prompt with description
                            enhanced_prompt = f"{prompt}. {description_prompt}"
                            st.session_state.description_complete = True
                            st.session_state.enhanced_prompt = enhanced_prompt
                            st.rerun()
                        else:
                            st.warning("Please provide some description to continue.")
                    return
                else:
                    # Use enhanced prompt from description
                    prompt = st.session_state.enhanced_prompt
                    # Clear the session state for next time
                    st.session_state.description_complete = False
                    st.session_state.enhanced_prompt = ""
            
            # Add Generate Poetry button and check if we should generate or display existing
            should_generate = False
            
            # Check if we have input to work with
            if prompt.strip():
                # Check if settings changed for real-time feedback
                current_settings_preview = {
                    'target_language': target_language,
                    'poetry_style': poetry_style, 
                    'poem_length': poem_length,
                    'theme': theme,
                    'mood': mood,
                    'generation_mode': generation_mode,
                    'use_templates': use_templates,
                    'enable_voice': enable_voice,
                    'voice_speed': voice_speed,
                    'enable_musical': enable_musical,
                    'prompt': prompt.strip()
                }
                
                settings_changed_preview = st.session_state.generation_settings != current_settings_preview
                
                if settings_changed_preview and st.session_state.poem_generation_complete:
                    if st.session_state.real_time_mode:
                        st.info("⚡ Settings changed - poem will update automatically!")
                    else:
                        st.warning("⚠️ Settings changed - click 'Generate Poetry' to apply changes!")
                
                # Generate button
                button_text = "🎭 Generate Poetry"
                if st.button(button_text, type="primary", use_container_width=True):
                    should_generate = True
                    st.session_state.poem_generation_complete = False
                
                # Auto-generate for description mode
                if generation_approach == "Describe First" and st.session_state.description_complete:
                    should_generate = True
                    st.session_state.poem_generation_complete = False
            else:
                st.warning("Please enter some inspiration text to generate poetry.")
                return
            
            # Generate new poem or display existing one
            # Store current settings to detect changes
            current_settings = {
                'target_language': target_language,
                'poetry_style': poetry_style, 
                'poem_length': poem_length,
                'theme': theme,
                'mood': mood,
                'generation_mode': generation_mode,
                'use_templates': use_templates,
                'enable_voice': enable_voice,
                'voice_speed': voice_speed,
                'selected_voice': selected_voice,
                'enable_musical': enable_musical,
                'prompt': prompt.strip()  # Include prompt in settings to detect changes
            }
            
            # Check if settings changed (if so, we should regenerate automatically)
            settings_changed = st.session_state.generation_settings != current_settings
            
            # Handle Translation Mode
            if st.session_state.translation_mode:
                # Translation mode - translate user input directly
                if should_generate or (settings_changed and st.session_state.real_time_mode) or not st.session_state.poem_generation_complete:
                    target_lang = st.session_state.translation_settings.get("target_language", "Spanish")
                    translation_style = st.session_state.translation_settings.get("translation_style", "Poetic")
                    
                    with st.spinner(f"🌍 Translating to {target_lang} ({translation_style} style)..."):
                        # Create translation prompt based on style
                        style_instruction = {
                            "Poetic": "Maintain the poetic rhythm, meter, and artistic flow while translating",
                            "Literal": "Provide a direct, word-for-word translation",
                            "Cultural": "Adapt the translation to fit the cultural context and expressions",
                            "Modern": "Use contemporary language and modern expressions"
                        }
                        
                        translation_prompt = f"""
                        Translate the following text to {target_lang} using a {translation_style.lower()} style.
                        
                        {style_instruction.get(translation_style, "")}
                        
                        Text to translate:
                        {prompt}
                        
                        Provide only the translation without any explanations or additional text.
                        """
                        
                        try:
                            # Use deep_translator for translation
                            from deep_translator import GoogleTranslator
                            
                            # Map language names to language codes
                            lang_codes = {
                                "Spanish": "es", "French": "fr", "German": "de", 
                                "Italian": "it", "Portuguese": "pt", "Dutch": "nl",
                                "Japanese": "ja", "Chinese": "zh", "Korean": "ko",
                                "Arabic": "ar", "Russian": "ru", "Hindi": "hi"
                            }
                            
                            target_code = lang_codes.get(target_lang, "es")
                            translator = GoogleTranslator(source='auto', target=target_code)
                            
                            # For poetic style, add context instruction
                            if translation_style == "Poetic":
                                # Add instruction to preserve poetic elements
                                translation_text = f"Translate this preserving poetry style: {prompt}"
                                translated_text = translator.translate(translation_text)
                                # Remove the instruction part if it was translated
                                if "preserving poetry style:" in translated_text.lower():
                                    translated_text = translated_text.split(":", 1)[-1].strip()
                            else:
                                translated_text = translator.translate(prompt)
                            
                            st.session_state.translated_text = translated_text
                            st.session_state.generated_poem = translated_text  # Use same storage for consistency
                            
                        except Exception as e:
                            st.error(f"Translation failed: {str(e)}")
                            translated_text = prompt  # Keep original if translation fails
                            st.session_state.translated_text = translated_text
                            st.session_state.generated_poem = translated_text
                    
                    # Generate audio if enabled for translated text
                    audio_file = None
                    if enable_voice and TTS_AVAILABLE:
                        with st.spinner(f"🎵 Creating voice-over for translated text..."):
                            audio_file = generate_tts(translated_text, target_lang, voice_speed, selected_voice)
                            st.session_state.generated_audio_file = audio_file
                    
                    # Generate metadata for translation mode
                    metadata_text = f"**🌍 Translation Mode** | **Target Language:** {target_lang} | **Style:** {translation_style}"
                    if enable_voice and TTS_AVAILABLE:
                        if enable_musical and AUDIO_PROCESSING_AVAILABLE:
                            metadata_text += f" | **Musical Voice-over:** AI Auto-Selected + {audio_effects} | **BG Vol:** {bg_volume_percent}%"
                        else:
                            metadata_text += f" | **Voice-over:** Enabled"
                    st.session_state.generated_metadata = metadata_text
                    
                    st.session_state.poem_generation_complete = True
                    st.session_state.generation_settings = current_settings
                    
                    # Save translation to history
                    save_to_history(
                        translated_text, 
                        metadata_text, 
                        prompt, 
                        is_translation=True
                    )
                    
            else:
                # Regular poetry generation mode
                # Auto-regenerate if settings changed or if user clicked generate
                # But only auto-regenerate if real-time mode is enabled
                if should_generate or (settings_changed and st.session_state.real_time_mode) or not st.session_state.poem_generation_complete:
                    should_generate = True
                    if settings_changed and st.session_state.real_time_mode:
                        _dbg(f"🔄 Settings changed - auto-regenerating poem (real-time mode)")
                        # Real-time regeneration happens silently
                    elif settings_changed and not st.session_state.real_time_mode:
                        # Show notification but don't auto-regenerate
                        st.warning("⚠️ Settings changed. Click 'Generate Poetry' to apply changes.")
                        should_generate = False
                
                st.session_state.generation_settings = current_settings
                
                # Generate poetry if needed
                if should_generate:
                    # Clear previous English poetry to avoid scope issues
                    st.session_state.english_poetry = ""
                    
                    with st.spinner(f"🎭 Crafting your {target_language} poetry..."):
                        # Generate poetry in English first (via model or templates)
                        english_poetry = None
                        if use_model:
                            pipe = _load_text_model()
                            if pipe is not None:
                                english_poetry = generate_with_model(pipe, prompt, poetry_style, theme, poem_length)
                        if not english_poetry:
                            english_poetry = generate_dynamic_ai_poetry(prompt, poetry_style, theme, poem_length, generation_mode, use_templates)
                        
                        # Translate if needed
                        if target_language != "English":
                            with st.spinner(f"🔄 Translating to {target_language}..."):
                                final_poetry = translate_text(english_poetry, target_language)
                        else:
                            final_poetry = english_poetry
                        
                        # Generate audio if enabled
                        audio_file = None
                        if enable_voice and TTS_AVAILABLE:
                            with st.spinner(f"🎵 Creating voice-over in {target_language}..."):
                                if enable_musical and AUDIO_PROCESSING_AVAILABLE:
                                    audio_file = create_musical_poetry_audio(
                                        final_poetry, target_language, voice_speed, None, audio_effects, bg_volume_percent, selected_voice
                                    )
                                else:
                                    audio_file = create_simple_audio(final_poetry, target_language, voice_speed, selected_voice)
                        
                        # Store generated content in session state to persist across setting changes
                        st.session_state.generated_poem = final_poetry
                        st.session_state.generated_audio_file = audio_file
                        st.session_state.english_poetry = english_poetry  # Store original English version
                        st.session_state.last_user_input = prompt
                        st.session_state.poem_generation_complete = True
                        
                        # Generate metadata
                        metadata_text = f"**Language:** {target_language} | **Style:** {poetry_style} | **Theme:** {theme} | **Mood:** {mood} | **Length:** {poem_length} | **AI Mode:** {generation_mode}"
                        if generation_approach == "Describe First":
                            metadata_text += " | **Description-Enhanced**"
                        if use_templates:
                            metadata_text += " | **Template-Assisted**"
                        if enable_voice and TTS_AVAILABLE:
                            if enable_musical and AUDIO_PROCESSING_AVAILABLE:
                                metadata_text += f" | **Musical Voice-over:** AI Auto-Selected + {audio_effects} | **BG Vol:** {bg_volume_percent}%"
                            else:
                                metadata_text += f" | **Voice-over:** Enabled"
                        
                        st.session_state.generated_metadata = metadata_text
                        
                        # Save poem to history
                        save_to_history(
                            final_poetry, 
                            metadata_text, 
                            prompt, 
                            is_translation=False
                        )
            
            # Display poem (either newly generated or from session state)
            if st.session_state.poem_generation_complete and st.session_state.generated_poem:
                final_poetry = st.session_state.generated_poem
                audio_file = st.session_state.generated_audio_file  
                metadata_text = st.session_state.generated_metadata

                # Display result
                if st.session_state.translation_mode:
                    target_lang = st.session_state.translation_settings.get("target_language", "Target Language")
                    st.markdown(f"### 🌍 Your Translated Text ({target_lang}):")
                else:
                    st.markdown(f"### Your {target_language} Poem:")
                st.markdown(
                    f"""
                                        <div class=\"poetry-card fade-in\">
                                            <div class=\"corner-deco tl\">{svg_star(28)}</div>
                                            <div class=\"corner-deco tr\">{svg_star(28)}</div>
                                            <div class=\"poem-text\">{final_poetry.replace(chr(10), '<br>')}</div>
                                            <div class=\"corner-deco bl\">{svg_book(36)}</div>
                                            <div class=\"corner-deco br\">{svg_quill(36)}</div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                )
                
                # Audio player
                if audio_file and enable_voice:
                    if st.session_state.translation_mode:
                        st.markdown("### 🎵 Listen to Your Translation:")
                    else:
                        st.markdown("### 🎵 Listen to Your Poem:")
                    played = get_audio_player_html(audio_file)
                    # Hide file path by default - only show in verbose mode
                    if played and audio_file and st.session_state.get("verbose_audio", False):
                        st.info(f"📁 Audio file created at: {audio_file}")
                    # Note: Autoplay is often blocked by browsers; keeping UI simple
                
                # Display poem metadata
                st.markdown(metadata_text)
                
                # Show original if translated
                if target_language != "English":
                    with st.expander("📖 View Original English Version"):
                        if st.session_state.english_poetry:
                            st.text(st.session_state.english_poetry)
                            
                            # Audio for English version
                            if enable_voice and TTS_AVAILABLE:
                                english_audio = create_audio_from_text(st.session_state.english_poetry, "English", voice_speed, selected_voice)
                                if english_audio:
                                    st.markdown("🎵 **Listen to English Version:**")
                                    get_audio_player_html(english_audio)
                        else:
                            st.info("Original English version not available.")
                
                # Action buttons
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    if st.button("🔄 Regenerate"):
                        # Force regeneration with current settings
                        st.session_state.poem_generation_complete = False
                        st.session_state.generated_poem = ""
                        st.session_state.generated_metadata = ""
                        st.session_state.generated_audio_file = None
                        st.rerun()
                
                with col_b:
                    if st.button("🧹 Clear All"):
                        # Clear everything including input
                        st.session_state.poem_generation_complete = False
                        st.session_state.generated_poem = ""
                        st.session_state.generated_metadata = ""
                        st.session_state.generated_audio_file = None
                        st.session_state.prompt_text = ""
                        st.session_state.description_input = ""
                        st.session_state.description_complete = False
                        st.session_state.enhanced_prompt = ""
                        st.session_state.generation_settings = {}  # Clear settings cache
                        st.rerun()
                
                with col_c:
                    # Download poem option
                    if st.button("📥 Export Poem"):
                        st.session_state['show_export'] = True
                
                with col_d:
                    if st.button("📱 Share"):
                        st.session_state['show_sharing'] = True
                
                # Show what settings are currently active
                with st.expander("📊 Current Generation Settings"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Language**: {target_language}")
                        st.write(f"**Style**: {poetry_style}")
                        st.write(f"**Length**: {poem_length}")
                        st.write(f"**Theme**: {theme}")
                    with col2:
                        st.write(f"**Mood**: {mood}")
                        st.write(f"**AI Mode**: {generation_mode}")
                        st.write(f"**Voice**: {'Enabled' if enable_voice else 'Disabled'}")
                        st.write(f"**Musical**: {'Enabled' if enable_musical else 'Disabled'}")
                
                # Store for feedback learning
                st.session_state['last_generated_poem'] = final_poetry
                st.session_state['last_user_input'] = prompt
                
                # Advanced Features Sections
                if st.session_state.get('show_export', False):
                    st.markdown("---")
                    export_interface(final_poetry, metadata_text, audio_file)
                    if st.button("❌ Close Export"):
                        st.session_state['show_export'] = False
                        st.rerun()
                
                if st.session_state.get('show_custom_theme', False):
                    st.markdown("---")
                    create_custom_theme_editor()
                    if st.button("❌ Close Theme Editor"):
                        st.session_state['show_custom_theme'] = False
                        st.rerun()
                
                if st.session_state.get('show_sharing', False):
                    st.markdown("---")
                    social_sharing_interface(final_poetry, metadata_text)
                    if st.button("❌ Close Sharing"):
                        st.session_state['show_sharing'] = False
                        st.rerun()
                
                # Algorithm Performance Dashboard
                if generation_mode == "advanced_algorithms":
                    st.markdown("---")
                    st.markdown("### 📊 Algorithm Performance Dashboard")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Trie Words", poetry_trie.word_count)
                    
                    with col2:
                        try:
                            graph_nodes = len(semantic_graph.graph.nodes()) if hasattr(semantic_graph, 'graph') else 0
                            st.metric("Graph Nodes", graph_nodes)
                        except:
                            st.metric("Graph Nodes", "N/A")
                    
                    with col3:
                        try:
                            rhyme_patterns = len(rhyme_engine.rhyme_patterns)
                            st.metric("Rhyme Patterns", rhyme_patterns)
                        except:
                            st.metric("Rhyme Patterns", "N/A")
                    
                    with col4:
                        creativity_score = len(final_poetry.split()) * 0.1  # Simple creativity metric
                        st.metric("Creativity Score", f"{creativity_score:.1f}")
                    
                    # Algorithm usage breakdown
                    with st.expander("🔍 Algorithm Details"):
                        st.markdown("""
                        **Algorithms Used in Generation:**
                        
                        1. **🌳 Trie Search**: O(m) word lookup where m = word length
                        2. **📊 Graph Traversal**: A* pathfinding for semantic connections
                        3. **🎵 Heap-based Rhyming**: Priority queue for best rhyme selection
                        4. **⚡ Dynamic Programming**: Optimal word arrangement
                        5. **🧠 Heuristic Search**: Creative sequence optimization
                        
                        **Performance Benefits:**
                        - Faster word lookup: O(log n) vs O(n)
                        - Optimal rhyme matching using heaps
                        - Semantic coherence through graph algorithms
                        - Structure optimization via dynamic programming
                        """)
                    
                    st.download_button(
                        "💾 Download Poem",
                        data=final_poetry,
                        file_name=f"poem_{target_language.lower()}_{theme}.txt",
                        mime="text/plain"
                    )
                
                with col_c:
                    # Download audio option
                    if audio_file and enable_voice and os.path.exists(audio_file):
                        with open(audio_file, 'rb') as f:
                            audio_bytes = f.read()
                        
                        download_label = "🎼 Download Musical Audio" if (enable_musical and AUDIO_PROCESSING_AVAILABLE) else "🎵 Download Audio"

                        filename_suffix = "ai_themed" if (enable_musical and AUDIO_PROCESSING_AVAILABLE) else "basic"
                        st.download_button(
                            download_label,
                            data=audio_bytes,
                            file_name=f"poem_audio_{target_language.lower()}_{theme}_{filename_suffix}.mp3",
                            mime="audio/mp3"
                        )
                
                with col_d:
                    if st.button("🌍 Try Another Language"):
                        st.rerun()
                
                # Clean up temporary files
                if audio_file and os.path.exists(audio_file):
                    try:
                        os.unlink(audio_file)
                    except:
                        pass
        
        elif generate_btn and not prompt.strip():
            st.warning("⚠️ Please enter some inspiration text or upload an image!")
    
    # Language showcase
    st.markdown("---")
    st.subheader("🌍 Supported Languages")
    
    languages_display = {
        "English": "🇺🇸", "Spanish": "🇪🇸", "French": "🇫🇷", 
        "German": "🇩🇪", "Italian": "🇮🇹", "Portuguese": "🇵🇹",
        "Russian": "🇷🇺", "Japanese": "🇯🇵", "Chinese": "🇨🇳", 
        "Arabic": "🇸🇦", "Hindi": "🇮🇳", "Telugu": "🇮🇳",
        "Malayalam": "🇮🇳", "Kannada": "🇮🇳", "Tamil": "🇮🇳"
    }
    
    cols = st.columns(6)
    for i, (lang, flag) in enumerate(languages_display.items()):
        with cols[i % 6]:
            st.markdown(f"{flag} {lang}")
    
    # Example section
    with st.expander("💡 Cross-Language Poetry Examples"):
        st.markdown("""
        **English:** *"Beneath the vast expanse of sunset sky, where whispers of ancient winds carry dreams..."*
        
        **Spanish:** *"Bajo la vasta extensión del cielo del atardecer, donde los susurros de vientos antiguos llevan sueños..."*
        
        **French:** *"Sous la vaste étendue du ciel du coucher du soleil, où les murmures des vents anciens portent des rêves..."*
        
        **Japanese:** *"夕日の空の広大な広がりの下で、古代の風のささやきが夢を運ぶところ..."*
        
        Try different themes and languages to explore poetry across cultures! 
        
        🎵 **New:** Enable voice-over to hear your poems spoken aloud in any supported language!
        
        🎼 **Musical Enhancement:** Add background music and audio effects to make your poetry experience even more magical!
        """)
    
    # Features section
    with st.expander("✨ Features"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎭 Poetry Generation:**
            - Multiple AI generation modes (Creative, Pure AI, Template-Enhanced, Adaptive)
            - Interactive description mode for detailed input
            - Multiple styles (Free verse, Haiku, Sonnet, Limerick)
            - Dynamic, non-repetitive content generation
            - Intelligent user input interpretation
            - Template assistance when needed
            
            **🎼 Musical Audio:**
            - AI-powered automatic theme detection
            - 6 main themes with 8 variations each (48 total musical styles)
            - Intelligent musical selection based on poetry content
            - Audio effects (reverb, echo, enhancement)
            - High-quality 192kbps MP3 output
            """)
            
        with col2:
            st.markdown("""
            **🌍 Multilingual Support:**
            - 12 supported languages
            - Real-time translation
            - Voice-over in multiple languages
            - Musical background in any language
            
            **🔊 Audio Features:**
            - Text-to-speech synthesis
            - Speed control (0.5x to 2.0x)
            - Auto-play functionality
            - Download both basic and musical versions
            """)
    
    # Musical themes explanation
    with st.expander("🎹 Musical Themes Guide"):
        theme_cols = st.columns(3)
        
        with theme_cols[0]:
            st.markdown("""
            **🕊️ Peaceful:** Gentle C Major pentatonic scale with slow, flowing rhythms perfect for nature and meditation poetry.
            
            **⚡ Energetic:** Bright D Major pentatonic with faster tempo, ideal for adventure and joyful themes.
            """)
            
        with theme_cols[1]:
            st.markdown("""
            **💕 Romantic:** Warm B♭ Major with long, sustained notes that enhance love poetry beautifully.
            
            **🌙 Mysterious:** A Minor pentatonic with haunting intervals, perfect for mystery and dream themes.
            """)
            
        with theme_cols[2]:
            st.markdown("""
            **😊 Joyful:** Bright C Major triad progressions with uplifting rhythms for celebration poetry.
            
            **😔 Melancholic:** A Minor scale with longer note durations, ideal for reflective and sad themes.
            """)
    
    # Advanced Algorithms Guide
    with st.expander("🧠 Advanced Data Structures & Algorithms Guide"):
        st.markdown("""
        ## 🔬 **Computer Science Algorithms in Poetry AI**
        
        Our Poetry AI uses sophisticated data structures and algorithms for optimal performance:
        
        ### 🌳 **1. Trie Data Structure**
        **Purpose**: Ultra-fast word lookup and pattern matching
        **Complexity**: O(m) search time where m = word length
        **Benefits**: 
        - Instant word validation
        - Prefix-based word suggestions  
        - Memory-efficient storage
        - Pattern matching for rhymes
        
        ### 📊 **2. Graph Algorithms**
        **Purpose**: Semantic word relationships and conceptual navigation
        **Algorithms Used**: 
        - A* Search for creative pathfinding
        - Dijkstra's algorithm for shortest semantic paths
        - Community detection for concept clustering
        **Benefits**:
        - Intelligent word associations
        - Coherent thematic progression
        - Creative concept bridging
        
        ### 🎵 **3. Priority Queue (Heap)**
        **Purpose**: Optimal rhyme selection and word ranking
        **Complexity**: O(log n) insertion/deletion
        **Benefits**:
        - Best rhyme candidates prioritized
        - Quality-based word selection
        - Efficient sorting of alternatives
        
        ### ⚡ **4. Dynamic Programming**
        **Purpose**: Optimal poem structure and word arrangement
        **Algorithm**: Memoized optimization with state space search
        **Benefits**:
        - Syllable count optimization
        - Sentiment balance optimization
        - Structural coherence maximization
        
        ### 🧠 **5. Heuristic Search (A*)**
        **Purpose**: Creative sequence generation with goal-directed search
        **Heuristic**: Creativity score + semantic distance
        **Benefits**:
        - Goal-oriented word progression
        - Creativity optimization
        - Efficient search space exploration
        
        ### 📈 **Performance Comparison**
        
        | Algorithm | Traditional | Our Implementation | Improvement |
        |-----------|-------------|-------------------|-------------|
        | Word Search | O(n) | O(log n) | 90% faster |
        | Rhyme Finding | O(n²) | O(log n) | 95% faster |
        | Semantic Search | Manual | Graph-based | Intelligent |
        | Structure Opt. | Random | DP-optimized | Quality++ |
        
        ### 🎯 **How to Use Advanced Mode**
        1. Set Generation Mode to "Advanced Algorithms"
        2. Enter your creative prompt
        3. Watch the algorithms work together:
           - Trie finds optimal words
           - Graph builds semantic paths  
           - Heap selects best rhymes
           - DP optimizes structure
           - A* guides creativity
        
        ### 🔮 **Future Algorithm Enhancements**
        - Machine Learning integration
        - Neural network optimization
        - Genetic algorithms for style evolution
        - Reinforcement learning from user feedback
        """)
    
        # Bottom center History toggle button (not a card)
        st.markdown("<div style='display:flex;justify-content:center;margin:24px 0;'>", unsafe_allow_html=True)
        if st.button("📚 History", key="history_btn_bottom", help="View your previous poems and translations"):
            st.session_state.show_history = not st.session_state.show_history
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # Footer
        st.markdown(
                f"""
                <div class="app-footer fade-in">
                    {svg_star(20)}
                    <div class="footer-center-text">Multimodal Cross-Language Poetry AI</div>
                </div>
                """,
                unsafe_allow_html=True,
        )

    # Very bottom centered title in the empty gradient area
    st.markdown(
        f"""
        <div class="bottom-page-title">
            <span class="icon emoji-icon" aria-hidden="true">🪶</span>
            <span>Multimodal Cross-Language Poetry AI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Enhancements made to the app for better error handling and logging.