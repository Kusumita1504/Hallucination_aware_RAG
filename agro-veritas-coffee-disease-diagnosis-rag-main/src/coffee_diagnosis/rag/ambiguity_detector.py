"""
Ambiguity Detection Module
Detects missing symptom information in user queries
"""

from typing import List, Set
import re


class AmbiguityDetector:
    def __init__(self):
        """Initialize ambiguity detector with symptom keywords"""
        self.color_keywords = {
            'yellow', 'brown', 'red', 'orange', 'black', 'grey', 'white',
            'rusty', 'dark', 'pale', 'bright', 'golden', 'fading', 'bleached'
        }

        self.pattern_keywords = {
            'spots', 'patches', 'streaks', 'rings', 'dots', 'blotches',
            'pustules', 'lesions', 'necrotic', 'concentric', 'angular',
            'circular', 'irregular', 'powder', 'dust'
        }

        self.location_keywords = {
            'leaves', 'leaf', 'stem', 'branches', 'roots', 'fruits', 'cherries',
            'berries', 'flowers', 'veins', 'margins', 'center', 'surface',
            'undersurface', 'petioles', 'pods'
        }

        self.spread_keywords = {
            'isolated', 'widespread', 'localized', 'scattered', 'clustered',
            'spreading', 'rapid', 'slow', 'severe', 'mild', 'moderate',
            'throughout', 'patchy', 'everywhere', 'few', 'many'
        }

        self.timing_keywords = {
            'new', 'old', 'recent', 'morning', 'evening', 'wet', 'dry',
            'during', 'after', 'before', 'suddenly', 'gradually', 'rainy',
            'week', 'day', 'month', 'year', 'ago', 'yesterday', 'today',
            'last', 'this', 'past', 'early', 'late', 'season', 'started',
            'began', 'appeared', 'noticed', 'seen'
        }

    def detect_ambiguity(self, query: str) -> dict:
        """
        Detect missing symptom information

        Args:
            query: User query about coffee disease symptoms

        Returns:
            Dictionary with detected attributes and missing info
        """
        query_lower = query.lower()

        detected = {
            'color': self._detect_colors(query_lower),
            'pattern': self._detect_patterns(query_lower),
            'location': self._detect_locations(query_lower),
            'spread': self._detect_spread(query_lower),
            'timing': self._detect_timing(query_lower)
        }

        missing = self._identify_missing(detected)

        return {
            'detected': detected,
            'missing': missing,
            'full_query': query
        }

    def _detect_colors(self, query: str) -> List[str]:
        """Detect color information in query"""
        found = []
        query_lower = query.lower()
        for color in self.color_keywords:
            # Match color word, variations like yellowish, yellowing, etc.
            if color in query_lower:
                found.append(color)
        return found

    def _detect_patterns(self, query: str) -> List[str]:
        """Detect pattern information in query"""
        found = []
        query_lower = query.lower()
        for pattern in self.pattern_keywords:
            # Match pattern word and variations
            if pattern in query_lower:
                found.append(pattern)
        return found

    def _detect_locations(self, query: str) -> List[str]:
        """Detect location information in query"""
        found = []
        query_lower = query.lower()
        for location in self.location_keywords:
            # Match location word and variations
            if location in query_lower:
                found.append(location)
        return found

    def _detect_spread(self, query: str) -> List[str]:
        """Detect spread information in query"""
        found = []
        query_lower = query.lower()

        # Direct matches
        for spread in self.spread_keywords:
            if spread in query_lower:
                found.append(spread)

        # Typo tolerance: check for similar words
        spread_variations = {
            'wide': ['widspread', 'wide spread', 'widely'],
            'isolated': ['isolate'],
            'localized': ['localize', 'local'],
            'scattered': ['scatter'],
            'clustered': ['cluster'],
        }

        for key, variations in spread_variations.items():
            for var in variations:
                if var in query_lower:
                    found.append(key)
                    break

        return list(set(found))  # Remove duplicates

    def _detect_timing(self, query: str) -> List[str]:
        """Detect timing information in query"""
        found = []
        query_lower = query.lower()
        for timing in self.timing_keywords:
            # Match timing word and variations
            if timing in query_lower:
                found.append(timing)
        return found

    def _identify_missing(self, detected: dict) -> dict:
        """Identify what information is missing"""
        missing = {}

        if not detected['color']:
            missing['color'] = "color"

        if not detected['pattern']:
            missing['pattern'] = "pattern"

        if not detected['location']:
            missing['location'] = "location"

        if not detected['spread']:
            missing['spread'] = "spread"

        if not detected['timing']:
            missing['timing'] = "timing"

        return missing

    def get_missing_priority(self, missing: dict) -> List[str]:
        """
        Get missing attributes in priority order

        Args:
            missing: Missing info dictionary

        Returns:
            List of missing attribute keys in priority order
        """
        # Priority: location > color > pattern > spread
        priority = ['location', 'color', 'pattern', 'spread']
        result = []

        for key in priority:
            if key in missing:
                result.append(key)

        return result
