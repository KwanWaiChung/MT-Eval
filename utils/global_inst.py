# Adapted from https://github.com/google-research/google-research/tree/master/instruction_following_eval

import random
import emoji
import string
import json
import nltk
import functools
import re
import langdetect
import logging
from langdetect import detect
from typing import List, Any, Dict

logger = logging.getLogger(__name__)

_SEEDER = random.Random(11)
# fmt: off
_WORD_LIST = ["western", "sentence", "signal", "dump", "spot", "opposite", "bottom", "potato", "administration", "working", "welcome", "morning", "good", "agency", "primary", "wish", "responsibility", "press", "problem", "president", "steal", "brush", "read", "type", "beat", "trainer", "growth", "lock", "bone", "case", "equal", "comfortable", "region", "replacement", "performance", "mate", "walk", "medicine", "film", "thing", "rock", "tap", "total", "competition", "ease", "south", "establishment", "gather", "parking", "world", "plenty", "breath", "claim", "alcohol", "trade", "dear", "highlight", "street", "matter", "decision", "mess", "agreement", "studio", "coach", "assist", "brain", "wing", "style", "private", "top", "brown", "leg", "buy", "procedure", "method", "speed", "high", "company", "valuable", "pie", "analyst", "session", "pattern", "district", "pleasure", "dinner", "swimming", "joke", "order", "plate", "department", "motor", "cell", "spend", "cabinet", "difference", "power", "examination", "engine", "horse", "dimension", "pay", "toe", "curve", "literature", "bother", "fire", "possibility", "debate", "activity", "passage", "hello", "cycle", "background", "quiet", "author", "effect", "actor", "page", "bicycle", "error", "throat", "attack", "character", "phone", "tea", "increase", "outcome", "file", "specific", "inspector", "internal", "potential", "staff", "building", "employer", "shoe", "hand", "direction", "garden", "purchase", "interview", "study", "recognition", "member", "spiritual", "oven", "sandwich", "weird", "passenger", "particular", "response", "reaction", "size", "variation", "a", "cancel", "candy", "exit", "guest", "condition", "fly", "price", "weakness", "convert", "hotel", "great", "mouth", "mind", "song", "sugar", "suspect", "telephone", "ear", "roof", "paint", "refrigerator", "organization", "jury", "reward", "engineering", "day", "possession", "crew", "bar", "road", "description", "celebration", "score", "mark", "letter", "shower", "suggestion", "sir", "luck", "national", "progress", "hall", "stroke", "theory", "offer", "story", "tax", "definition", "history", "ride", "medium", "opening", "glass", "elevator", "stomach", "question", "ability", "leading", "village", "computer", "city", "grand", "confidence", "candle", "priest", "recommendation", "point", "necessary", "body", "desk", "secret", "horror", "noise", "culture", "warning", "water", "round", "diet", "flower", "bus", "tough", "permission", "week", "prompt", "connection", "abuse", "height", "save", "corner", "border", "stress", "drive", "stop", "rip", "meal", "listen", "confusion", "girlfriend", "living", "relation", "significance", "plan", "creative", "atmosphere", "blame", "invite", "housing", "paper", "drink", "roll", "silver", "drunk", "age", "damage", "smoke", "environment", "pack", "savings", "influence", "tourist", "rain", "post", "sign", "grandmother", "run", "profit", "push", "clerk", "final", "wine", "swim", "pause", "stuff", "singer", "funeral", "average", "source", "scene", "tradition", "personal", "snow", "nobody", "distance", "sort", "sensitive", "animal", "major", "negotiation", "click", "mood", "period", "arrival", "expression", "holiday", "repeat", "dust", "closet", "gold", "bad", "sail", "combination", "clothes", "emphasis", "duty", "black", "step", "school", "jump", "document", "professional", "lip", "chemical", "front", "wake", "while", "inside", "watch", "row", "subject", "penalty", "balance", "possible", "adult", "aside", "sample", "appeal", "wedding", "depth", "king", "award", "wife", "blow", "site", "camp", "music", "safe", "gift", "fault", "guess", "act", "shame", "drama", "capital", "exam", "stupid", "record", "sound", "swing", "novel", "minimum", "ratio", "machine", "shape", "lead", "operation", "salary", "cloud", "affair", "hit", "chapter", "stage", "quantity", "access", "army", "chain", "traffic", "kick", "analysis", "airport", "time", "vacation", "philosophy", "ball", "chest", "thanks", "place", "mountain", "advertising", "red", "past", "rent", "return", "tour", "house", "construction", "net", "native", "war", "figure", "fee", "spray", "user", "dirt", "shot", "task", "stick", "friend", "software", "promotion", "interaction", "surround", "block", "purpose", "practice", "conflict", "routine", "requirement", "bonus", "hole", "state", "junior", "sweet", "catch", "tear", "fold", "wall", "editor", "life", "position", "pound", "respect", "bathroom", "coat", "script", "job", "teach", "birth", "view", "resolve", "theme", "employee", "doubt", "market", "education", "serve", "recover", "tone", "harm", "miss", "union", "understanding", "cow", "river", "association", "concept", "training", "recipe", "relationship", "reserve", "depression", "proof", "hair", "revenue", "independent", "lift", "assignment", "temporary", "amount", "loss", "edge", "track", "check", "rope", "estimate", "pollution", "stable", "message", "delivery", "perspective", "mirror", "assistant", "representative", "witness", "nature", "judge", "fruit", "tip", "devil", "town", "emergency", "upper", "drop", "stay", "human", "neck", "speaker", "network", "sing", "resist", "league", "trip", "signature", "lawyer", "importance", "gas", "choice", "engineer", "success", "part", "external", "worker", "simple", "quarter", "student", "heart", "pass", "spite", "shift", "rough", "lady", "grass", "community", "garage", "youth", "standard", "skirt", "promise", "blind", "television", "disease", "commission", "positive", "energy", "calm", "presence", "tune", "basis", "preference", "head", "common", "cut", "somewhere", "presentation", "current", "thought", "revolution", "effort", "master", "implement", "republic", "floor", "principle", "stranger", "shoulder", "grade", "button", "tennis", "police", "collection", "account", "register", "glove", "divide", "professor", "chair", "priority", "combine", "peace", "extension", "maybe", "evening", "frame", "sister", "wave", "code", "application", "mouse", "match", "counter", "bottle", "half", "cheek", "resolution", "back", "knowledge", "make", "discussion", "screw", "length", "accident", "battle", "dress", "knee", "log", "package", "it", "turn", "hearing", "newspaper", "layer", "wealth", "profile", "imagination", "answer", "weekend", "teacher", "appearance", "meet", "bike", "rise", "belt", "crash", "bowl", "equivalent", "support", "image", "poem", "risk", "excitement", "remote", "secretary", "public", "produce", "plane", "display", "money", "sand", "situation", "punch", "customer", "title", "shake", "mortgage", "option", "number", "pop", "window", "extent", "nothing", "experience", "opinion", "departure", "dance", "indication", "boy", "material", "band", "leader", "sun", "beautiful", "muscle", "farmer", "variety", "fat", "handle", "director", "opportunity", "calendar", "outside", "pace", "bath", "fish", "consequence", "put", "owner", "go", "doctor", "information", "share", "hurt", "protection", "career", "finance", "force", "golf", "garbage", "aspect", "kid", "food", "boot", "milk", "respond", "objective", "reality", "raw", "ring", "mall", "one", "impact", "area", "news", "international", "series", "impress", "mother", "shelter", "strike", "loan", "month", "seat", "anything", "entertainment", "familiar", "clue", "year", "glad", "supermarket", "natural", "god", "cost", "conversation", "tie", "ruin", "comfort", "earth", "storm", "percentage", "assistance", "budget", "strength", "beginning", "sleep", "other", "young", "unit", "fill", "store", "desire", "hide", "value", "cup", "maintenance", "nurse", "function", "tower", "role", "class", "camera", "database", "panic", "nation", "basket", "ice", "art", "spirit", "chart", "exchange", "feedback", "statement", "reputation", "search", "hunt", "exercise", "nasty", "notice", "male", "yard", "annual", "collar", "date", "platform", "plant", "fortune", "passion", "friendship", "spread", "cancer", "ticket", "attitude", "island", "active", "object", "service", "buyer", "bite", "card", "face", "steak", "proposal", "patient", "heat", "rule", "resident", "broad", "politics", "west", "knife", "expert", "girl", "design", "salt", "baseball", "grab", "inspection", "cousin", "couple", "magazine", "cook", "dependent", "security", "chicken", "version", "currency", "ladder", "scheme", "kitchen", "employment", "local", "attention", "manager", "fact", "cover", "sad", "guard", "relative", "county", "rate", "lunch", "program", "initiative", "gear", "bridge", "breast", "talk", "dish", "guarantee", "beer", "vehicle", "reception", "woman", "substance", "copy", "lecture", "advantage", "park", "cold", "death", "mix", "hold", "scale", "tomorrow", "blood", "request", "green", "cookie", "church", "strip", "forever", "beyond", "debt", "tackle", "wash", "following", "feel", "maximum", "sector", "sea", "property", "economics", "menu", "bench", "try", "language", "start", "call", "solid", "address", "income", "foot", "senior", "honey", "few", "mixture", "cash", "grocery", "link", "map", "form", "factor", "pot", "model", "writer", "farm", "winter", "skill", "anywhere", "birthday", "policy", "release", "husband", "lab", "hurry", "mail", "equipment", "sink", "pair", "driver", "consideration", "leather", "skin", "blue", "boat", "sale", "brick", "two", "feed", "square", "dot", "rush", "dream", "location", "afternoon", "manufacturer", "control", "occasion", "trouble", "introduction", "advice", "bet", "eat", "kill", "category", "manner", "office", "estate", "pride", "awareness", "slip", "crack", "client", "nail", "shoot", "membership", "soft", "anybody", "web", "official", "individual", "pizza", "interest", "bag", "spell", "profession", "queen", "deal", "resource", "ship", "guy", "chocolate", "joint", "formal", "upstairs", "car", "resort", "abroad", "dealer", "associate", "finger", "surgery", "comment", "team", "detail", "crazy", "path", "tale", "initial", "arm", "radio", "demand", "single", "draw", "yellow", "contest", "piece", "quote", "pull", "commercial", "shirt", "contribution", "cream", "channel", "suit", "discipline", "instruction", "concert", "speech", "low", "effective", "hang", "scratch", "industry", "breakfast", "lay", "join", "metal", "bedroom", "minute", "product", "rest", "temperature", "many", "give", "argument", "print", "purple", "laugh", "health", "credit", "investment", "sell", "setting", "lesson", "egg", "middle", "marriage", "level", "evidence", "phrase", "love", "self", "benefit", "guidance", "affect", "you", "dad", "anxiety", "special", "boyfriend", "test", "blank", "payment", "soup", "obligation", "reply", "smile", "deep", "complaint", "addition", "review", "box", "towel", "minor", "fun", "soil", "issue", "cigarette", "internet", "gain", "tell", "entry", "spare", "incident", "family", "refuse", "branch", "can", "pen", "grandfather", "constant", "tank", "uncle", "climate", "ground", "volume", "communication", "kind", "poet", "child", "screen", "mine", "quit", "gene", "lack", "charity", "memory", "tooth", "fear", "mention", "marketing", "reveal", "reason", "court", "season", "freedom", "land", "sport", "audience", "classroom", "law", "hook", "win", "carry", "eye", "smell", "distribution", "research", "country", "dare", "hope", "whereas", "stretch", "library", "if", "delay", "college", "plastic", "book", "present", "use", "worry", "champion", "goal", "economy", "march", "election", "reflection", "midnight", "slide", "inflation", "action", "challenge", "guitar", "coast", "apple", "campaign", "field", "jacket", "sense", "way", "visual", "remove", "weather", "trash", "cable", "regret", "buddy", "beach", "historian", "courage", "sympathy", "truck", "tension", "permit", "nose", "bed", "son", "person", "base", "meat", "usual", "air", "meeting", "worth", "game", "independence", "physical", "brief", "play", "raise", "board", "she", "key", "writing", "pick", "command", "party", "yesterday", "spring", "candidate", "physics", "university", "concern", "development", "change", "string", "target", "instance", "room", "bitter", "bird", "football", "normal", "split", "impression", "wood", "long", "meaning", "stock", "cap", "leadership", "media", "ambition", "fishing", "essay", "salad", "repair", "today", "designer", "night", "bank", "drawing", "inevitable", "phase", "vast", "chip", "anger", "switch", "cry", "twist", "personality", "attempt", "storage", "being", "preparation", "bat", "selection", "white", "technology", "contract", "side", "section", "station", "till", "structure", "tongue", "taste", "truth", "difficulty", "group", "limit", "main", "move", "feeling", "light", "example", "mission", "might", "wait", "wheel", "shop", "host", "classic", "alternative", "cause", "agent", "consist", "table", "airline", "text", "pool", "craft", "range", "fuel", "tool", "partner", "load", "entrance", "deposit", "hate", "article", "video", "summer", "feature", "extreme", "mobile", "hospital", "flight", "fall", "pension", "piano", "fail", "result", "rub", "gap", "system", "report", "suck", "ordinary", "wind", "nerve", "ask", "shine", "note", "line", "mom", "perception", "brother", "reference", "bend", "charge", "treat", "trick", "term", "homework", "bake", "bid", "status", "project", "strategy", "orange", "let", "enthusiasm", "parent", "concentrate", "device", "travel", "poetry", "business", "society", "kiss", "end", "vegetable", "employ", "schedule", "hour", "brave", "focus", "process", "movie", "illegal", "general", "coffee", "ad", "highway", "chemistry", "psychology", "hire", "bell", "conference", "relief", "show", "neat", "funny", "weight", "quality", "club", "daughter", "zone", "touch", "tonight", "shock", "burn", "excuse", "name", "survey", "landscape", "advance", "satisfaction", "bread", "disaster", "item", "hat", "prior", "shopping", "visit", "east", "photo", "home", "idea", "father", "comparison", "cat", "pipe", "winner", "count", "lake", "fight", "prize", "foundation", "dog", "keep", "ideal", "fan", "struggle", "peak", "safety", "solution", "hell", "conclusion", "population", "strain", "alarm", "measurement", "second", "train", "race", "due", "insurance", "boss", "tree", "monitor", "sick", "course", "drag", "appointment", "slice", "still", "care", "patience", "rich", "escape", "emotion", "royal", "female", "childhood", "government", "picture", "will", "sock", "big", "gate", "oil", "cross", "pin", "improvement", "championship", "silly", "help", "sky", "pitch", "man", "diamond", "most", "transition", "work", "science", "committee", "moment", "fix", "teaching", "dig", "specialist", "complex", "guide", "people", "dead", "voice", "original", "break", "topic", "data", "degree", "reading", "recording", "bunch", "reach", "judgment", "lie", "regular", "set", "painting", "mode", "list", "player", "bear", "north", "wonder", "carpet", "heavy", "officer", "negative", "clock", "unique", "baby", "pain", "assumption", "disk", "iron", "bill", "drawer", "look", "double", "mistake", "finish", "future", "brilliant", "contact", "math", "rice", "leave", "restaurant", "discount", "sex", "virus", "bit", "trust", "event", "wear", "juice", "failure", "bug", "context", "mud", "whole", "wrap", "intention", "draft", "pressure", "cake", "dark", "explanation", "space", "angle", "word", "efficiency", "management", "habit", "star", "chance", "finding", "transportation", "stand", "criticism", "flow", "door", "injury", "insect", "surprise", "apartment"]  # pylint: disable=line-too-long
# fmt: on
_NUM_KEYWORDS = 2
_NUM_PLACEHOLDERS = 2
_NUM_BULLETS = 4

# Used in EndPhrase
_ENDING_OPTIONS = (
    "Any other questions?",
    "Is there anything else I can help with?",
)

# Used in ConstrainedResponse
_RESPONSE_OPTIONS = (
    "My answer is yes.",
    "My answer is no.",
    "My answer is maybe.",
)

_NUM_WORDS_LOWER_LIMIT = 100
_NUM_WORDS_UPPER_LIMIT = 200
_MAX_NUM_SENTENCES = 6
# _COMPARISON_RElATION = ["less than", "more than"]
_COMPARISON_RElATION = ["less than"]


def count_words(text):
    """Counts the number of words."""
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)
    num_words = len(tokens)
    return num_words


def count_sentences(text):
    """Count the number of sentences."""
    sent_count = len(nltk.tokenize.sent_tokenize(text))
    return sent_count


def start_emoji_checker(response: str):
    return emoji.is_emoji(response.strip()[0])


class Instruction:
    def __init__(*args, **kwargs):
        pass

    def get_args(self) -> Dict[str, Any]:
        raise NotImplementedError("`get_args` not implemented.")

    def get_prompt(self) -> str:
        raise NotImplementedError("`get_prmopt` not implemented.")

    def check_following(self, response: str) -> bool:
        raise NotImplementedError("`check_following` not implemented.")


class StartChar(Instruction):
    description = "Start with a particular letter."

    def __init__(self, letter: str = ""):
        if not letter:
            letter = _SEEDER.choice(string.ascii_lowercase)
        self.letter = letter

    def get_args(self) -> Dict[str, Any]:
        return {"letter": self.letter}

    def get_prompt(self):
        return (
            "Begin all your responses in the upcoming conversation with the"
            f" letter {self.letter}."
        )

    def check_following(self, response):
        return response.strip().lower().startswith(self.letter.lower())


class StartEmoji(Instruction):
    description = "Start with an emoji."

    def get_args(self) -> Dict[str, Any]:
        return {}

    def get_prompt(self):
        return (
            "Begin all your responses in the upcoming conversation with emoji."
        )

    def check_following(self, response):
        return emoji.is_emoji(response.strip()[0])


class EndPhrase(Instruction):
    description = "End with a particular phrase."

    def __init__(self, end_phrase: str = None) -> None:
        if end_phrase is None:
            end_phrase = _SEEDER.choice(_ENDING_OPTIONS)
        self.end_phrase = end_phrase

    def get_args(self) -> Dict[str, Any]:
        return {"end_phrase": self.end_phrase}

    def get_prompt(self):
        return (
            "All your responses in the upcoming conversation must end with"
            f" this exact phrase '{self.end_phrase}'. No other words should"
            " follow this phrase."
        )

    def check_following(self, response):
        response = response.strip().strip("'").lower()
        return response.endswith(self.end_phrase.lower())


class ResponseLanguage(Instruction):
    description = "Respond in a particular language."

    langs = {
        "English": "en",
        "Spanish": "es",
        "Portuguese": "pt",
        "Arabic": "ar",
        "Hindi": "hi",
        "French": "fr",
        "Russian": "ru",
        "German": "de",
        "Japanese": "ja",
        "Italian": "it",
        "Bengali": "bn",
        "Ukrainian": "uk",
        "Thai": "th",
        "Urdu": "ur",
        "Tamil": "ta",
        "Telugu": "te",
        "Bulgarian": "bg",
        "Korean": "ko",
        "Polish": "pl",
        "Hebrew": "he",
        "Persian": "fa",
        "Vietnamese": "vi",
        "Nepali": "ne",
        "Swahili": "sw",
        "Kannada": "kn",
        "Marathi": "mr",
        "Gujarati": "gu",
        "Punjabi": "pa",
        "Malayalam": "ml",
        "Finnish": "fi",
    }

    def __init__(self, lang: str = ""):
        if not lang:
            lang = _SEEDER.choice(list(self.langs.keys()))
        self.lang = lang

    def get_args(self) -> Dict[str, Any]:
        return {"lang": self.lang}

    def get_prompt(self):
        return (
            "All your responses in the upcoming conversation must be written"
            f" in {self.lang} language, no other language is allowed."
        )

    def check_following(self, response):
        response = response.strip()
        if not response:
            return False
        try:
            langdetect.detect(response)
        except langdetect.LangDetectException as e:
            logging.exception(e)
            return True
        return detect(response) == self.langs[self.lang]


class JsonFormat(Instruction):
    description = "Respond using JSON."
    """Check the Json format."""

    def get_args(self) -> Dict[str, Any]:
        return {}

    def get_prompt(self):
        return (
            "All your responses in the upcoming conversation must be wrapped"
            " in JSON format. You can use markdown ticks such as ```."
        )

    def check_following(self, response):
        response = (
            response.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(response)
        except ValueError as _:
            return False
        return True


class NumberWords(Instruction):
    description = "Respond with a specified word limit."

    def __init__(
        self,
        num_words: int = None,
        relation: str = "",
    ):
        if not num_words:
            num_words = _SEEDER.choice(
                list(
                    range(
                        _NUM_WORDS_LOWER_LIMIT, _NUM_WORDS_UPPER_LIMIT + 1, 10
                    )
                )
            )
        self.num_words = num_words
        if not relation:
            relation = _SEEDER.choice(_COMPARISON_RElATION)
        elif relation not in _COMPARISON_RElATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RElATION}, but `{relation}` is given."
            )
        self.relation = relation

    def get_args(self) -> Dict[str, Any]:
        return {"num_words": self.num_words, "relation": self.relation}

    def get_prompt(self):
        return (
            "All your responses in the upcoming conversation must be"
            f" {self.relation} {self.num_words} words."
        )

    def check_following(self, response):
        num_words = count_words(response)
        # less than
        if self.relation == _COMPARISON_RElATION[0]:
            return num_words < self.num_words
        elif self.relation == _COMPARISON_RElATION[1]:
            return num_words > self.num_words


class NumberSentences(Instruction):
    description = "Respond with a specified sentence limit."

    def __init__(
        self,
        num_sentences: int = None,
        relation: str = "",
    ):
        if not num_sentences:
            num_sentences = _SEEDER.randint(3, _MAX_NUM_SENTENCES)
        self.num_sentences = num_sentences
        if not relation:
            relation = _SEEDER.choice(_COMPARISON_RElATION)
        elif relation not in _COMPARISON_RElATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RElATION}, but {relation} is given."
            )
        self.relation = relation

    def get_args(self) -> Dict[str, Any]:
        return {"num_sentences": self.num_sentences, "relation": self.relation}

    def get_prompt(self):
        return (
            "All your responses in the upcoming conversation must be"
            f" {self.relation} {self.num_sentences} sentences."
        )

    def check_following(self, response):
        num_sentences = count_sentences(response)
        # less than
        if self.relation == _COMPARISON_RElATION[0]:
            return num_sentences < self.num_sentences
        elif self.relation == _COMPARISON_RElATION[1]:
            return num_sentences > self.num_sentences


class BulletList(Instruction):
    description = "Using a specified number of bullet lists."

    def __init__(
        self,
        num_bullets: int = None,
    ):
        """_summary_

        Args:
            num_bullet_list (int): Number of bullet lists that is required to
                appear in the response. 0 indicates no limitation.
        """
        if num_bullets is None:
            num_bullets = _SEEDER.randint(0, _NUM_BULLETS)
        self.num_bullets = num_bullets

    def get_args(self) -> Dict[str, Any]:
        return {"num_bullets": self.num_bullets}

    def get_prompt(self):
        prompt_suffix = (
            "Use the markdown bullet points such as:\n"
            "* This is point 1.\n"
            "* This is point 2.\n"
        )
        if self.num_bullets == 0:
            prompt = (
                "All your responses in the upcoming conversation must only"
                f" use bullet points. {prompt_suffix}"
            )
        else:
            prompt = (
                "All your responses in the upcoming conversation must contain"
                f" exactly {self.num_bullets} bullet points. {prompt_suffix}"
            )
        return prompt

    def check_following(self, response):
        bullet_lists = re.findall(
            r"^\s*\*[^\*].*$", response, flags=re.MULTILINE
        )
        bullet_lists_2 = re.findall(r"^\s*-.*$", response, flags=re.MULTILINE)
        num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
        if self.num_bullets == 0:
            return num_bullet_lists > 0
        return num_bullet_lists == self.num_bullets


class KeywordExistence(Instruction):
    description = "Include some specific keywords."

    def __init__(self, keywords: List[str] = None) -> None:
        if not keywords:
            keywords = _SEEDER.choices(_WORD_LIST, k=_NUM_KEYWORDS)
        self.keywords = sorted(keywords)

    def get_args(self) -> Dict[str, Any]:
        return {"keywords": self.keywords}

    def get_prompt(self) -> str:
        return (
            f"Include keywords {self.keywords} in all your responses in the"
            " upcoming conversation."
        )

    def check_following(self, response):
        for keyword in self.keywords:
            if not keyword.lower() in response.lower():
                return False
        return True


class CapitalLetter(Instruction):
    description = "Respond in uppercase."
    """Response is in all capital letters."""

    def get_args(self) -> Dict[str, Any]:
        return {}

    def get_prompt(self) -> str:
        return (
            "All your responses in the upcoming conversation must be in all"
            " capital letters."
        )

    def check_following(self, response: str) -> bool:
        return response.isupper()


class LowerCase(Instruction):
    description = "Respond in lowercase.."
    """Response is in all lower letters."""

    def get_args(self) -> Dict[str, Any]:
        return {}

    def get_prompt(self) -> str:
        return (
            "All your responses in the upcoming conversation must be in all"
            " lower letters."
        )

    def check_following(self, response: str) -> bool:
        return response.islower()


class Comma(Instruction):
    description = "Refrain from using commas."

    def get_args(self) -> Dict[str, Any]:
        return {}

    def get_prompt(self) -> str:
        return (
            "All your responses in the upcoming conversation must refrain from"
            " the use of any commas."
        )

    def check_following(self, response: str) -> bool:
        return "," not in response


class TwoResponses(Instruction):
    description = "Include two different responses."

    def get_args(self) -> Dict[str, Any]:
        return {}

    def get_prompt(self) -> str:
        return (
            "All your responses in the upcoming conversation must have two"
            " different responses. Your two different responses should be"
            " separated by 6 asterisk symbols: ******."
        )

    def check_following(self, response: str) -> bool:
        valid_responses = list()
        responses = response.split("******")
        for index, response in enumerate(responses):
            if not response.strip():
                # Might be empty spaces in the beginning or the end.
                if index != 0 and index != len(responses) - 1:
                    return False
            else:
                valid_responses.append(response)
        return (
            len(valid_responses) == 2
            and valid_responses[0].strip() != valid_responses[1].strip()
        )


class PlaceHolder(Instruction):
    description = "Include a certain amount of placeholders."

    def __init__(self, num_placeholder: int = None) -> None:
        if num_placeholder is None or num_placeholder < 0:
            num_placeholder = _SEEDER.randint(1, _NUM_PLACEHOLDERS)
        self.num_placeholder = num_placeholder

    def get_args(self) -> Dict[str, Any]:
        return {"num_placeholder": self.num_placeholder}

    def get_prompt(self) -> str:
        return (
            "All your responses in the upcoming conversation must contain at"
            f" least {self.num_placeholder} placeholders represented by square"
            " brackets, such as [address]."
        )

    def check_following(self, response: str) -> bool:
        placeholders = re.findall(r"\[.*?\]", response)
        num_placeholders = len(placeholders)
        return num_placeholders >= self.num_placeholder


class ConstrainedResponse(Instruction):
    description = "Reply with one of the provided response options."

    def __init__(self, response_options: List[str] = None) -> None:
        if response_options is None:
            response_options = _RESPONSE_OPTIONS
        self.response_options = response_options

    def get_args(self) -> Dict[str, Any]:
        return {"response_options": self.response_options}

    def get_prompt(self) -> str:
        return (
            "All your responses in the upcoming conversation must be one of"
            f" the following options: {self.response_options}."
        )

    def check_following(self, response: str) -> bool:
        response = response
        for const in self.response_options:
            if const in response:
                return True
        return False


INSTRUCTIONS = {
    "startend:start_char": StartChar,
    "startend:start_emoji": StartEmoji,
    "startend:end_phrase": EndPhrase,
    "language:response_language": ResponseLanguage,
    "format:json_format": JsonFormat,
    "format:bullet_list": BulletList,
    # "length_constraints:number_words": NumberWords,
    "length_constraints:number_sentences": NumberSentences,
    "keywords:existence": KeywordExistence,
    "change_case:capital_letter": CapitalLetter,
    "change_case:lowercase": LowerCase,
    "punctuation:no_comma": Comma,
    "combination:two_responses": TwoResponses,
    "content:placeholder": PlaceHolder,
    "format:constrained_response": ConstrainedResponse,
}
