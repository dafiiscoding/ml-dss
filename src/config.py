import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, "sample")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Create directories if they don't exist
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# Model configuration
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
RANDOM_STATE = 42

# Late Fusion weights
WEIGHT_TEXT_PROB = 0.6
WEIGHT_IMAGE_PROB = 0.4
INFORMATIVE_THRESHOLD = 0.5
MANUAL_REVIEW_CONFLICT_THRESHOLD = 0.5
FUSION_CONFIG_PATH = os.path.join(MODELS_DIR, "fusion_config.json")

# Humanitarian Category Weights for Risk Score
# Aligned to the 8 official CrisisMMD v2.0 humanitarian categories.
CATEGORY_WEIGHTS = {
    "injured_or_dead_people": 1.00,                 # direct threat to life
    "missing_or_found_people": 0.95,                # search & rescue, life at risk
    "rescue_volunteering_or_donation_effort": 0.85, # active relief / donation
    "infrastructure_and_utility_damage": 0.80,      # damage to roads/utilities
    "affected_individuals": 0.70,                   # displaced / affected people
    "vehicle_damage": 0.50,                         # lower-severity damage
    "other_relevant_information": 0.40,             # relevant but not direct threat
    "not_humanitarian": 0.00                        # non-humanitarian / noise
}

# Default weight if category is not found
DEFAULT_CATEGORY_WEIGHT = 0.40

# Risk Score formulation weights
# risk_score = WEIGHT_FUSION * fusion_prob + WEIGHT_CAT * cat_weight + WEIGHT_KEYWORD * keyword_score + WEIGHT_CONF * confidence_score
WEIGHT_RISK_FUSION = 40
WEIGHT_RISK_CATEGORY = 25
WEIGHT_RISK_KEYWORD = 20
WEIGHT_RISK_CONFIDENCE = 15

# Priority levels
PRIORITY_LOW_MAX = 39
PRIORITY_MEDIUM_MAX = 69

# Routing & Action config (8 CrisisMMD humanitarian categories)
ROUTING_RULES = {
    "injured_or_dead_people": {
        "team": "Emergency Team",
        "action": "Verify location immediately and dispatch rescue/medical crews."
    },
    "missing_or_found_people": {
        "team": "Emergency Team",
        "action": "Forward to search & rescue; cross-check missing/found persons registry."
    },
    "rescue_volunteering_or_donation_effort": {
        "team": "Relief Team",
        "action": "Coordinate with logistics to deliver emergency supplies or direct volunteer groups."
    },
    "infrastructure_and_utility_damage": {
        "team": "Infrastructure Team",
        "action": "Inspect blocked roads or utility failures to plan repairs."
    },
    "affected_individuals": {
        "team": "Relief Team",
        "action": "Route info to local shelter/medical hubs to support affected citizens."
    },
    "vehicle_damage": {
        "team": "Infrastructure Team",
        "action": "Log damaged/blocking vehicles for road-clearing and towing."
    },
    "other_relevant_information": {
        "team": "Coordination Team",
        "action": "Monitor situation and log for reference in situational reports."
    },
    "not_humanitarian": {
        "team": "No Action",
        "action": "Archive post and monitor for future developments."
    }
}

# Crisis Keywords for Keyword Score
CRISIS_KEYWORDS = [
    "help", "rescue", "trapped", "injured", "dead", "casualty", "blood", "hospital",
    "flood", "earthquake", "hurricane", "fire", "wildfire", "storm", "cyclone", "typhoon",
    "blocked", "collapsed", "damaged", "destroyed", "bridge", "road", "landslide",
    "emergency", "urgent", "danger", "missing", "found", "survivor", "crying", "scream",
    "water", "food", "shelter", "blanket", "donation", "volunteer", "doctor", "nurse"
]
