# AceRL Configuration Settings

# Game Constants
SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
HIGH_RANKS = ['Q', 'K', 'A']
NUM_CARDS = len(SUITS) * len(RANKS)

# Deep Learning Hyperparameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
HIDDEN_SIZE = 2048
MAX_HISTORY = 60

# Reinforcement Learning Parameters
MEMORY_SIZE = 200000  # Increased for better experience replay
GAMMA = 0.98  # Discount factor
EPSILON = 1.0  # Initial exploration rate
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.01

# Training Parameters
NUM_EPISODES = 3000  # Increased for better convergence
TARGET_UPDATE_FREQUENCY = 100
EVALUATION_FREQUENCY = 200
SAVE_FREQUENCY = 500

# Enhanced Reward Structure
REWARD_WIN_TRICK = 2.5
REWARD_LOSE_TRICK = -0.75
PENALTY_HIGH_CARD_EARLY = -1.5
BONUS_STRATEGIC_LATE = 0.75
BONUS_HIGH_CARDS_IN_HAND = 0.5

# Model Paths
MODEL_DIR = 'models'
BEST_MODEL_PATH = f'{MODEL_DIR}/best_ace_rl_agent.pth'
ENHANCED_MODEL_PATH = f'{MODEL_DIR}/best_enhanced_ace_rl_agent.pth'
TRAINING_STATS_PATH = f'{MODEL_DIR}/training_stats.json'
TRAINING_PLOT_PATH = f'{MODEL_DIR}/training_progress.png'

# Enhanced Agent Constants
CONFIDENCE_TEMPERATURE = 2.0
HIGH_CONFIDENCE_THRESHOLD = 0.6

# Evaluation Parameters
EVAL_GAMES = 200
FINAL_EVAL_GAMES = 1000
TEST_GAMES = 2000   