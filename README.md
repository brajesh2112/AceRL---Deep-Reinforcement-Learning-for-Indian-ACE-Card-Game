# AceRL---Deep-Reinforcement-Learning-for-Indian-ACE-Card-Game
🎯 Project Overview
AceRL implements a deep reinforcement learning solution for the Indian ACE card game using PyTorch. The agent learns optimal strategies through self-play and can provide real-time strategic advice to human players through an intuitive Tkinter-based dashboard.
Key Features

✨ Deep Q-Network Agent with prioritized experience replay
🎮 Interactive GUI Dashboard with real-time game state visualization
🧠 AI Strategy Suggestions with confidence scores and reasoning
🔥 Two Game Rule Variants:

Disappear Rule: Same suit tricks make cards disappear
Cutting Rule: Unable to follow suit means cutting and collecting more cards


📊 Training Visualization with performance metrics and plots
💾 Model Persistence with checkpoint saving and loading
📈 Session Statistics tracking wins, performance, and improvement

🎲 Game Rules
Indian ACE Card Game
Objective: Be the first player to empty your hand completely.
Core Mechanics

4 players, 52-card deck dealt evenly (13 cards each)
Player with Ace of Spades starts the game
Must follow suit if possible
First to empty hand wins

Rule Variants
1. Disappear Rule (Default)

All 4 cards same suit → Cards disappear (no one collects them)
Mixed suits → Highest card wins all 4 cards

2. Cutting Rule

Can't follow suit → Player "cuts" with any card
Cutting player gets all cards played so far
Remaining players skip their turn
More cards = harder to win!

🚀 Quick Start
Prerequisites
bashPython 3.8+
pip (Python package manager)
Installation

Clone the repository

bashgit clone https://github.com/yourusername/acerl.git
cd acerl

Install dependencies

bashpip install -r requirements.txt

Run setup (optional)

bashpython setup_project.py
Running the Application
Train AI Agent
bashpython train_agent.py
Or use the quick launcher:
bashpython quick_train.py
Launch Game Dashboard
bashpython ace_dashboard.py
Or use the quick launcher:
bashpython quick_dashboard.py
📁 Project Structure
acerl/
│
├── game_engine.py          # Core game logic and mechanics
├── train_agent.py          # DQN agent training script
├── ace_dashboard.py        # Interactive GUI dashboard
├── config.py               # Configuration and hyperparameters
├── setup_project.py        # Project setup and initialization
├── requirements.txt        # Python dependencies
├── train.sh               # Training shell script
│
├── models/                 # Trained model storage
│   ├── best_ace_rl_agent.pth
│   └── training_progress.png
│
├── logs/                   # Training logs
├── screenshots/           # Dashboard screenshots
└── README.md              # This file
🧠 Technical Architecture
Deep Q-Network (DQN) Agent
Neural Network Architecture:

Input Layer: Game state encoding (52 cards + history + context)
Hidden Layers: 1024 → 512 → 256 neurons
Output Layer: 52 actions (one per card)
Additional: Uncertainty estimation head

Key Components:

Prioritized Experience Replay: Samples important experiences more frequently
Target Network: Stabilizes training with periodic updates
Epsilon-Greedy Exploration: Balances exploration vs exploitation
Enhanced Reward Shaping: Strategic rewards based on game phase

State Representation
pythonState Vector Components:
- Hand encoding (52 dims): Binary vector of cards in hand
- Played cards (52 dims): Public information of played cards
- Move history (120 dims): Recent 40 moves with metadata
- Game context (7 dims): Trick number, cards remaining, etc.

Total: 231-dimensional state vector
Reward Structure
pythonBase Rewards:
- Win trick: +2.5
- Lose trick: -0.75
- Game won: +5.0
- Game lost: -1.0

Strategic Bonuses:
- Conservative early play: +0.3
- Strategic late aggression: +0.75
- High cards in hand: +0.5
- Penalty for wasting high cards early: -1.5
📊 Training
Hyperparameters
pythonLEARNING_RATE = 0.0001
BATCH_SIZE = 128
MEMORY_SIZE = 200,000
GAMMA = 0.98
EPSILON_DECAY = 0.999
NUM_EPISODES = 3,000
Training Progress
The training script provides:

Episode rewards with moving averages
Win rate vs random baseline (25%)
Average tricks won per game
Training loss curves
Model uncertainty evolution

Performance Benchmarks
GradeWin RateDescriptionA+≥35%Outstanding performanceA≥30%Excellent strategic playB+≥25%Good, beats baselineB≥20%Learning progressC<20%Developing skills
🎮 Using the Dashboard
Main Features

Game Controls

New Game: Start fresh game
AI Help: Get strategic suggestions
Play Card: Execute selected move
Rules: View game rules


Your Hand

Click cards to select
Valid moves highlighted
Ace of Spades specially marked


AI Suggestions Panel

Top 3-5 strategic recommendations
Confidence scores (0-100%)
Detailed reasoning for each suggestion
Strategic tips based on game state


Players Status

Cards remaining in each hand
Cards collected per player
Current turn indicator


Game Log

Real-time move history
Trick results
Game events



AI Suggestion Example
🧠 AI INDIAN ACE STRATEGY (DISAPPEAR RULE)

🎯 SITUATION: You are LEADING this trick
💡 Your card choice sets the suit others must follow

🏆 TOP STRATEGIC RECOMMENDATIONS:

1. 🃏 8♣ (Value: 8)
   Strategy: LEAD 8♣ - Control trick
   Confidence: 85.3%
   Reasoning: Medium value card, safe lead

2. 🃏 6♥ (Value: 6)
   Strategy: LEAD 6♥ - Set suit
   Confidence: 78.1%
   Reasoning: Low risk, conserves high cards
🔬 Advanced Features
Model Adaptation
The agent supports transfer learning from existing models:
python# Load and adapt existing model
agent, model_info = smart_load_model('models/best_ace_rl_agent.pth')

# Continue training from checkpoint
python train_agent.py  # Automatically detects and loads
Custom Training
Modify config.py to adjust:

Network architecture (hidden layer sizes)
Learning rates and batch sizes
Reward structure
Exploration parameters

Evaluation Metrics
python# Evaluate agent performance
win_rate, avg_tricks = evaluate_agent_hidden_info(agent, num_games=200)

# Metrics tracked:
- Win rate vs random baseline
- Average tricks won
- Cards collected per game
- Strategic decision quality
📈 Results & Analysis
Training Results
After 3,000 episodes:

Win Rate: 30-35% (vs 25% random baseline)
Improvement: +20-40% over baseline
Convergence: Typically within 2,000 episodes
Training Time: 15-30 minutes (CPU)

Strategic Insights
The agent learns:

Early Game: Conservative play, preserve high cards
Mid Game: Balanced strategy, tactical wins
Late Game: Aggressive play, empty hand quickly
Suit Management: When to follow vs create mixed suits
Cutting Strategy: Risk/reward analysis of collecting cards
