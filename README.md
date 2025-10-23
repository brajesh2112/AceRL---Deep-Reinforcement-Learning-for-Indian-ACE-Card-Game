# AceRL - Deep Q-Network Agent for Indian ACE Card Game

A Deep Reinforcement Learning agent that plays the Indian ACE card game using Deep Q-Network (DQN) with PyTorch. Features an interactive GUI dashboard with AI strategy suggestions.

## Game Rules

**Indian ACE** - 4 player card game where the goal is to empty your hand first.

- 52 cards dealt evenly (13 per player)
- Player with Ace of Spades starts
- Must follow suit if possible
- First player to empty hand wins

**Two Rule Variants:**

1. **Disappear Rule**: All same suit = cards disappear, mixed suits = highest wins all
2. **Cutting Rule**: Can't follow suit = cut and collect MORE cards

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/acerl.git
cd acerl

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 1.9+
- NumPy
- Matplotlib
- Tkinter

## Quick Start

**Train AI Agent:**
```bash
python train_agent.py
```

**Play Game with GUI:**
```bash
python ace_dashboard.py
```

## Project Structure

```
acerl/
├── game_engine.py       # Core game logic
├── train_agent.py       # DQN training script
├── ace_dashboard.py     # GUI dashboard
├── config.py            # Configuration
├── requirements.txt     # Dependencies
└── models/              # Saved models
```

## Technical Details

**DQN Architecture:**
- Input: 231-dimensional state (hand + history + context)
- Hidden Layers: 1024 → 512 → 256
- Output: 52 actions (one per card)
- Features: Prioritized replay, target network, uncertainty estimation

**Training:**
- Episodes: 3,000
- Learning Rate: 0.0001
- Batch Size: 128
- Memory: 200,000 experiences

**Performance:**
- Win Rate: 30-35% (vs 25% random baseline)
- Training Time: ~20 minutes on CPU

## Features

✅ Deep Q-Network with prioritized experience replay  
✅ Interactive GUI with card selection  
✅ Real-time AI strategy suggestions  
✅ Two game rule variants  
✅ Session statistics tracking  
✅ Model saving and loading  
✅ Training visualization  

## Usage

### Training
```bash
# Full training (3000 episodes)
python train_agent.py

# Quick test
python quick_train.py
```

### Playing
```bash
# Launch dashboard
python ace_dashboard.py

# Or use quick launcher
python quick_dashboard.py
```

### Dashboard Controls
- **New Game**: Start new game
- **AI Help**: Get AI suggestions
- **Play Card**: Play selected card
- **Rules**: View game rules

## Configuration

Edit `config.py` to adjust:
- Network architecture
- Learning rate and batch size
- Training episodes
- Reward structure

## Performance Grades

| Win Rate | Grade | Description |
|----------|-------|-------------|
| ≥35% | A+ | Outstanding |
| ≥30% | A | Excellent |
| ≥25% | B+ | Good |
| ≥20% | B | Learning |
| <20% | C | Developing |

## Files

- `game_engine.py` - Game mechanics, rules, bot AI
- `train_agent.py` - DQN agent training
- `ace_dashboard.py` - Tkinter GUI
- `config.py` - Hyperparameters
- `setup_project.py` - Setup automation
