# Player Re-Identification in a Single Feed

This project identifies and re-identifies soccer players in a 15-second video using a YOLOv11 object detector and CLIP embeddings.

## 🔧 Setup Instructions

1. **Install dependencies**:
pip install torch torchvision transformers pillow opencv-python scikit-learn

Project structure:

player_reid_project/          
├── input/                                
└── src/   

Run the code:

python -m src.detect_players
