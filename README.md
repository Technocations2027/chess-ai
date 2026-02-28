# Chess Screen AI

This project watches your Chess dot com board on screen, reads the position with YOLO models, turns that into a FEN string, and then asks Stockfish for the best move. The move and FEN are drawn right on top of the live video.

CHESSCOMBOARD.pt not included due to file size.

## Features

- Detects the Chess dot com board area with a YOLO board model
- Detects each piece with a YOLO piece model
- Maps piece boxes to the correct chess squares
- Builds a FEN string for the position
- Sends the FEN to Stockfish and shows the best move
- Draws an optional grid overlay and debug labels

## Requirements

- Python 3 dot 10 or newer
- OpenCV
- NumPy
- ultralytics YOLO
- python chess
- Stockfish installed on your system

## Install

Clone the repo:

```bash
git clone https://github.com/Technocations2027/chess-screen-ai.git
cd chess-screen-ai


python -m venv .venv
source .venv/bin/activate   # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt


python -m venv .venv
source .venv/bin/activate   # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
