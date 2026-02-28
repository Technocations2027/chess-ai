# chess_screen_chesscom.py
# Chess.com screen reader: YOLO board box + YOLO pieces -> FEN -> Stockfish best move
# Assumes the Chess.com board is always in the same place on screen (OBS virtual cam or direct capture)

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict
from ultralytics import YOLO
import chess
import chess.engine

# ── CONFIG ────────────────────────────────────────────────────────────────
CAM_INDEX = 1            # 0 or 1 depending on OBS / camera routing
DEBUG_MODE = True
PIECE_DEBUG = True

BOARD_MODEL_PATH = "ChessComBoard.pt"
PIECE_MODEL_PATH = "ChessComModel.pt"

GRID_OVERLAY = True           # Draw Chess.com 8x8 grid overlay on the screen
SHOW_TOPDOWN = False          # If True, also show a warped top-down view (not required for correctness)

TOPDOWN_SIZE = 800            # Size of warped board if SHOW_TOPDOWN is True

# Padding inside the YOLO board box to ignore Chess.com borders (tune these)
PADDING_TOP = 10
PADDING_BOTTOM = 10
PADDING_LEFT = 10
PADDING_RIGHT = 10

STOCKFISH_PATH = "/usr/local/bin/stockfish"
current_turn = 'w'            # You can toggle manually with 't' for now
last_fen: Optional[str] = None
fen_stable_frames = 0
FEN_STABILITY_THRESHOLD = 1.5

# ── LOAD MODELS ───────────────────────────────────────────────────────────
try:
    board_model = YOLO(BOARD_MODEL_PATH)
    print(f"Loaded board model: {BOARD_MODEL_PATH}")
except Exception as e:
    print(f"ERROR loading {BOARD_MODEL_PATH}: {e}")
    exit(1)

try:
    piece_model = YOLO(PIECE_MODEL_PATH)
    print(f"Loaded piece model: {PIECE_MODEL_PATH}")
except Exception as e:
    print(f"ERROR loading {PIECE_MODEL_PATH}: {e}")
    exit(1)

class PieceDetector:
    def __init__(self, model: YOLO, conf: float = 0.30, iou: float = 0.50, device: str = "cpu"):
        self.model = model
        self.conf = conf
        self.iou = iou
        self.device = device
        self.class_names: Dict[int, str] = self.model.names

    def detect(self, img_bgr: np.ndarray):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.model.predict(
            img_rgb,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )
        detections = []
        if not results:
            return detections
        r = results[0]
        boxes = r.boxes
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            conf = float(b.conf[0].cpu().numpy())
            cls_id = int(b.cls[0].cpu().numpy())
            detections.append((xyxy, conf, cls_id))
        return detections

PIECE_CONFIDENCE = 0.15
piece_detector = PieceDetector(piece_model, conf=PIECE_CONFIDENCE, iou=0.50, device="cpu")

# ── STOCKFISH INIT ────────────────────────────────────────────────────────
engine = None
try:
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    print(f"Stockfish loaded from: {STOCKFISH_PATH}")
except FileNotFoundError:
    print(f"WARNING: Stockfish not found at {STOCKFISH_PATH}")
except Exception as e:
    print(f"WARNING: Stockfish error: {e}")

def get_best_move_from_fen(fen: str, movetime_ms: int = 500) -> Optional[str]:
    if engine is None:
        return None
    if fen.startswith("8/8/8/8/8/8/8/8"):
        return None
    try:
        board = chess.Board(fen)
        if not board.is_valid():
            if DEBUG_MODE:
                print("Invalid board:", board.status())
            return None
        if board.legal_moves.count() == 0:
            return None
        result = engine.play(board, chess.engine.Limit(time=movetime_ms / 1000.0))
        return result.move.uci()
    except Exception as e:
        if DEBUG_MODE:
            print("Stockfish error:", e)
        return None

# ── GEOMETRY HELPERS ──────────────────────────────────────────────────────

def box_center(xyxy: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ( (x1 + x2) / 2.0, (y1 + y2) / 2.0 )

def square_from_screen(
    pt: Tuple[float, float],
    board_box: Tuple[int, int, int, int]
) -> Tuple[int, int]:
    """
    Map a point in screen coordinates to (file_idx, rank_idx) 0..7
    board_box = (x1, y1, x2, y2) of the tightened board grid area.
    """
    x, y = pt
    x1, y1, x2, y2 = board_box
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return 0, 0
    fx = np.clip(int((x - x1) * 8.0 / w), 0, 7)
    fr = np.clip(int((y - y1) * 8.0 / h), 0, 7)
    return fx, fr

def algebraic_from_indices(f: int, r: int) -> str:
    files = "abcdefgh"
    ranks = "87654321"
    return f"{files[f]}{ranks[r]}"

CLASS_TO_FEN = {
    "white-pawn": "P", "white-knight": "N", "white-bishop": "B",
    "white-rook": "R", "white-queen": "Q", "white-king": "K",
    "black-pawn": "p", "black-knight": "n", "black-bishop": "b",
    "black-rook": "r", "black-queen": "q", "black-king": "k",
}

CLASS_SHORT_NAMES = {
    "white-pawn": "wP", "white-knight": "wN", "white-bishop": "wB",
    "white-rook": "wR", "white-queen": "wQ", "white-king": "wK",
    "black-pawn": "bP", "black-knight": "bN", "black-bishop": "bB",
    "black-rook": "bR", "black-queen": "bQ", "black-king": "bK",
}

def build_fen_from_detections(
    dets: List[Tuple[str, Tuple[int, int]]],
    turn: str = 'w'
) -> str:
    board = [["" for _ in range(8)] for _ in range(8)]
    for cname, (f, r) in dets:
        fen_ch = CLASS_TO_FEN.get(cname, "")
        if fen_ch:
            board[r][f] = fen_ch
    fen_rows = []
    for r in range(8):
        row = board[r]
        empties = 0
        fen_row = ""
        for f in range(8):
            if row[f] == "":
                empties += 1
            else:
                if empties > 0:
                    fen_row += str(empties)
                    empties = 0
                fen_row += row[f]
        if empties > 0:
            fen_row += str(empties)
        fen_rows.append(fen_row if fen_row else "8")
    fen_board = "/".join(fen_rows)
    # No castling / ep tracking here; just give Stockfish a static board
    return f"{fen_board} {turn} - - 0 1"

def update_turn_on_fen_change(
    current_fen: str,
    last_fen: Optional[str],
    stable_frames: int,
    current_turn: str,
    threshold: int
) -> Tuple[str, str, int]:
    """
    If FEN changes and stays stable for N frames, flip the side to move.
    Returns (new_turn, new_last_fen, new_stable_frames).
    """
    if last_fen is None:
        # First frame with a FEN
        return current_turn, current_fen, 1

    if current_fen == last_fen:
        # No change, increment stability
        stable_frames += 1
        return current_turn, last_fen, stable_frames

    # FEN changed
    # If we were stable for long enough before this change, treat it as a move
    if stable_frames >= threshold:
        new_turn = 'b' if current_turn == 'w' else 'w'
        if DEBUG_MODE:
            print(f"Move detected → turn flipped to: {new_turn}, FEN: {current_fen}")
        return new_turn, current_fen, 1  # reset stability for new FEN

    # Change came too soon (noise) → update FEN but don't flip turn
    if DEBUG_MODE:
        print(f"FEN changed but not stable long enough (stable_frames={stable_frames})")
    return current_turn, current_fen, 1


# ── BOARD DETECTION ───────────────────────────────────────────────────────

last_board_box: Optional[Tuple[int, int, int, int]] = None

def detect_board_box(frame_bgr: np.ndarray, display_frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Use ChessComBoard.pt to detect the whole board on the Chess.com screen.
    For now, do NOT square or shrink aggressively – just trust YOLO's box
    and (optionally) apply small tunable paddings.
    Returns (x1, y1, x2, y2) or last known box.
    """
    global last_board_box
    
    results = board_model.predict(
        frame_bgr,
        conf=0.4,
        iou=0.4,
        imgsz=640,
        verbose=False,
    )
    if not results or len(results[0].boxes) == 0:
        return last_board_box
    
    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    
    # Highest-confidence detection
    idx = int(np.argmax(confs))
    x1, y1, x2, y2 = boxes[idx]
    
    # OPTIONAL: very small border trim, start with 0s and tune slowly
    pad_top    = 0
    pad_bottom = 0
    pad_left   = 0
    pad_right  = 0
    
    x1_t = x1 + pad_left
    y1_t = y1 + pad_top
    x2_t = x2 - pad_right
    y2_t = y2 - pad_bottom
    
    # Clamp
    H, W = frame_bgr.shape[:2]
    x1_t = max(0, x1_t)
    y1_t = max(0, y1_t)
    x2_t = min(W, x2_t)
    y2_t = min(H, y2_t)
    
    tightened = (x1_t, y1_t, x2_t, y2_t)
    
    # Draw cyan for raw YOLO, magenta for grid area (now almost the same)
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)     # raw boardBox
    cv2.rectangle(display_frame, (x1_t, y1_t), (x2_t, y2_t), (255, 0, 255), 2)  # grid box
    
    last_board_box = tightened
    return tightened


def draw_grid_on_screen(display: np.ndarray, board_box: Tuple[int, int, int, int]) -> None:
    """Draw 8x8 grid lines over the Chess.com board area."""
    x1, y1, x2, y2 = board_box
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return
    cell_w = w / 8.0
    cell_h = h / 8.0
    for i in range(9):
        # vertical
        x = int(x1 + i * cell_w)
        cv2.line(display, (x, y1), (x, y2), (255, 0, 255), 1)
        # horizontal
        y = int(y1 + i * cell_h)
        cv2.line(display, (x1, y), (x2, y), (255, 0, 255), 1)

# ── CAMERA SETUP ──────────────────────────────────────────────────────────

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)

print("\n=== Chess.com Screen AI ===")
print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"Stockfish: {'Available' if engine else 'Not Available'}")
print("Controls: q=quit, d=toggle debug labels, g=toggle grid, t=toggle turn")
print()

cv2.namedWindow("Chess AI")

latest_fen = None
latest_best_move = None

# ── MAIN LOOP ─────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display = frame.copy()
    
    # 1) Detect board on Chess.com screen
    board_box = detect_board_box(frame, display)
    
    if board_box is not None:
        x1, y1, x2, y2 = board_box
        
        # 2) Draw 8x8 grid overlay
        if GRID_OVERLAY:
            draw_grid_on_screen(display, board_box)
        
        # 3) Crop board region and run piece detector
        board_region = frame[y1:y2, x1:x2]
        piece_dets = piece_detector.detect(board_region)
        
        # Limit to 32 strongest detections
        if len(piece_dets) > 32:
            piece_dets = sorted(piece_dets, key=lambda x: x[1], reverse=True)[:32]
        
        dets_for_fen: List[Tuple[str, Tuple[int, int]]] = []
        
        for (xyxy, conf, cls_id) in piece_dets:
            cname = piece_detector.class_names.get(cls_id, str(cls_id))
            # Center (relative to board_region)
            cx_rel, cy_rel = box_center(xyxy)
            # Map to screen coords
            cx_screen = x1 + cx_rel * ( (x2 - x1) / board_region.shape[1] )
            cy_screen = y1 + cy_rel * ( (y2 - y1) / board_region.shape[0] )
            
            f_idx, r_idx = square_from_screen((cx_screen, cy_screen), board_box)
            square_alg = algebraic_from_indices(f_idx, r_idx)
            dets_for_fen.append((cname, (f_idx, r_idx)))
            
            if PIECE_DEBUG:
                # Yellow rectangle for piece in screen coords
                bx1 = int(x1 + xyxy[0] * ( (x2 - x1) / board_region.shape[1] ))
                by1 = int(y1 + xyxy[1] * ( (y2 - y1) / board_region.shape[0] ))
                bx2 = int(x1 + xyxy[2] * ( (x2 - x1) / board_region.shape[1] ))
                by2 = int(y1 + xyxy[3] * ( (y2 - y1) / board_region.shape[0] ))
                
                cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                
                short_name = CLASS_SHORT_NAMES.get(cname, cname[:2])
                label = f"{short_name} @ {square_alg}"
                
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                lx = int(cx_screen) - tw // 2
                ly = by1 - 5
                
                cv2.rectangle(display, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), (0, 0, 0), -1)
                cv2.putText(display, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 4) Build FEN
        latest_fen = build_fen_from_detections(dets_for_fen, current_turn)
        if DEBUG_MODE:
            print("FEN:", latest_fen)

        # 5) Auto-detect turn based on stable FEN changes
        current_turn, last_fen, fen_stable_frames = update_turn_on_fen_change(
            latest_fen,
            last_fen,
            fen_stable_frames,
            current_turn,
            FEN_STABILITY_THRESHOLD
        )

        # 6) Call Stockfish on the stable FEN
        latest_best_move = get_best_move_from_fen(latest_fen)

        # Overlay FEN + best move
        fen_display = latest_fen[:60] + "..." if len(latest_fen) > 60 else latest_fen
        cv2.putText(display, f"FEN: {fen_display}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        turn_text = "White" if current_turn == 'w' else "Black"
        cv2.putText(display, f"Turn: {turn_text}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if latest_best_move:
            cv2.putText(display, f"{latest_best_move}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.putText(display, f"Turn: {turn_text}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if latest_best_move:
            cv2.putText(display, f"{latest_best_move}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    else:
        cv2.putText(display, "No board detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Chess AI", display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        PIECE_DEBUG = not PIECE_DEBUG
        print("Piece debug:", "ON" if PIECE_DEBUG else "OFF")
    elif key == ord('g'):
        GRID_OVERLAY = not GRID_OVERLAY
        print("Grid overlay:", "ON" if GRID_OVERLAY else "OFF")
    elif key == ord('t'):
        current_turn = 'b' if current_turn == 'w' else 'w'
        print("Turn:", "White" if current_turn == 'w' else "Black")

cap.release()
if engine is not None:
    engine.quit()
cv2.destroyAllWindows()
