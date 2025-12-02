# import cv2
# import numpy as np
# import pickle
# import os

# try:
#     # OpenCV >= 4.7 uses cv2.aruco
#     aruco = cv2.aruco
# except AttributeError as exc:  # pragma: no co8UYFGHer
#     raise ImportError(
#         "OpenCV ArUco module is required. Install opencv-contrib-python."
#     ) from exc


# class ImagePipeline:
#     def __init__(self):
#         # Load camera calibration
#         with open("cameraMatrix.pkl", 'rb') as f:
#             self.camera_matrix = np.asarray(pickle.load(f), dtype=float)
#         with open("dist.pkl", 'rb') as f:
#             self.dist_coeffs = np.asarray(pickle.load(f), dtype=float)

#         # Initialize ArUco detector
#         self.marker_length = 5.0 / 100.0  # 5cm in meters
#         self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
#         self.parameters = aruco.DetectorParameters()
#         self.corner_map = {0: 2, 1: 3, 2: 0, 3: 1}  # Maps marker ID to board corner (TL, TR, BR, BL)

#         # Instantiate the detector (newer OpenCV releases)
#         self.detector = None
#         if hasattr(aruco, "ArucoDetector"):
#             self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)

#     def detect_markers(self, frame):
#         """Detect ArUco markers and extract board corner points."""
#         if self.detector is not None:
#             corners_list, ids, _ = self.detector.detectMarkers(frame)
#         else:
#             corners_list, ids, _ = aruco.detectMarkers(
#                 frame, self.aruco_dict, parameters=self.parameters
#             )
#         if ids is None:
#             return None, None, None

#         ids = ids.flatten()
        
#         # Note: We don't need pose estimation for marker detection and warping
#         # The corner positions are sufficient for our purposes
        
#         # Get board corners from markers
#         src_pts = np.zeros((4, 2), dtype=np.float32)
#         for idx, mid in enumerate(ids):
#             mid = int(mid)
#             if mid not in self.corner_map:
#                 continue
#             pix = corners_list[idx].reshape(4, 2)
#             bidx = self.corner_map[mid]
#             src_pts[bidx] = pix[bidx]

#         return src_pts, corners_list, ids

#     def draw_grid(self, img, src_pts, grid_size=8, color=(0, 255, 0), thickness=2):
#         """Draw an 8×8 grid overlay on the image."""
#         tl, tr, br, bl = src_pts
#         for i in range(1, grid_size):
#             α = i / float(grid_size)
#             # Draw vertical lines
#             start = tuple((tl + α * (tr - tl)).astype(int))
#             end = tuple((bl + α * (br - bl)).astype(int))
#             cv2.line(img, start, end, color, thickness)
#             # Draw horizontal lines
#             start = tuple((tl + α * (bl - tl)).astype(int))
#             end = tuple((tr + α * (br - tr)).astype(int))
#             cv2.line(img, start, end, color, thickness)

#     def warp_to_topdown(self, frame, src_pts):
#         """Warp the image to a flat top-down view."""
#         # Compute grid size based on marker spacing
#         pixel_per_m = np.linalg.norm(src_pts[0] - src_pts[1]) / self.marker_length
#         PIX_PER_INCH = pixel_per_m * 0.0254
#         GRID_PIX = int(round(2.2 * PIX_PER_INCH))
#         size = GRID_PIX * 8

#         # Define destination points for perspective transform
#         dst_pts = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
        
#         # Compute perspective transform matrix
#         H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
#         # Warp the image
#         flat_view = cv2.warpPerspective(frame, H, (size, size))
        
#         return flat_view, GRID_PIX

#     def process_image(self, image_path):
#         """Process a single image: detect markers, draw grid, and warp to top-down view."""
#         # Load image
#         if not os.path.exists(image_path):
#             raise FileNotFoundError(f"Image not found: {image_path}")
        
#         frame = cv2.imread(image_path)
#         if frame is None:
#             raise ValueError(f"Could not read image: {image_path}")
        
#         print(f"Processing image: {image_path}")
        
#         # Detect ArUco markers
#         src_pts, corners_list, ids = self.detect_markers(frame)
#         if src_pts is None:
#             print("Error: No ArUco markers detected!")
#             return None, None
        
#         print(f"Detected {len(ids)} ArUco markers")                   
        
#         # Create a copy for drawing grid overlay
#         frame_with_grid = frame.copy()
        
#         # Draw 8×8 grid overlay
#         self.draw_grid(frame_with_grid, src_pts)
        
#         # Draw detected markers for visualization
#         if corners_list is not None and hasattr(aruco, 'drawDetectedMarkers'):
#             aruco.drawDetectedMarkers(frame_with_grid, corners_list, ids)
        
#         # Warp to flat top-down view
#         flat_view, grid_pix = self.warp_to_topdown(frame, src_pts)
        
#         # Draw grid lines on the warped image for clarity
#         size = grid_pix * 8
#         for i in range(9):
#             x = i * grid_pix
#             cv2.line(flat_view, (x, 0), (x, size), (255, 0, 0), 1)
#             cv2.line(flat_view, (0, x), (size, x), (255, 0, 0), 1)
        
#         return frame_with_grid, flat_view

# def main():
#     pipeline = ImagePipeline()
    
#     # Process chesstest.jpg
#     image_path = "test4.jpeg"
    
#     try:
#         frame_with_grid, flat_view = pipeline.process_image(image_path)
        
#         if frame_with_grid is None:
#             print("Failed to process image")
#             return
        
#         # Save results
#         output_grid = "chesstest_with_grid.png"
#         output_flat = "chesstest_topdown.png"
        
#         cv2.imwrite(output_grid, frame_with_grid)
#         cv2.imwrite(output_flat, flat_view)
        
#         print(f"\nResults saved:")
#         print(f"  - Grid overlay: {output_grid}")
#         print(f"  - Top-down view: {output_flat}")
        
#         # Display results
#         cv2.imshow('Original with Grid Overlay', frame_with_grid)
#         cv2.imshow('Warped Top-Down View', flat_view)
#         print("\nPress any key to close windows...")
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
        
#     except Exception as e:
#         print(f"Error: {e}")

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import os
import json
import pickle
from PIL import Image
import chess

from google import genai                  # google-genai package
from moveDetection import BoardState
from test_moves import get_lichess_best_move

# ---- ArUco module ----
try:
    aruco = cv2.aruco
except AttributeError as exc:
    raise ImportError(
        "OpenCV ArUco module is required. Install opencv-contrib-python."
    ) from exc


# ---------- MOVE DETECTION FROM BOARD + DETECTED STATE ----------

def detect_white_move_from_frame(board, detected_state):
    """
    Given a python-chess Board and an 8x8 detected_state array (strings),
    infer the WHITE move by comparing the board's previous layout with the
    newly detected layout.

    detected_state[r][c] must be one of: "empty", "piece-white", "piece-black".
    """
    previous_board_state = [['empty' for _ in range(8)] for _ in range(8)]

    # Build previous_board_state from the internal chess.Board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        file = chess.square_file(square)  # 0-7 (a-h)
        rank = chess.square_rank(square)  # 0-7 (1-8)
        col = 7 - file  # h->a maps to 0->7
        row = rank      # 1->8 maps to 0->7
        if piece:
            previous_board_state[row][col] = (
                "piece-white" if piece.color == chess.WHITE else "piece-black"
            )

    from_square = None
    to_square = None

    for row in range(8):
        for col in range(8):
            before = previous_board_state[row][col]
            after = detected_state[row][col]
            if before != after:
                file = 7 - col  # 0->7 maps to h->a
                rank = row      # 0->7 maps to 1->8
                square = chess.square(file, rank)
                if before != "empty" and after == "empty":
                    from_square = square
                elif before == "empty" and after != "empty":
                    to_square = square

    if from_square is None or to_square is None:
        raise Exception("Could not detect a valid move from the frame.")

    move_uci = chess.square_name(from_square) + chess.square_name(to_square)
    return move_uci


# ---------- MAIN PIPELINE CLASS ----------

class ImagePipeline:
    def __init__(self):
        # ---- Camera calibration ----
        with open("cameraMatrix.pkl", "rb") as f:
            self.camera_matrix = np.asarray(pickle.load(f), dtype=float)
        with open("dist.pkl", "rb") as f:
            self.dist_coeffs = np.asarray(pickle.load(f), dtype=float)

        # ---- ArUco setup ----
        self.marker_length = 5.0 / 100.0  # 5 cm in meters
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()

        # Maps marker ID -> board corner index (TL, TR, BR, BL)
        self.corner_map = {0: 2, 1: 3, 2: 0, 3: 1}

        # Newer OpenCV releases have ArucoDetector
        self.detector = None
        if hasattr(aruco, "ArucoDetector"):
            self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)

        # ---- Chess state ----
        self.board_state = BoardState()   # expects .board to be a chess.Board()

        # ---- Gemini API keys ----
        self.keys = [
            os.getenv("GEMINI_API_KEY"),
            os.getenv("GEMINI_API_KEY2"),
            os.getenv("GEMINI_API_KEY3"),
            os.getenv("GEMINI_API_KEY4"),
        ]
        self.current_key_index = 0

        # ---- Output dirs ----
        os.makedirs("grid_piece", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)

    # ---------- ARUCO + GEOMETRY ----------

    def detect_markers(self, frame):
        """Detect ArUco markers and extract board corner points."""
        if self.detector is not None:
            corners_list, ids, _ = self.detector.detectMarkers(frame)
        else:
            corners_list, ids, _ = aruco.detectMarkers(
                frame, self.aruco_dict, parameters=self.parameters
            )

        if ids is None or len(corners_list) == 0:
            return None, None, None

        ids = ids.flatten()

        # Get board corners from markers
        src_pts = np.zeros((4, 2), dtype=np.float32)
        for idx, mid in enumerate(ids):
            mid = int(mid)
            if mid not in self.corner_map:
                continue
            pix = corners_list[idx].reshape(4, 2)
            bidx = self.corner_map[mid]
            src_pts[bidx] = pix[bidx]

        # Ensure we actually filled something
        if np.all(src_pts == 0):
            return None, corners_list, ids

        return src_pts, corners_list, ids

    def draw_grid(self, img, src_pts, grid_size=8, color=(0, 255, 0), thickness=2):
        """Draw an 8×8 grid overlay on the image."""
        tl, tr, br, bl = src_pts
        for i in range(1, grid_size):
            alpha = i / float(grid_size)
            # Vertical lines
            start = tuple((tl + alpha * (tr - tl)).astype(int))
            end = tuple((bl + alpha * (br - bl)).astype(int))
            cv2.line(img, start, end, color, thickness)
            # Horizontal lines
            start = tuple((tl + alpha * (bl - tl)).astype(int))
            end = tuple((tr + alpha * (br - tr)).astype(int))
            cv2.line(img, start, end, color, thickness)

    def warp_to_topdown(self, frame, src_pts):
        """Warp the image to a flat top-down view and return (flat_view, GRID_PIX)."""
        pixel_per_m = np.linalg.norm(src_pts[0] - src_pts[1]) / self.marker_length
        PIX_PER_M = pixel_per_m
        PIX_PER_INCH = PIX_PER_M * 0.0254
        GRID_PIX = int(round(2.2 * PIX_PER_INCH))  # ~2.2 inch squares
        size = GRID_PIX * 8

        dst_pts = np.array(
            [[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32
        )

        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        flat_view = cv2.warpPerspective(frame, H, (size, size))

        return flat_view, GRID_PIX

    # ---------- GRID CELL HANDLING ----------

    def save_grid_cells(self, flat_view, grid_pix):
        """
        Slice the warped board into 8×8 cells and save each as:
        grid_piece/piece_r{r}_c{c}.png
        """
        size = grid_pix * 8
        h, w = flat_view.shape[:2]
        if h < size or w < size:
            raise ValueError("Warped image is smaller than expected grid size.")

        for r in range(8):
            for c in range(8):
                y0, y1 = r * grid_pix, (r + 1) * grid_pix
                x0, x1 = c * grid_pix, (c + 1) * grid_pix
                cell = flat_view[y0:y1, x0:x1]
                out_path = os.path.join("grid_piece", f"piece_r{r}_c{c}.png")
                cv2.imwrite(out_path, cell)

    # ---------- GEMINI CLASSIFICATION ----------

    def classify_pieces(self):
        """
        Use Gemini to classify each cell as:
        - "empty"
        - "piece-white"
        - "piece-black"

        Returns: list of (r, c, label)
        """
        results = []
        for r in range(8):
            for c in range(8):
                filename = f"piece_r{r}_c{c}.png"
                image_path = os.path.join("grid_piece", filename)

                if not os.path.isfile(image_path):
                    continue

                label = "empty"
                attempts = 0

                while attempts < len(self.keys):
                    key = self.keys[self.current_key_index]

                    if not key:
                        # skip empty key
                        self.current_key_index = (self.current_key_index + 1) % len(self.keys)
                        attempts += 1
                        continue

                    try:
                        client = genai.Client(api_key=key)
                        img = Image.open(image_path)

                        prompt = (
                            "You are analyzing a single square from a real chessboard, "
                            "top-down view. Classify the CONTENT of this square into exactly "
                            "ONE of these labels:\n"
                            "- empty\n"
                            "- piece-white\n"
                            "- piece-black\n"
                            "Reply with JUST the label, nothing else."
                        )

                        response = client.models.generate_content(
                            model="gemini-2.0-flash-exp",
                            contents=[img, prompt],
                        )

                        label = response.text.strip().lower()
                        # Basic safety: normalize unexpected responses
                        if label not in ["empty", "piece-white", "piece-black"]:
                            label = "empty"
                        break

                    except Exception as e:
                        print(f"Error with key {key}: {e}")
                        self.current_key_index = (self.current_key_index + 1) % len(self.keys)
                        attempts += 1

                results.append((r, c, label))

        return results

    # ---------- DETECTION HISTORY ----------

    def update_detection_json(self, results):
        """Append the current frame classification results into detection.json."""
        detection_file = "detection.json"

        if os.path.exists(detection_file):
            with open(detection_file, "r") as f:
                data = json.load(f)
        else:
            data = []

        new_id = len(data) + 1
        run_entry = {
            "id": new_id,
            "results": [
                {"r": r, "c": c, "label": label}
                for r, c, label in results
            ],
        }

        data.append(run_entry)
        with open(detection_file, "w") as f:
            json.dump(data, f, indent=2)

    # ---------- MOVE LOGIC (WHITE + LICHESS BLACK) ----------

    def process_moves(self):
        """
        Use the last detection frame to infer White's move from the internal board,
        then query Lichess for Black's best move and apply it.
        """
        detection_file = "detection.json"
        if not os.path.exists(detection_file):
            print("No detection.json found; skipping move processing.")
            return

        try:
            with open(detection_file, "r") as f:
                data = json.load(f)

            if len(data) < 1:
                print("No detection frames in detection.json.")
                return

            curr_frame = data[-1]

            # Build detected_state from last frame
            curr_state = [["empty" for _ in range(8)] for _ in range(8)]
            for cell in curr_frame["results"]:
                curr_state[cell["r"]][cell["c"]] = cell["label"]

            # Detect White's move
            detected_white_move = detect_white_move_from_frame(
                self.board_state.board,
                curr_state,
            )
            white_move_obj = chess.Move.from_uci(detected_white_move)

            if white_move_obj in self.board_state.board.legal_moves:
                print(f"White move: {detected_white_move}")
                self.board_state.board.push(white_move_obj)
                print("\nBoard after White move:")
                print(self.board_state.board)
                fen_after_white = self.board_state.board.fen()
                print(f"FEN after White move: {fen_after_white}")

                # Get Lichess best move for Black
                best_black_move = get_lichess_best_move(fen_after_white)
                print(f"Lichess best move for Black: {best_black_move}")
                black_move_obj = chess.Move.from_uci(best_black_move)

                if black_move_obj in self.board_state.board.legal_moves:
                    self.board_state.board.push(black_move_obj)
                    print("\nBoard after Black move:")
                    print(self.board_state.board)
                    print(f"FEN after Black move: {self.board_state.board.fen()}")
                else:
                    print(f"⚠ Illegal Black move suggested: {best_black_move}")
            else:
                print(f"⚠ Illegal White move detected: {detected_white_move}")

        except Exception as e:
            print(f"Error processing moves: {e}")

    # ---------- END-TO-END ON ONE IMAGE ----------

    def process_image(self, image_path):
        """
        Main pipeline for a single image:
        - load image
        - detect ArUco markers
        - warp to top-down
        - save 8x8 grid cells
        - classify each cell (Gemini)
        - update detection.json
        - detect White move + Lichess best reply

        Returns: (original_with_grid, flat_topdown)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        print(f"Processing image: {image_path}")

        # Detect markers
        src_pts, corners_list, ids = self.detect_markers(frame)
        if src_pts is None:
            print("No valid ArUco markers detected or not enough corners!")
            return None, None

        print(f"Detected {len(ids)} ArUco markers.")

        # Draw grid on original image
        frame_with_grid = frame.copy()
        self.draw_grid(frame_with_grid, src_pts)

        if corners_list is not None and hasattr(aruco, "drawDetectedMarkers"):
            aruco.drawDetectedMarkers(frame_with_grid, corners_list, ids)

        # Warp to top-down
        flat_view, grid_pix = self.warp_to_topdown(frame, src_pts)

        # Draw grid lines on warped image (for visualization)
        size = grid_pix * 8
        for i in range(9):
            x = i * grid_pix
            cv2.line(flat_view, (x, 0), (x, size), (255, 0, 0), 1)
            cv2.line(flat_view, (0, x), (size, x), (255, 0, 0), 1)

        # Save each cell
        self.save_grid_cells(flat_view, grid_pix)
        print("Saved 8x8 grid cells in grid_piece/")

        # Classify with Gemini
        results = self.classify_pieces()
        print("Classification complete for all cells.")

        # Update detection history
        self.update_detection_json(results)
        print("Updated detection.json")

        # Try detecting moves
        self.process_moves()

        return frame_with_grid, flat_view


# ---------- MAIN ----------

def main():
    image_path = "test4.jpeg"

    pipeline = ImagePipeline()
    orig_with_grid, flat_topdown = pipeline.process_image(image_path)

    if orig_with_grid is None or flat_topdown is None:
        print("Failed to fully process image.")
        return

    grid_path = os.path.join("outputs", "board_with_grid.png")
    topdown_path = os.path.join("outputs", "board_topdown.png")

    cv2.imwrite(grid_path, orig_with_grid)
    cv2.imwrite(topdown_path, flat_topdown)

    print("\nResults saved:")
    print(f"  - Grid overlay: {grid_path}")
    print(f"  - Top-down view: {topdown_path}")

    cv2.imshow("Original with Grid Overlay", orig_with_grid)
    cv2.imshow("Warped Top-Down View", flat_topdown)
    print("\nPress any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
