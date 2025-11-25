import cv2
import numpy as np
import pickle
import os

try:
    # OpenCV >= 4.7 uses cv2.aruco
    aruco = cv2.aruco
except AttributeError as exc:  # pragma: no co8UYFGHer
    raise ImportError(
        "OpenCV ArUco module is required. Install opencv-contrib-python."
    ) from exc


class ImagePipeline:
    def __init__(self):
        # Load camera calibration
        with open("cameraMatrix.pkl", 'rb') as f:
            self.camera_matrix = np.asarray(pickle.load(f), dtype=float)
        with open("dist.pkl", 'rb') as f:
            self.dist_coeffs = np.asarray(pickle.load(f), dtype=float)

        # Initialize ArUco detector
        self.marker_length = 5.0 / 100.0  # 5cm in meters
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        self.corner_map = {0: 2, 1: 3, 2: 0, 3: 1}  # Maps marker ID to board corner (TL, TR, BR, BL)

        # Instantiate the detector (newer OpenCV releases)
        self.detector = None
        if hasattr(aruco, "ArucoDetector"):
            self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)

    def detect_markers(self, frame):
        """Detect ArUco markers and extract board corner points."""
        if self.detector is not None:
            corners_list, ids, _ = self.detector.detectMarkers(frame)
        else:
            corners_list, ids, _ = aruco.detectMarkers(
                frame, self.aruco_dict, parameters=self.parameters
            )
        if ids is None:
            return None, None, None

        ids = ids.flatten()
        
        # Note: We don't need pose estimation for marker detection and warping
        # The corner positions are sufficient for our purposes
        
        # Get board corners from markers
        src_pts = np.zeros((4, 2), dtype=np.float32)
        for idx, mid in enumerate(ids):
            mid = int(mid)
            if mid not in self.corner_map:
                continue
            pix = corners_list[idx].reshape(4, 2)
            bidx = self.corner_map[mid]
            src_pts[bidx] = pix[bidx]

        return src_pts, corners_list, ids

    def draw_grid(self, img, src_pts, grid_size=8, color=(0, 255, 0), thickness=2):
        """Draw an 8×8 grid overlay on the image."""
        tl, tr, br, bl = src_pts
        for i in range(1, grid_size):
            α = i / float(grid_size)
            # Draw vertical lines
            start = tuple((tl + α * (tr - tl)).astype(int))
            end = tuple((bl + α * (br - bl)).astype(int))
            cv2.line(img, start, end, color, thickness)
            # Draw horizontal lines
            start = tuple((tl + α * (bl - tl)).astype(int))
            end = tuple((tr + α * (br - tr)).astype(int))
            cv2.line(img, start, end, color, thickness)

    def warp_to_topdown(self, frame, src_pts):
        """Warp the image to a flat top-down view."""
        # Compute grid size based on marker spacing
        pixel_per_m = np.linalg.norm(src_pts[0] - src_pts[1]) / self.marker_length
        PIX_PER_INCH = pixel_per_m * 0.0254
        GRID_PIX = int(round(2.2 * PIX_PER_INCH))
        size = GRID_PIX * 8

        # Define destination points for perspective transform
        dst_pts = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
        
        # Compute perspective transform matrix
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Warp the image
        flat_view = cv2.warpPerspective(frame, H, (size, size))
        
        return flat_view, GRID_PIX

    def process_image(self, image_path):
        """Process a single image: detect markers, draw grid, and warp to top-down view."""
        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        print(f"Processing image: {image_path}")
        
        # Detect ArUco markers
        src_pts, corners_list, ids = self.detect_markers(frame)
        if src_pts is None:
            print("Error: No ArUco markers detected!")
            return None, None
        
        print(f"Detected {len(ids)} ArUco markers")                   
        
        # Create a copy for drawing grid overlay
        frame_with_grid = frame.copy()
        
        # Draw 8×8 grid overlay
        self.draw_grid(frame_with_grid, src_pts)
        
        # Draw detected markers for visualization
        if corners_list is not None and hasattr(aruco, 'drawDetectedMarkers'):
            aruco.drawDetectedMarkers(frame_with_grid, corners_list, ids)
        
        # Warp to flat top-down view
        flat_view, grid_pix = self.warp_to_topdown(frame, src_pts)
        
        # Draw grid lines on the warped image for clarity
        size = grid_pix * 8
        for i in range(9):
            x = i * grid_pix
            cv2.line(flat_view, (x, 0), (x, size), (255, 0, 0), 1)
            cv2.line(flat_view, (0, x), (size, x), (255, 0, 0), 1)
        
        return frame_with_grid, flat_view

def main():
    pipeline = ImagePipeline()
    
    # Process chesstest.jpg
    image_path = "chesstest.jpg"
    
    try:
        frame_with_grid, flat_view = pipeline.process_image(image_path)
        
        if frame_with_grid is None:
            print("Failed to process image")
            return
        
        # Save results
        output_grid = "chesstest_with_grid.png"
        output_flat = "chesstest_topdown.png"
        
        cv2.imwrite(output_grid, frame_with_grid)
        cv2.imwrite(output_flat, flat_view)
        
        print(f"\nResults saved:")
        print(f"  - Grid overlay: {output_grid}")
        print(f"  - Top-down view: {output_flat}")
        
        # Display results
        cv2.imshow('Original with Grid Overlay', frame_with_grid)
        cv2.imshow('Warped Top-Down View', flat_view)
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

