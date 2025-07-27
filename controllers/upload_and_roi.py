import fitz  # PyMuPDF
import cv2
import numpy as np
import os
from PIL import Image
import logging
from typing import Tuple, Optional
from flask import Blueprint, jsonify, request
import json
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDFGridCropper:
    """Professional PDF to grid cropping tool with configurable parameters."""
    
    def __init__(self, dpi: int = 400):
        """
        Initialize the PDF Grid Cropper.
        
        Args:
            dpi: Resolution for PDF conversion (default: 400)
        """
        self.dpi = dpi
        
    def pdf_to_png(self, pdf_path: str, output_path: str = "output.png") -> Optional[Tuple[int, int]]:
        """
        Convert PDF to high-resolution PNG.
        
        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output PNG file
            
        Returns:
            Tuple of (width, height) if successful, None if failed
        """
        try:
            pdf_document = fitz.open(pdf_path)
            page = pdf_document[0]
            
            zoom = self.dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            pix.save(output_path)
            width, height = pix.width, pix.height
            pdf_document.close()
            
            logger.info(f"PDF converted successfully: {width}x{height} pixels at {self.dpi} DPI")
            return width, height
            
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            return None

    def extract_and_save_roi(self, img, bbox, row_idx, col_idx, padding=5):
        """Extract ROI dan simpan sebagai gambar terpisah"""
        x, y, w, h = bbox

        # Add padding with bounds checking
        img_h, img_w = img.shape[:2]
        x_pad = max(0, x - padding)
        y_pad = max(0, y - padding)
        x_end = min(img_w, x + w + padding)
        y_end = min(img_h, y + h + padding)

        # Crop region dengan padding
        roi = img[y_pad:y_end, x_pad:x_end]

        # Check if ROI is valid
        if roi.size == 0:
            print(f"Empty ROI for row{row_idx}col{col_idx}")
            return None

        # Resize jika terlalu kecil (opsional untuk konsistensi)
        min_size = 20
        if roi.shape[0] < min_size or roi.shape[1] < min_size:
            scale_factor = max(min_size/roi.shape[0], min_size/roi.shape[1])
            new_width = int(roi.shape[1] * scale_factor)
            new_height = int(roi.shape[0] * scale_factor)
            roi = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Save ROI with better filename formatting
        filename = f"row{row_idx}col{col_idx}.png"
        filepath = os.path.join(self.output_dir, filename)

        try:
            success = cv2.imwrite(filepath, roi)
            if success:
                print(f"Saved: {filename} (size: {roi.shape[1]}x{roi.shape[0]}, bbox: {bbox})")
                return filepath
            else:
                print(f"Failed to save: {filename}")
                return None
        except Exception as e:
            print(f"Error saving {filename}: {e}")
            return None

    def sort_regions_bottom_to_top(self, regions):
        """Sort regions dari bawah ke atas (row0 = paling bawah)"""
        if not regions:
            return []
        
        # Sort berdasarkan center_y secara descending (bawah ke atas)
        # Kemudian berdasarkan x secara ascending (kiri ke kanan)
        sorted_regions = sorted(regions, key=lambda r: (-r['center_y'], r['bbox'][0]))
        return sorted_regions

    def find_digit_regions(self, processed_img, min_area=50, max_area=5000):
        """Deteksi region yang mengandung digit menggunakan contour detection"""
        # Apply closing untuk menggabungkan bagian karakter yang terpisah
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digit_regions = []

        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0

            # Filter berdasarkan ukuran dan aspect ratio yang masuk akal untuk digit
            if (min_area <= area <= max_area and
                0.1 <= aspect_ratio <= 2.0 and
                h >= 10 and w >= 5):  # Minimal size untuk digit

                digit_regions.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'center_y': y + h//2  # Untuk sorting berdasarkan posisi vertikal
                })

        return digit_regions

    def crop_roi(self, image_path: str, output_dir: str = "crops",
                start_x: float = 65, start_y: float = 4588,
                width: float = 100, height: float = -4574,
                x_increment: float = 170,
                cols: int = 40) -> int:
        """
        CROP ROI using get bounding rectangle algorithm
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Cannot read image: {image_path}")
                return 0
                
            # Create output directory
            self.output_dir = output_dir  # Store for use in extract_and_save_roi
            os.makedirs(output_dir, exist_ok=True)

            # Calculate y coordinates and ensure correct order
            y1 = int(start_y)
            y2 = int(start_y + height)
            if y1 > y2:
                y1, y2 = y2, y1

            saved_files = []
            img_h, img_w = image.shape[:2]
            

            ##TODO REPLACE THIS WITH DYNAMIC COORDINATE
            data = [
                (65, 160), (230, 340), (395, 520), (560, 665), (730, 830),
                (900, 1000), (1050, 1170), (1220, 1320), (1385, 1485),
                (1550, 1650), (1720, 1820), (1880, 1990), (2050, 2150),
                (2210, 2320), (2375, 2480), (2545, 2654), (2705, 2820),
                (2870, 2985), (3030, 3150), (3195, 3315), (3360, 3470),
                (3530, 3640), (3690, 3800), (3850, 3965), (4020, 4130),
                (4175, 4290), (4340, 4460), (4505, 4625), (4670, 4785),
                (4840, 4950), (5000, 5110), (5165, 5285), (5330, 5450),
                (5495, 5610), (5660, 5780), (5825, 5940), (5990, 6100),
                (6150, 6270), (6320, 6440), (6480, 6600)
            ]

            # Process each column
            for col_idx in range(cols):
                try:
                    # Calculate column coordinates
                    x1, x2 = data[col_idx]
                    
                    # Validate coordinates
                    if x1 < 0 or x2 > img_w or y1 < 0 or y2 > img_h:
                        print(f"Column {col_idx}: coordinates out of bounds, skipping")
                        continue
                    
                    # Crop answers columns
                    answers_columns = image[y1:y2, x1:x2]

                    ## UNCOMMENT TO DEBUG cv2.imwrite(f"{col_idx}RAW.png", answers_columns)
                    
                    if answers_columns.size == 0:
                        print(f"Column {col_idx}: empty crop, skipping")
                        continue

                    # Preprocess image
                    grayed_image = cv2.cvtColor(answers_columns, cv2.COLOR_BGR2GRAY)
                    _, processed_img = cv2.threshold(grayed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    if np.mean(processed_img) > 127:
                        processed_img = cv2.bitwise_not(processed_img)

                    # Find digits get bounding rectangle
                    digit_regions = self.find_digit_regions(processed_img, min_area=0, max_area=5000)
                    
                    if not digit_regions:
                        print(f"Column {col_idx}: no digit regions found")
                        continue
                    
                    # Sort region by y from bottom to top
                    sorted_regions = self.sort_regions_bottom_to_top(digit_regions)

                    # Process each detected region
                    for row_idx, region in enumerate(sorted_regions):
                        bbox = region['bbox']

                        # Extract dan save ROI
                        filepath = self.extract_and_save_roi(
                            processed_img, bbox, row_idx, col_idx, padding=5
                        )

                        if filepath:
                            saved_files.append(filepath)
                            
                except Exception as e:
                    print(f"Error processing column {col_idx}: {e}")
                    continue
            
            print(f"Successfully processed {len(saved_files)} ROIs")
            return len(saved_files)  # Return count of saved files
            
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            return 0
            
    # def crop_grid_cells(self, image_path: str, output_dir: str = "crops",
    #                start_x: float = 62.5, start_y: float = 4565,
    #                width: float = 132.5, height: float = 78,
    #                x_increment: float = 165.2, y_increment: float = 78,
    #                rows: int = 56, cols: int = 40) -> int:
    #     """
    #     Crop grid cells from image with top-left origin coordinate system (matching grid drawing).
        
    #     Args:
    #         image_path: Path to input image
    #         output_dir: Directory to save cropped images
    #         start_x, start_y: Starting position (top-left origin, same as grid drawing)
    #         width, height: Cell dimensions
    #         x_increment, y_increment: Cell spacing
    #         rows, cols: Grid dimensions
            
    #     Returns:
    #         Number of successfully cropped cells
    #     """
    #     try:
    #         # Load image
    #         image = cv2.imread(image_path)
    #         if image is None:
    #             logger.error(f"Cannot read image: {image_path}")
    #             return 0
                
    #         image_height, image_width = image.shape[:2]
            
    #         # Create output directory
    #         os.makedirs(output_dir, exist_ok=True)
            
    #         logger.info(f"Starting grid cropping: {rows}x{cols} cells")
    #         logger.info(f"Start position: ({start_x}, {start_y}) pixels")
    #         logger.info(f"Cell size: {width} x {height} pixels")
    #         logger.info(f"Increment: {x_increment} x {y_increment} pixels")
            
    #         successful_crops = 0
            
    #         for row in range(rows):
    #             for col in range(cols):
    #                 # Calculate coordinates using top-left origin (same as grid drawing)
    #                 img_x1 = int(start_x + (col * x_increment))
    #                 img_y1 = int(start_y + (row * y_increment))  # Row 0 = top
    #                 img_x2 = int(img_x1 + width)
    #                 img_y2 = int(img_y1 + height)
                    
    #                 # Debug for first few cells
    #                 if successful_crops < 3:
    #                     logger.info(f"Cell row{row}col{col}: ({img_x1},{img_y1}) to ({img_x2},{img_y2})")
                    
    #                 # Validate bounds
    #                 if (0 <= img_x1 < image_width and 0 <= img_y1 < image_height and
    #                     0 <= img_x2 <= image_width and 0 <= img_y2 <= image_height and
    #                     img_x1 < img_x2 and img_y1 < img_y2):
                        
    #                     # Crop cell
    #                     cell_image = image[img_y1:img_y2, img_x1:img_x2]
                        
    #                     # Save with naming convention: row{Y}col{X}.png
    #                     filename = f"row{row}col{col}.png"
    #                     output_path = os.path.join(output_dir, filename)
                        
    #                     cv2.imwrite(output_path, cell_image)
    #                     successful_crops += 1
                        
    #                 else:
    #                     if successful_crops < 10:  # Only log first few out-of-bounds
    #                         logger.warning(f"Cell row{row}col{col} out of bounds: ({img_x1},{img_y1}) to ({img_x2},{img_y2})")
                            
    #             # Progress reporting
    #             if (row + 1) % 10 == 0:
    #                 logger.info(f"Progress: {row + 1}/{rows} rows completed, {successful_crops} cells cropped")
                            
    #         logger.info(f"Grid cropping completed: {successful_crops}/{rows*cols} cells saved to '{output_dir}'")
    #         return successful_crops
            
    #     except Exception as e:
    #         logger.error(f"Grid cropping failed: {e}")
    #         return 0
            
    def process_pdf(self, pdf_path: str, output_dir: str = "crops",
                   png_temp: str = "temp_converted.png",
                   grid_config: dict = None) -> bool:
        """
        Complete pipeline: PDF conversion and grid cropping.
        
        Args:
            pdf_path: Input PDF file path
            output_dir: Directory for cropped images
            png_temp: Temporary PNG file path
            grid_config: Grid cox1, x2 in datanfiguration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if grid_config is None:
            grid_config = {
                'start_x': 65,
                'start_y': 4588,
                'width': 100,
                'height': -4574,
                'x_increment': 170,
                'cols': 40
            }
            
        try:
            logger.info("Starting PDF processing pipeline")
            
            # Step 1: Convert PDF to PNG
            dimensions = self.pdf_to_png(pdf_path, png_temp)
            if dimensions is None:
                return False
                
            # Step 2: Crop grid cells
            crop_count = self.crop_roi(png_temp, output_dir, **grid_config)
            
            # Clean up temporary file
            if os.path.exists(png_temp):
               os.remove(png_temp)
                
            success = crop_count > 0
            if success:
                logger.info(f"Pipeline completed successfully: {crop_count} cells cropped")
            else:
                logger.error("Pipeline failed: no cells were cropped")
                
            return success
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return False

class UploadAndRoIHandler:
    def __init__(self, persistent_path='./persistent'):
        self.persistent_path = persistent_path

    def handle_upload_roi(self, filename):

        input_pdf = f"{self.persistent_path}/uploaded/{filename}.pdf"
        output_directory = f"{self.persistent_path}/roi_result/{filename}/answers"
        
        if not os.path.exists(input_pdf):
            logger.error(f"Input PDF file not found: {input_pdf}")
            return
            
        # Initialize cropper
        cropper = PDFGridCropper(dpi=400)
        
        # Configure grid parameters
        grid_settings = {
            'start_x': 65,
            'start_y': 4588,
            'width': 100,
            'height': -4574,
            'x_increment': 170,
            'cols': 40
        }
        
        # Process PDF
        success = cropper.process_pdf(
            pdf_path=input_pdf,
            output_dir=output_directory,
            grid_config=grid_settings
        )

        return {
            "message":success
        }



def create_upload_roi_blueprint(cfg):
    upload_roi_handler = UploadAndRoIHandler(persistent_path=cfg.persistent_path)
    upload_roi_bp = Blueprint('upload_roi_controller', __name__)

    @upload_roi_bp.route('/upload_roi/<filename>', methods=['POST'])
    def upload_roi(filename):
        try:
            # Check if request has file data
            if not request.data and 'file' not in request.files:
                return jsonify({"error": "No file data provided"}), 400
            
            # Handle multipart form upload
            if 'file' in request.files:
                uploaded_file = request.files['file']
                if uploaded_file.filename == '':
                    return jsonify({"error": "No file selected"}), 400
                
                # Save the uploaded file
                uploaded_file.save(f"{cfg.persistent_path}/uploaded/{filename}.pdf")
            
            # Handle raw binary data upload
            elif request.data:
                # Save raw PDF data to file
                file_path = f"{cfg.persistent_path}/uploaded/{filename}.pdf"
                with open(file_path, 'wb') as f:
                    f.write(request.data)
            
            # Process the uploaded file
            result = upload_roi_handler.handle_upload_roi(filename)
            return jsonify(result), 200
            
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return upload_roi_bp