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

        avg_center_x = sum([r['bbox'][0] + r['bbox'][2] // 2 for r in digit_regions]) / len(digit_regions)
        return avg_center_x < center_x_threshold

    def find_digit_regions(self, processed_img, min_area=50, max_area=5000, min_center_x=8):
        """Deteksi region yang mengandung digit menggunakan contour detection"""
        # Apply closing untuk menggabungkan bagian karakter yang terpisah
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digit_regions = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            center_x = x + w // 2

            # Skip jika titik tengah terlalu ke kiri
            if x == 0 and center_x < min_center_x:
                continue

            if (min_area <= area <= max_area and
                0.1 <= aspect_ratio <= 2.0 and
                h >= 10 and w >= 5):
                digit_regions.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'center_y': y + h // 2
                })

        return digit_regions

    def detect_columns_with_projection(self,image, min_height_ratio=0.1):
        """
        Deteksi kolom menggunakan proyeksi horizontal
        Lebih cepat dan robust untuk layout kolom yang teratur
        """
        
        # Load dan preprocess image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Hitung proyeksi vertikal (sum setiap kolom)
        vertical_projection = np.sum(binary, axis=0)
        
        # Tentukan threshold berdasarkan tinggi gambar
        height_threshold = binary.shape[0] * min_height_ratio * 255
        
        # Cari area yang memiliki konten (nilai proyeksi > threshold)
        content_areas = vertical_projection > height_threshold
        
        # Cari transisi dari False ke True dan True ke False
        transitions = np.diff(content_areas.astype(int))
        starts = np.where(transitions == 1)[0] + 1  # Start of content area
        ends = np.where(transitions == -1)[0] + 1   # End of content area
        
        # Handle edge cases
        if content_areas[0]:
            starts = np.concatenate([[0], starts])
        if content_areas[-1]:
            ends = np.concatenate([ends, [len(content_areas)]])
        
        # Buat list koordinat kolom
        columns = []
        for start, end in zip(starts, ends):
            columns.append((int(start), int(end)))
        
        return columns


    def shift_pairs(self,pairs):
        """
        Menggeser pasangan: [(x1,x2),(x3,x4),(x5,x6),...] 
        menjadi:            [(x2,x3),(x4,x5),(x6,x7),...]

        Parameters:
            pairs (list of tuple): List berisi pasangan angka.

        Returns:
            list of tuple: List pasangan hasil pergeseran.
        """
        # Ambil semua elemen dalam pairs menjadi satu list
        flat = [item for pair in pairs for item in pair]
        
        # Buat pasangan sliding dari elemen 1 ke n-2
        result = [(flat[i], flat[i+1]) for i in range(1, len(flat)-2+1, 2)]
        return result

    def crop_roi(self, image_path: str, output_dir: str = "crops",
                start_y: float = 4588,
                width: float = 100, height: float = -4574,
                cols: int = 40, filename:str = "filename") -> int:
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
            

            top_image = image[:100, :]
            # cv2.imwrite(f"TOPIMAGE_{filename}.png", top_image) ## UNCOMMENT TO DEBUG
            data = self.detect_columns_with_projection(top_image)

            dataShiftPairs = self.shift_pairs(data)

            ## append dataShiftPairs terakhir adalah (x2 terakhir dari data, x max image) (khusus answer column terakhir bukan hasil deteksi)
            if data:
                last_x = data[-1][1]  # Ambil x2 dari pasangan terakhir
                if last_x < img_w:
                    dataShiftPairs.append((last_x, img_w))

            #### shift coordinate kiri 15 pixel untuk setiap x2 (reduce questions digit noise)
            N = 15
            dataShiftPairs = [(xa, max(xb - N, xa)) for (xa, xb) in dataShiftPairs]
            #### shift coordinate kanan 2 pixel untuk setiap x1 (reduce questions digit noise)
            # N2 = 1
            # dataShiftPairs = [(xa+N, max(xb,xa+N)) for (xa, xb) in dataShiftPairs]

            ## preprocess remove right coordinate for tolerance


            # Process each column
            for col_idx in range(cols):
                try:
                    # Calculate column coordinates
                    x1, x2 = dataShiftPairs[col_idx]
                    
                    # Validate coordinates
                    if x1 < 0 or x2 > img_w or y1 < 0 or y2 > img_h:
                        print(f"Column {col_idx}: coordinates out of bounds, skipping")
                        continue
                    
                    # Crop answers columns
                    x1 = min(x1, img_w)
                    x2 = min(x2, img_w)
                    y1 = min(y1, img_h)
                    y2 = min(y2, img_h)

                    # Validasi: hanya crop kalau ukuran valid
                    if x2 > x1 and y2 > y1:
                        answers_columns = image[y1:y2, x1:x2]
                        # cv2.imwrite(f"{filename}_{col_idx}RAW.png", answers_columns) ##UNCOMMENT TO DEBUG
                    else:
                        print(f"[SKIP] Area invalid atau terlalu kecil di col {col_idx}")
                    
                    if answers_columns.size == 0:
                        print(f"Column {col_idx}: empty crop, skipping")
                        continue

                    # Preprocess image
                    grayed_image = cv2.cvtColor(answers_columns, cv2.COLOR_BGR2GRAY)
                    _, processed_img = cv2.threshold(grayed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    if np.mean(processed_img) > 127:
                        processed_img = cv2.bitwise_not(processed_img)

                    # Find digits get bounding rectangle
                    digit_regions = self.find_digit_regions(processed_img, min_area=300, max_area=5000)
                    
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
            
    def process_pdf(self, pdf_path: str, output_dir: str = "crops",
                   png_temp: str = "temp_converted.png",
                   grid_config: dict = None, filename: str = "filename") -> bool:
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
                'start_y': 4588,
                'width': 100,
                'height': -4574,
                'cols': 40,
                "filename":filename,
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

    def handle_upload_roi(self, filename, additional_data):

        input_pdf = f"{self.persistent_path}/uploaded/{filename}.pdf"
        output_directory = f"{self.persistent_path}/roi_result/{filename}/answers"
        
        if not os.path.exists(input_pdf):
            logger.error(f"Input PDF file not found: {input_pdf}")
            return
            
        # Initialize cropper
        cropper = PDFGridCropper(dpi=400)
        
        # Configure grid parameters
        grid_settings = {
            'start_y': 4588,
            'width': 100,
            'height': -4574,
            'cols': 40,
            'filename':filename,
        }
        
        # Process PDF
        success = cropper.process_pdf(
            pdf_path=input_pdf,
            output_dir=output_directory,
            grid_config=grid_settings
        )

        # Save additional data as JSON metadata
        if additional_data:
            metadata_dir = f"{self.persistent_path}/metadata"
            os.makedirs(metadata_dir, exist_ok=True)
            
            metadata_path = f"{metadata_dir}/{filename}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(additional_data, f, ensure_ascii=False, indent=2)

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
            if 'file' not in request.files:
                return jsonify({"error": "No file provided"}), 400
            
            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            # Extract additional form data
            occupancy_and_role = request.form.get('occupacyAndRole', '')
            last_edu = request.form.get('lastEdu', '')
            pob = request.form.get('pob', '')  # Place of Birth
            dob = request.form.get('dob', '')  # Date of Birth
            name = request.form.get('name', '')  # Name
            testDate = request.form.get('testDate', '')
            
            # Validate lastEdu enum
            valid_edu_levels = ['SMA', 'D1', 'D2', 'D3', 'D4', 'S1', 'S2', 'S3', 'Lainnya']
            if last_edu and last_edu not in valid_edu_levels:
                return jsonify({"error": f"Invalid lastEdu value. Must be one of: {', '.join(valid_edu_levels)}"}), 400
            
            # Validate date format (assuming YYYY-MM-DD format)
            if dob:
                try:
                    from datetime import datetime
                    datetime.strptime(dob, '%Y-%m-%d')
                except ValueError:
                    return jsonify({"error": "Invalid date format for dob. Use YYYY-MM-DD format"}), 400
            
            # Save the uploaded file
            file_path = f"{cfg.persistent_path}/uploaded/{filename}.pdf"
            uploaded_file.save(file_path)
            
            # Prepare additional data to pass to handler
            additional_data = {
                'occupacyAndRole': occupancy_and_role,
                'lastEdu': last_edu,
                'pob': pob,
                'dob': dob,
                'name': name,
                'testDate': testDate
            }
            
            # Process the uploaded file with additional data
            result = upload_roi_handler.handle_upload_roi(filename, additional_data=additional_data)
            return jsonify(result), 200
            
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return upload_roi_bp