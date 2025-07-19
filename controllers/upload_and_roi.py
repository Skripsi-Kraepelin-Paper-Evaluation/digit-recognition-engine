import fitz  # PyMuPDF
import cv2
import numpy as np
import os
from PIL import Image
import logging
from typing import Tuple, Optional
from flask import Blueprint, jsonify, request
import json

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
            
    def crop_grid_cells(self, image_path: str, output_dir: str = "crops",
                   start_x: float = 62.5, start_y: float = 4565,
                   width: float = 132.5, height: float = 78,
                   x_increment: float = 165.2, y_increment: float = 78,
                   rows: int = 56, cols: int = 40) -> int:
        """
        Crop grid cells from image with top-left origin coordinate system (matching grid drawing).
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save cropped images
            start_x, start_y: Starting position (top-left origin, same as grid drawing)
            width, height: Cell dimensions
            x_increment, y_increment: Cell spacing
            rows, cols: Grid dimensions
            
        Returns:
            Number of successfully cropped cells
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Cannot read image: {image_path}")
                return 0
                
            image_height, image_width = image.shape[:2]
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Starting grid cropping: {rows}x{cols} cells")
            logger.info(f"Start position: ({start_x}, {start_y}) pixels")
            logger.info(f"Cell size: {width} x {height} pixels")
            logger.info(f"Increment: {x_increment} x {y_increment} pixels")
            
            successful_crops = 0
            
            for row in range(rows):
                for col in range(cols):
                    # Calculate coordinates using top-left origin (same as grid drawing)
                    img_x1 = int(start_x + (col * x_increment))
                    img_y1 = int(start_y + (row * y_increment))  # Row 0 = top
                    img_x2 = int(img_x1 + width)
                    img_y2 = int(img_y1 + height)
                    
                    # Debug for first few cells
                    if successful_crops < 3:
                        logger.info(f"Cell row{row}col{col}: ({img_x1},{img_y1}) to ({img_x2},{img_y2})")
                    
                    # Validate bounds
                    if (0 <= img_x1 < image_width and 0 <= img_y1 < image_height and
                        0 <= img_x2 <= image_width and 0 <= img_y2 <= image_height and
                        img_x1 < img_x2 and img_y1 < img_y2):
                        
                        # Crop cell
                        cell_image = image[img_y1:img_y2, img_x1:img_x2]
                        
                        # Save with naming convention: row{Y}col{X}.png
                        filename = f"row{row}col{col}.png"
                        output_path = os.path.join(output_dir, filename)
                        
                        cv2.imwrite(output_path, cell_image)
                        successful_crops += 1
                        
                    else:
                        if successful_crops < 10:  # Only log first few out-of-bounds
                            logger.warning(f"Cell row{row}col{col} out of bounds: ({img_x1},{img_y1}) to ({img_x2},{img_y2})")
                            
                # Progress reporting
                if (row + 1) % 10 == 0:
                    logger.info(f"Progress: {row + 1}/{rows} rows completed, {successful_crops} cells cropped")
                            
            logger.info(f"Grid cropping completed: {successful_crops}/{rows*cols} cells saved to '{output_dir}'")
            return successful_crops
            
        except Exception as e:
            logger.error(f"Grid cropping failed: {e}")
            return 0
            
    def process_pdf(self, pdf_path: str, output_dir: str = "crops",
                   png_temp: str = "temp_converted.png",
                   grid_config: dict = None) -> bool:
        """
        Complete pipeline: PDF conversion and grid cropping.
        
        Args:
            pdf_path: Input PDF file path
            output_dir: Directory for cropped images
            png_temp: Temporary PNG file path
            grid_config: Grid configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if grid_config is None:
            grid_config = {
                'start_x': 62.5,
                'start_y': 4487,
                'width': 100,
                'height': 78,
                'x_increment': 165,
                'y_increment': -78,
                'rows': 58,
                'cols': 45
            }
            
        try:
            logger.info("Starting PDF processing pipeline")
            
            # Step 1: Convert PDF to PNG
            dimensions = self.pdf_to_png(pdf_path, png_temp)
            if dimensions is None:
                return False
                
            # Step 2: Crop grid cells
            crop_count = self.crop_grid_cells(png_temp, output_dir, **grid_config)
            
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
            'start_x': 62.5,
            'start_y': 4487,
            'width': 100,
            'height': 78,
            'x_increment': 165,
            'y_increment': -78,
            'rows': 56,
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