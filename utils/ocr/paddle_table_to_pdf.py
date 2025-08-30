"""
PDF reconstruction from PaddleOCR TableRecognitionPipelineV2 JSON output
"""

import os
import json
import fitz
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

class PaddleTableToPDF:
    """Convert PaddleOCR table recognition output to PDF"""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the converter
        
        Args:
            output_dir (str): Directory to save output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # PDF settings
        self.page_width, self.page_height = A4
        self.margin = 50  # points
        self.font_size = 10
        self.line_width = 0.5
        
        # Try to register Arial font for better Unicode support
        try:
            pdfmetrics.registerFont(TTFont('Arial', 'Arial.ttf'))
            self.font_name = 'Arial'
        except:
            self.font_name = 'Helvetica'  # Fallback font

    def _clean_text(self, text: str) -> str:
        """Clean text for PDF rendering"""
        if not text:
            return ""
        # Remove control characters but keep newlines
        return ''.join(char for char in str(text) if ord(char) >= 32 or char in '\n\r')

    def _scale_coordinates(self, coords: List[List[float]], original_size: Tuple[int, int]) -> List[List[float]]:
        """Scale coordinates from original image size to PDF size"""
        orig_width, orig_height = original_size
        scale_x = (self.page_width - 2 * self.margin) / orig_width
        scale_y = (self.page_height - 2 * self.margin) / orig_height
        
        scaled = []
        for x, y in coords:
            scaled_x = x * scale_x + self.margin
            # Flip Y coordinate for PDF (origin is bottom-left)
            scaled_y = self.page_height - (y * scale_y + self.margin)
            scaled.append([scaled_x, scaled_y])
            
        return scaled

    def _draw_table_structure(self, c: canvas.Canvas, table_data: Dict[str, Any], original_size: Tuple[int, int]):
        """Draw table lines and borders"""
        # Set line properties
        c.setStrokeColor(colors.grey)
        c.setLineWidth(self.line_width)
        
        # Draw cells
        cells = table_data.get('cells', [])
        for cell in cells:
            bbox = cell.get('bbox', [])
            if len(bbox) == 4:  # [x1, y1, x2, y2]
                # Convert bbox to polygon coordinates
                poly = [
                    [bbox[0], bbox[1]],  # top-left
                    [bbox[2], bbox[1]],  # top-right
                    [bbox[2], bbox[3]],  # bottom-right
                    [bbox[0], bbox[3]]   # bottom-left
                ]
                # Scale coordinates
                scaled_poly = self._scale_coordinates(poly, original_size)
                
                # Draw cell rectangle
                c.rect(
                    scaled_poly[0][0],  # x
                    scaled_poly[0][1],  # y
                    scaled_poly[1][0] - scaled_poly[0][0],  # width
                    scaled_poly[2][1] - scaled_poly[1][1],  # height
                    stroke=1,
                    fill=0
                )

    def _add_table_text(self, c: canvas.Canvas, table_data: Dict[str, Any], original_size: Tuple[int, int]):
        """Add text content to the table"""
        c.setFont(self.font_name, self.font_size)
        c.setFillColor(colors.black)
        
        cells = table_data.get('cells', [])
        for cell in cells:
            text = cell.get('text', '')
            bbox = cell.get('bbox', [])
            
            if text and len(bbox) == 4:
                # Scale coordinates
                scaled_coords = self._scale_coordinates([[bbox[0], bbox[1]]], original_size)[0]
                
                # Add text with small offset from cell border
                text_x = scaled_coords[0] + 2
                text_y = scaled_coords[1] - self.font_size - 2
                
                # Clean and draw text
                clean_text = self._clean_text(text)
                c.drawString(text_x, text_y, clean_text)

    def reconstruct_pdf(self, json_path: str, image_size: Optional[Tuple[int, int]] = None) -> str:
        """
        Reconstruct PDF from PaddleOCR table recognition JSON output
        
        Args:
            json_path (str): Path to JSON file from TableRecognitionPipelineV2
            image_size (tuple): Original image size (width, height). If None, will try to get from JSON
            
        Returns:
            str: Path to reconstructed PDF
        """
        try:
            # Load JSON data
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if not data:
                raise ValueError("Empty JSON data")
            
            # Get image size
            if not image_size:
                # Try to get from JSON if available
                if isinstance(data, dict) and 'image_size' in data:
                    image_size = tuple(data['image_size'])
                else:
                    # Default to A4 proportions
                    image_size = (595, 842)
            
            # Create output PDF path
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            pdf_path = os.path.join(self.output_dir, f"{base_name}_reconstructed.pdf")
            
            # Create PDF
            c = canvas.Canvas(pdf_path, pagesize=A4)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Multiple tables
                for table_data in data:
                    self._draw_table_structure(c, table_data, image_size)
                    self._add_table_text(c, table_data, image_size)
                    c.showPage()  # New page for each table
            elif isinstance(data, dict):
                # Single table
                self._draw_table_structure(c, data, image_size)
                self._add_table_text(c, data, image_size)
            
            c.save()
            print(f"✅ Reconstructed PDF saved to: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            print(f"❌ Error reconstructing PDF: {str(e)}")
            raise

    def save_tables_to_excel(self, json_path: str) -> Optional[str]:
        """
        Save tables from JSON to Excel
        
        Args:
            json_path (str): Path to JSON file from TableRecognitionPipelineV2
            
        Returns:
            str: Path to Excel file if successful, None otherwise
        """
        try:
            # Load JSON data
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if not data:
                return None
            
            # Create Excel writer
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            excel_path = os.path.join(self.output_dir, f"{base_name}_tables.xlsx")
            
            with pd.ExcelWriter(excel_path) as writer:
                if isinstance(data, list):
                    # Multiple tables
                    for i, table_data in enumerate(data):
                        df = self._table_to_dataframe(table_data)
                        if not df.empty:
                            df.to_excel(writer, sheet_name=f'Table_{i+1}', index=False)
                elif isinstance(data, dict):
                    # Single table
                    df = self._table_to_dataframe(data)
                    if not df.empty:
                        df.to_excel(writer, sheet_name='Table_1', index=False)
            
            print(f"✅ Tables saved to Excel: {excel_path}")
            return excel_path
            
        except Exception as e:
            print(f"❌ Error saving tables to Excel: {str(e)}")
            return None

    def _table_to_dataframe(self, table_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert table data to pandas DataFrame"""
        try:
            cells = table_data.get('cells', [])
            if not cells:
                return pd.DataFrame()
            
            # Get table dimensions
            max_row = max(cell.get('row_idx', 0) for cell in cells)
            max_col = max(cell.get('col_idx', 0) for cell in cells)
            
            # Create empty DataFrame
            df = pd.DataFrame(np.empty((max_row + 1, max_col + 1), dtype=object))
            
            # Fill in cell values
            for cell in cells:
                row = cell.get('row_idx', 0)
                col = cell.get('col_idx', 0)
                text = cell.get('text', '')
                df.iloc[row, col] = text
            
            # Clean up
            df = df.fillna('')
            
            # If first row looks like headers, use it
            first_row = df.iloc[0]
            if not first_row.str.contains(r'\d').any():  # If first row has no numbers
                df.columns = first_row
                df = df.iloc[1:]
            
            return df
            
        except Exception as e:
            print(f"❌ Error converting table to DataFrame: {str(e)}")
            return pd.DataFrame()

def main():
    """Example usage"""
    converter = PaddleTableToPDF(output_dir="output")
    
    # Example with known image size
    json_path = "path/to/table_recognition_output.json"
    image_size = (800, 1000)  # Example size
    
    # Reconstruct PDF
    pdf_path = converter.reconstruct_pdf(json_path, image_size)
    
    # Save to Excel
    excel_path = converter.save_tables_to_excel(json_path)

if __name__ == "__main__":
    main() 