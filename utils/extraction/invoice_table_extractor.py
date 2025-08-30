"""
Invoice table extraction utilities
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from img2table.document import PDF as Img2TablePDF
import warnings
import json
import fitz

warnings.filterwarnings("ignore")

class InvoiceTableExtractor:
    """
    Extract tables from invoice PDFs with special handling for:
    - Table continuations across pages
    - Trailing rows that belong to main table
    - Header and footer detection using invoice keywords
    """
    
    def __init__(self, output_dir="temp", keywords_csv="data/master_csv/Invoice_allkeys.csv"):
        """
        Initialize the invoice table extractor.
        
        Args:
            output_dir (str): Directory for output files
            keywords_csv (str): Path to CSV containing invoice field keywords
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load invoice keywords
        self.keywords_df = pd.read_csv(keywords_csv)
        
        # Extract table boundary keywords
        self.table_start_keywords = self.keywords_df[
            self.keywords_df['field_name'] == 'table_start_position'
        ]['key'].str.lower().tolist()
        
        self.table_end_keywords = self.keywords_df[
            self.keywords_df['field_name'] == 'table_end_position'
        ]['key'].str.lower().tolist()
        
        # Get line item field names and their keywords
        self.line_fields = {}
        for _, row in self.keywords_df[self.keywords_df['field_type'] == 'Line'].iterrows():
            field_name = row['field_name']
            if field_name not in self.line_fields:
                self.line_fields[field_name] = []
            self.line_fields[field_name].append(row['key'].lower())

    def _clean_dataframe_text(self, df):
        """Clean text in DataFrame to remove artifacts and standardize format."""
        if df.empty:
            return df
            
        def clean_text(text):
            if pd.isna(text) or text is None:
                return ""
            
            text = str(text).strip()
            
            # Remove Unicode artifacts
            import re
            text = re.sub(r'[^\x20-\x7E\u00A0-\uFFFF]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        
        cleaned_df = df.copy()
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].apply(clean_text)
        
        return cleaned_df

    def _is_invoice_table(self, df):
        """
        Check if a table is likely an invoice line items table by looking for key columns.
        Returns (is_table, column_mapping, score) where:
        - is_table: bool indicating if this is likely an invoice table
        - column_mapping: maps standard names to found columns
        - score: confidence score (0-100) for how likely this is the main invoice table
        """
        if df.empty or len(df.columns) < 2:
            return False, {}, 0
            
        # Get all column names as lowercase text
        columns = [str(col).lower().strip() for col in df.columns]
        column_text = ' '.join(columns)
        
        # Check first row if columns are empty
        if not any(columns):
            first_row = [str(val).lower().strip() for val in df.iloc[0]] if len(df) > 0 else []
            column_text = ' '.join(first_row)
            if any(first_row):
                columns = first_row
        
        # Initialize column mapping
        column_map = {}
        
        # Look for key invoice columns
        required_fields = ['Description', 'Quantity', 'Amount']
        found_fields = 0
        total_fields = 0
        
        for field, keywords in self.line_fields.items():
            # Try to find matching column
            matched_col = None
            for idx, col in enumerate(columns):
                if any(kw in col for kw in keywords):
                    matched_col = df.columns[idx]
                    break
                # Check cell values if column name is unclear
                if len(df) > 0:
                    cell_values = [str(val).lower().strip() for val in df[df.columns[idx]]]
                    cell_text = ' '.join(cell_values)
                    if any(kw in cell_text for kw in keywords):
                        matched_col = df.columns[idx]
                        break
            
            if matched_col is not None:
                column_map[field] = matched_col
                total_fields += 1
                if field in required_fields:
                    found_fields += 1
        
        # Calculate confidence score
        score = 0
        if found_fields >= 2:  # Need at least 2 required fields
            # Base score from required fields
            score = (found_fields / len(required_fields)) * 60
            
            # Additional score from total fields found
            score += min(total_fields * 5, 20)  # Up to 20 points for additional fields
            
            # Bonus for having more rows (likely the main table)
            if len(df) >= 3:
                score += min(len(df), 20)  # Up to 20 points for number of rows
            
            # Penalty for very wide tables (likely not line items)
            if len(df.columns) > 10:
                score -= (len(df.columns) - 10) * 2
        
        # Table should have at least description and amount
        is_table = found_fields >= 2
        
        return is_table, column_map, min(max(score, 0), 100)

    def _standardize_table(self, df, column_map):
        """Standardize table columns using the column mapping."""
        if df.empty:
            return df
            
        # Create a new DataFrame with standardized columns
        standardized = pd.DataFrame()
        
        # Map and copy columns
        for std_name, orig_name in column_map.items():
            if orig_name in df.columns:
                standardized[std_name] = df[orig_name]
        
        # Add any unmapped columns at the end
        for col in df.columns:
            if col not in column_map.values():
                standardized[col] = df[col]
        
        return standardized

    def _merge_continuation_tables(self, tables_by_page):
        """
        Merge tables that are continuations of each other.
        Handles cases where a table spans multiple pages.
        """
        if not tables_by_page:
            return []
            
        merged_tables = []
        current_table = None
        current_mapping = None
        current_score = 0
        
        for page_num, page_tables in tables_by_page.items():
            for table in page_tables:
                df = table.df
                if df.empty:
                    continue
                
                # Clean the text
                df = self._clean_dataframe_text(df)
                
                # Check if this is an invoice table
                is_table, column_map, score = self._is_invoice_table(df)
                if not is_table:
                    continue
                
                if current_table is None:
                    current_table = df
                    current_mapping = column_map
                    current_score = score
                else:
                    # Check if this table has similar columns
                    _, new_mapping, new_score = self._is_invoice_table(df)
                    if set(new_mapping.keys()) & set(current_mapping.keys()):
                        # Similar columns - likely a continuation
                        current_table = pd.concat([current_table, df], ignore_index=True)
                        current_score = max(current_score, new_score)  # Keep highest score
                    else:
                        # Different columns - new table
                        if not current_table.empty:
                            merged_tables.append((current_table, current_mapping, current_score))
                        current_table = df
                        current_mapping = new_mapping
                        current_score = new_score
        
        # Add the last table
        if current_table is not None and not current_table.empty:
            merged_tables.append((current_table, current_mapping, current_score))
        
        return merged_tables

    def _post_process_table(self, df, column_map):
        """
        Post-process a table to:
        1. Merge continuation rows
        2. Remove summary rows
        3. Clean up formatting
        """
        if df.empty:
            return df
            
        # Clean text first
        df = self._clean_dataframe_text(df)
        
        # Standardize columns
        df = self._standardize_table(df, column_map)
        
        # Remove empty rows
        df = df.dropna(how='all').reset_index(drop=True)
        
        # Remove rows that look like headers
        df = df[~df.apply(lambda row: any(kw in ' '.join(str(val).lower() for val in row) for kw in self.table_start_keywords), axis=1)]
        
        # Remove summary rows (usually at the bottom)
        df = df[~df.apply(lambda row: any(kw in ' '.join(str(val).lower() for val in row) for kw in self.table_end_keywords), axis=1)]
        
        # Merge continuation rows
        rows_to_merge = []
        current_row_idx = None
        
        for idx, row in df.iterrows():
            row_values = row.dropna()
            if len(row_values) <= 2:  # Row has 2 or fewer non-empty cells
                if current_row_idx is not None:
                    rows_to_merge.append((current_row_idx, idx))
            else:
                current_row_idx = idx
        
        # Merge identified rows
        for target_idx, source_idx in rows_to_merge:
            source_row = df.iloc[source_idx]
            for col in df.columns:
                if pd.notna(source_row[col]) and str(source_row[col]).strip():
                    current_val = str(df.at[target_idx, col]) if pd.notna(df.at[target_idx, col]) else ""
                    new_val = str(source_row[col]).strip()
                    if current_val:
                        df.at[target_idx, col] = f"{current_val} {new_val}"
                    else:
                        df.at[target_idx, col] = new_val
        
        # Remove merged rows
        df = df.drop([idx for _, idx in rows_to_merge])
        
        # Reset index one final time
        df.reset_index(drop=True, inplace=True)
        
        return df

    def extract_tables(self, pdf_path, save_to_excel=True):
        """
        Extract tables from an invoice PDF.
        
        Args:
            pdf_path (str): Path to the PDF file
            save_to_excel (bool): Whether to save extracted tables to Excel
            
        Returns:
            list: List of extracted tables as DataFrames
        """
        try:
            print(f"📄 Processing invoice: {pdf_path}")
            
            # Try different settings for table extraction
            extraction_settings = [
                # First try: Standard settings
                {
                    'borderless_tables': True,
                    'implicit_rows': True,
                    'implicit_columns': True,
                    'min_confidence': 50
                },
                # Second try: More lenient settings
                {
                    'borderless_tables': True,
                    'implicit_rows': True,
                    'implicit_columns': True,
                    'min_confidence': 30
                },
                # Third try: Most lenient settings
                {
                    'borderless_tables': True,
                    'implicit_rows': True,
                    'implicit_columns': True,
                    'min_confidence': 20
                }
            ]
            
            tables = None
            pdf = Img2TablePDF(pdf_path)
            
            for settings in extraction_settings:
                print(f"Trying table extraction with settings: {settings}")
                try:
                    tables = pdf.extract_tables(**settings)
                    if tables:
                        print(f"✅ Found tables with settings: {settings}")
                        break
                except Exception as e:
                    print(f"⚠️ Extraction failed with settings: {e}")
                    continue
            
            if not tables:
                print("❌ No tables found in PDF")
                return []
            
            # Merge continuation tables
            merged_tables = self._merge_continuation_tables(tables)
            
            # Sort tables by confidence score
            merged_tables.sort(key=lambda x: x[2], reverse=True)
            
            # Post-process each table
            processed_tables = []
            for table, column_map, score in merged_tables:
                processed_df = self._post_process_table(table, column_map)
                if not processed_df.empty:
                    processed_tables.append(processed_df)
                    print(f"✅ Processed table: {processed_df.shape[0]} rows × {processed_df.shape[1]} columns (score: {score:.1f})")
                    print("Columns:", list(processed_df.columns))
            
            # Save to Excel if requested
            if save_to_excel and processed_tables:
                base_name = Path(pdf_path).stem
                excel_path = os.path.join(self.output_dir, f"{base_name}_tables.xlsx")
                
                with pd.ExcelWriter(excel_path) as writer:
                    for idx, df in enumerate(processed_tables):
                        sheet_name = f"Table_{idx + 1}"
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                print(f"💾 Saved tables to: {excel_path}")
            
            # Return only the highest scoring table if any found
            return processed_tables[:1] if processed_tables else []
            
        except Exception as e:
            print(f"❌ Error extracting tables: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

def main():
    """Example usage of InvoiceTableExtractor"""
    # Initialize extractor
    extractor = InvoiceTableExtractor(
        output_dir="invoice_outputs",
        keywords_csv="data/master_csv/Invoice_allkeys.csv"
    )
    
    # Example invoice
    pdf_path = "data/ocr_inputs/35 Invoices/18.1_Dunzo Daily.pdf"
    
    # Extract tables
    tables = extractor.extract_tables(pdf_path)
    
    # Print results
    print(f"\nExtracted {len(tables)} tables")
    for idx, df in enumerate(tables):
        print(f"\nTable {idx + 1}:")
        print(df.head())
        print("\nColumns:", list(df.columns))

if __name__ == "__main__":
    main() 