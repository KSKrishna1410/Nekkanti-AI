#!/usr/bin/env python3
"""
Bank statement table extraction module
"""

import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

from utils.extraction.table_extractor import DocumentTableExtractor


class BankStatementExtractor:
    """
    Specialized extractor for bank statement transaction tables.
    
    Features:
    - Automatically identifies the main transaction table
    - Finds and concatenates continuation tables with same column structure
    - Handles headers and data consistency
    - Returns consolidated bank statement table
    """
    
    def __init__(self, output_dir="bank_statement_output", save_reconstructed_pdfs=True):
        """
        Initialize the BankStatementExtractor.
        
        Args:
            output_dir (str): Directory to save output files
            save_reconstructed_pdfs (bool): Whether to save reconstructed PDFs
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize the underlying table extractor
        self.table_extractor = DocumentTableExtractor(
            output_dir=os.path.join(self.output_dir, "extracted_tables"),
            save_reconstructed_pdfs=save_reconstructed_pdfs
        )

    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted characters and normalizing whitespace.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or str(text).lower() in ['nan', 'none', 'null', '<na>'] or not str(text).strip():
            return ''
            
        # Convert to string and normalize whitespace
        text = str(text).strip()
        
        # Remove Unicode control characters and zero-width spaces
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        text = text.replace('\u200b', '')  # zero-width space
        text = text.replace('\ufeff', '')  # zero-width no-break space
        
        # Remove specific artifacts often found in bank statements
        text = text.replace('_x000D_', '')
        text = text.replace('\\n', ' ')
        text = text.replace('\\r', ' ')
        text = text.replace('\\t', ' ')
        
        # Remove _xFFFE_ and similar artifacts
        text = text.replace('_xFFFE_', ' ')
        text = text.replace('_x0020_', ' ')
        text = text.replace('_x000A_', ' ')
        text = text.replace('_x000B_', ' ')
        text = text.replace('_x000C_', ' ')
        text = text.replace('_x000D_', ' ')
        text = text.replace('_x000E_', ' ')
        text = text.replace('_x000F_', ' ')
        
        # Replace multiple spaces with single space
        text = ' '.join(text.split())
        
        # Clean up any remaining whitespace
        text = text.strip()
        
        # Handle NaN-like values
        if text.lower() in ['nan', 'none', 'null', '<na>']:
            return ''
        
        return text

    def _clean_dataframe_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text in DataFrame cells."""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Convert all values to string and clean
        for col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(lambda x: self._clean_text(str(x)))
            
        # Remove completely empty rows
        df = df.replace('', pd.NA)
        df = df.dropna(how='all')
        
        # Reset index after cleaning
        return df.reset_index(drop=True)

    def _should_protect_row(self, row_data: pd.Series) -> bool:
        """Check if a row should be protected from merging."""
        # Convert row data to string and lowercase for checking
        row_text = ' '.join(str(val).lower() for val in row_data)
        
        # Keywords that indicate a row should not be merged
        protected_keywords = [
            'opening', 'closing', 'balance',
            'brought forward', 'carried forward',
            'b/f', 'c/f', 'total', 'subtotal'
        ]
        
        # Check if any protected keyword is in the row
        return any(keyword in row_text for keyword in protected_keywords)

    def _merge_with_previous_if_needed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge rows only if they don't contain protected keywords."""
        if df.empty or len(df) < 2:
            return df
            
        result_df = df.copy()
        rows_to_drop = []
        
        for i in range(1, len(df)):
            current_row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Skip if current row should be protected
            if self._should_protect_row(current_row):
                print(f"   🔒 Protected row {i} from merging: {current_row.to_list()}")
                continue
        
            # Skip if previous row should be protected
            if self._should_protect_row(prev_row):
                continue
            
            # Check if current row might be a continuation
            current_empty = pd.isna(current_row) | (current_row == '')
            prev_empty = pd.isna(prev_row) | (prev_row == '')
            
            if not current_empty.all() and not prev_empty.all():
                # Only merge if there's a clear pattern of complementary data
                for col in df.columns:
                    if pd.isna(prev_row[col]) or prev_row[col] == '':
                        if not (pd.isna(current_row[col]) or current_row[col] == ''):
                            result_df.at[i-1, col] = current_row[col]
                rows_to_drop.append(i)
        
        # Remove merged rows
        result_df = result_df.drop(rows_to_drop)
        
        return result_df.reset_index(drop=True)
    
    def _is_transaction_table(self, df: pd.DataFrame, table_info: Dict) -> Tuple[bool, int]:
        """
        Determine if a table is likely a bank statement transaction table.
        
        Args:
            df (pd.DataFrame): The table dataframe
            table_info (dict): Table metadata
            
        Returns:
            tuple: (is_transaction_table, confidence_score)
        """
        if df.empty or df.shape[0] < 3 or df.shape[1] < 3:
            return False, 0
        
        confidence = 0
        
        # Check for typical bank statement columns (case-insensitive)
        transaction_keywords = [
            'date', 'transaction', 'description', 'amount', 'balance', 'debit', 'credit',
            'reference', 'particulars', 'details', 'value', 'withdrawal', 'deposit',
            'txn', 'ref', 'chq', 'cheque', 'transfer', 'payment', 'receipt', 'transdate',
            'valuedate', 'branch', 'withdraws', 'refichq', 'tran date', 'value date',
            'narration', 'remarks', 'mode', 'instrument', 'opening', 'closing'
        ]
        
        # Convert all text to lowercase for checking, handling complex nested structures
        def flatten_cell_text(cell):
            """Convert complex cell structures to clean text"""
            if pd.isna(cell) or cell is None:
                return ''
            
            if isinstance(cell, (list, tuple, np.ndarray)):
                try:
                    if len(cell) > 0:
                        return ' '.join(str(item) for item in cell if item is not None).lower()
                    else:
                        return ''
                except:
                    return str(cell).lower()
            
            return str(cell).lower()
        
        # Apply flattening to all cells
        df_flattened = df.applymap(flatten_cell_text)
        
        # Check first row for column headers
        first_row = ' '.join(df_flattened.iloc[0].values) if not df_flattened.empty else ''
        header_matches = [keyword for keyword in transaction_keywords if keyword in first_row]
        if len(header_matches) >= 2:
            confidence += 30
            print(f"   ✅ Found header keywords: {header_matches}")
        
        # Check all text for transaction-related content
        all_text = ' '.join(df_flattened.values.flatten())
        found_keywords = [keyword for keyword in transaction_keywords if keyword in all_text]
        keyword_matches = len(found_keywords)
        keyword_score = keyword_matches * 10
        confidence += keyword_score
        print(f"   ✅ Found {keyword_matches} transaction keywords: {found_keywords}")
        
        # Check if this looks like a header/info table
        header_keywords = ['branch', 'ifsc', 'address', 'nominee', 'currency', 'period', 'customer', 'account no']
        header_matches = [keyword for keyword in header_keywords if keyword in first_row]
        if len(header_matches) >= 2:
            print(f"   ❌ Found header table keywords: {header_matches}")
            return False, 0
        
        # Prefer tables with more rows (transaction tables are usually longer)
        row_score = 0
        if df.shape[0] > 10:
            row_score = 20
            confidence += row_score
            print(f"   ✅ Good row count: {df.shape[0]} rows")
        elif df.shape[0] > 5:
            row_score = 10
            confidence += row_score
            print(f"   ✓ Acceptable row count: {df.shape[0]} rows")
        
        # Prefer tables with 4-8 columns (typical for transaction tables)
        col_score = 0
        if 4 <= df.shape[1] <= 9:
            col_score = 15
            confidence += col_score
            print(f"   ✅ Good column count: {df.shape[1]} columns")
        
        # Check for date-like patterns in first few columns using flattened data
        date_score = 0
        for col_idx in range(min(3, df.shape[1])):
            col_data = df_flattened.iloc[:, col_idx]
            date_like_values = [val for val in col_data 
                              if any(pattern in val for pattern in ['/', '-', '2024', '2023', '2022', '2021'])]
            date_like_count = len(date_like_values)
            
            if date_like_count > df.shape[0] * 0.3:  # 30% of values look date-like
                date_score = 25
                confidence += date_score
                print(f"   ✅ Found date patterns in column {col_idx}")
                break
        
        # Check for amount-like patterns (numbers with decimals or commas) using flattened data
        amount_score = 0
        amount_patterns = 0
        for col_idx in range(df.shape[1]):
            col_data = df_flattened.iloc[:, col_idx]
            for val in col_data:
                if any(char in val for char in ['₹', '$', '€', '£', '.', ',']):
                    if any(char.isdigit() for char in val):
                        amount_patterns += 1
                        break
        
        if amount_patterns >= 2:  # At least two columns with amounts (typical for debit/credit or amount/balance)
            amount_score = 30
            confidence += amount_score
            print(f"   ✅ Found amount patterns in {amount_patterns} columns")
        
        # Penalize very small tables
        penalty = 0
        if df.shape[0] < 5:
            penalty = 20
            confidence -= penalty
            print(f"   ❌ Table too small: {df.shape[0]} rows")
        
        # Check for header/info table patterns using flattened data
        if df.shape[0] <= 5 and df.shape[1] <= 3:
            info_patterns = ['period', 'branch', 'currency', 'nominee', 'address', 'customer']
            info_matches = sum(1 for pattern in info_patterns if any(pattern in val for val in df_flattened.values.flatten()))
            if info_matches >= 2:
                print(f"   ❌ Found info table patterns: {info_matches} matches")
                return False, 0
        
        # Lower threshold for edge cases where pattern matching is difficult
        final_decision = confidence > 30
        print(f"   📊 Final confidence score: {confidence} (threshold: 30)")
        
        # Special case: If table has reasonable size and structure, give it benefit of doubt
        if not final_decision and df.shape[0] >= 5 and df.shape[1] >= 4:
            # Check if it has any numerical data that could be amounts
            has_numbers = False
            for col_idx in range(df.shape[1]):
                col_data = df_flattened.iloc[:, col_idx]
                if any(any(c.isdigit() for c in str(val)) for val in col_data):
                    has_numbers = True
                    break
            
            if has_numbers:
                print(f"   💡 Giving benefit of doubt - has numbers and good structure")
                return True, 35
        
        # Additional check: Look for obvious transaction data patterns
        if not final_decision:
            transaction_patterns = ['neft', 'rtgs', 'upi', 'imps', 'transfer', 'atm', 'pos', 'ach', 'ecs']
            pattern_count = 0
            for pattern in transaction_patterns:
                if any(pattern in val for val in df_flattened.values.flatten()):
                    pattern_count += 1
            
            if pattern_count >= 2:
                print(f"   💡 Found transaction patterns: {pattern_count} matches")
                return True, 40
        
        return final_decision, confidence
    
    def _remove_extraneous_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove extraneous columns that might be OCR artifacts or padding.
        Identifies and removes columns that are mostly empty or contain non-meaningful data.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with extraneous columns removed
        """
        if df.empty or df.shape[1] < 2:
            return df
            
        print(f"🔍 Analyzing columns for extraneous data (shape: {df.shape})")
        
        # Analyze each column
        columns_to_keep = []
        for col_idx in range(df.shape[1]):
            col_data = df.iloc[:, col_idx]
            col_name = df.columns[col_idx] if hasattr(df.columns, '__getitem__') else f'Column_{col_idx}'
            
            # Count non-empty, meaningful values
            non_empty_count = 0
            meaningful_count = 0
            
            for val in col_data:
                if pd.notna(val) and str(val).strip() not in ['', 'nan', 'None', '<NA>']:
                    non_empty_count += 1
                    # Check if value is meaningful (has letters/numbers, not just artifacts)
                    val_str = str(val).strip()
                    if (len(val_str) > 1 and 
                        (any(c.isalnum() for c in val_str) or 
                         any(char in val_str for char in ['/', '-', ':', '.', ',', '₹', '$']))):
                        meaningful_count += 1
            
            # Calculate meaningful ratio
            total_rows = len(col_data)
            meaningful_ratio = meaningful_count / total_rows if total_rows > 0 else 0
            non_empty_ratio = non_empty_count / total_rows if total_rows > 0 else 0
            
            # Keep column if it has meaningful content
            keep_column = (
                meaningful_ratio >= 0.1 or  # At least 10% meaningful content
                (non_empty_ratio >= 0.3 and col_idx <= 1) or  # First two columns with decent content
                (col_idx == 0 and non_empty_ratio >= 0.2)  # First column with some content
            )
            
            print(f"   Column {col_idx} ({col_name}): {meaningful_count}/{total_rows} meaningful "
                  f"({meaningful_ratio:.1%}), keep: {keep_column}")
            
            if keep_column:
                columns_to_keep.append(col_idx)
        
        # If we would remove too many columns, keep the original structure
        if len(columns_to_keep) < max(2, df.shape[1] * 0.5):
            print(f"   ⚠️ Would remove too many columns, keeping original structure")
            return df
        
        # Remove extraneous columns
        if len(columns_to_keep) < df.shape[1]:
            removed_count = df.shape[1] - len(columns_to_keep)
            df_cleaned = df.iloc[:, columns_to_keep].copy()
            print(f"   🔧 Removed {removed_count} extraneous columns")
            print(f"   📊 New shape: {df_cleaned.shape}")
            return df_cleaned
        else:
            print(f"   ✅ No extraneous columns detected")
            return df
    
    def _smart_align_continuation_table(self, main_df: pd.DataFrame, cont_df: pd.DataFrame) -> pd.DataFrame:
        """
        Intelligently align continuation table columns with main table after cleaning.
        This method respects the cleaned column structure and avoids force-removing meaningful columns.
        
        Args:
            main_df (pd.DataFrame): Main transaction table (already cleaned)
            cont_df (pd.DataFrame): Continuation table (already cleaned)
            
        Returns:
            pd.DataFrame: Properly aligned continuation table
        """
        if cont_df.empty:
            return cont_df
            
        main_cols = main_df.shape[1]
        cont_cols = cont_df.shape[1]
        
        print(f"   🔗 Smart alignment: main({main_cols} cols) vs cont({cont_cols} cols)")
        
        # If column counts match exactly, just align column names
        if cont_cols == main_cols:
            cont_df.columns = main_df.columns
            print(f"   ✅ Perfect match - aligned column names")
            return cont_df
        
        # If continuation has fewer columns, add padding columns
        elif cont_cols < main_cols:
            missing_cols = main_cols - cont_cols
            for i in range(missing_cols):
                cont_df[f'Col_{cont_cols + i}'] = pd.NA
            cont_df.columns = main_df.columns
            print(f"   🔧 Added {missing_cols} padding columns")
            return cont_df
        
        # If continuation has more columns, try to intelligently select the best matching ones
        else:
            print(f"   🔍 Continuation has {cont_cols - main_cols} extra columns, finding best alignment...")
            
            # Try different column selections to find the best match
            best_score = -1
            best_selection = list(range(main_cols))
            
            # Test different starting positions (shift left/right)
            for start_offset in range(min(3, cont_cols - main_cols + 1)):
                end_pos = start_offset + main_cols
                if end_pos <= cont_cols:
                    selection = list(range(start_offset, end_pos))
                    score = self._score_column_alignment(main_df, cont_df.iloc[:, selection])
                    
                    if score > best_score:
                        best_score = score
                        best_selection = selection
            
            # Apply the best column selection
            selected_cont_df = cont_df.iloc[:, best_selection].copy()
            selected_cont_df.columns = main_df.columns
            
            removed_cols = cont_cols - len(best_selection)
            print(f"   🎯 Selected columns {best_selection} (removed {removed_cols} columns, score: {best_score:.2f})")
            
            return selected_cont_df
    
    def _score_column_alignment(self, main_df: pd.DataFrame, cont_df: pd.DataFrame) -> float:
        """
        Score how well two tables align based on data patterns.
        
        Args:
            main_df (pd.DataFrame): Main table
            cont_df (pd.DataFrame): Continuation table subset
            
        Returns:
            float: Alignment score (higher is better)
        """
        if main_df.shape[1] != cont_df.shape[1]:
            return 0.0
            
        total_score = 0.0
        
        for col_idx in range(main_df.shape[1]):
            # Get sample data from both columns
            main_sample = main_df.iloc[:min(5, main_df.shape[0]), col_idx].astype(str)
            cont_sample = cont_df.iloc[:min(5, cont_df.shape[0]), col_idx].astype(str)
            
            # Check for pattern similarities
            patterns_match = 0
            pattern_checks = 0
            
            # Check for date patterns
            pattern_checks += 1
            main_has_dates = any(any(p in str(val).lower() for p in ['-', '/', '202', '201']) for val in main_sample)
            cont_has_dates = any(any(p in str(val).lower() for p in ['-', '/', '202', '201']) for val in cont_sample)
            if main_has_dates == cont_has_dates:
                patterns_match += 1
            
            # Check for numeric patterns
            pattern_checks += 1
            main_has_numbers = any(any(c.isdigit() for c in str(val)) for val in main_sample)
            cont_has_numbers = any(any(c.isdigit() for c in str(val)) for val in cont_sample)
            if main_has_numbers == cont_has_numbers:
                patterns_match += 1
            
            # Check for currency patterns
            pattern_checks += 1
            main_has_currency = any(any(c in str(val) for c in ['₹', '$', '.', ',']) for val in main_sample)
            cont_has_currency = any(any(c in str(val) for c in ['₹', '$', '.', ',']) for val in cont_sample)
            if main_has_currency == cont_has_currency:
                patterns_match += 1
            
            # Calculate column score
            col_score = patterns_match / pattern_checks if pattern_checks > 0 else 0
            total_score += col_score
        
        return total_score / main_df.shape[1] if main_df.shape[1] > 0 else 0.0
    
    def _ensure_clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame contains only clean scalar values, not pandas objects.
        This prevents Series object corruption during concatenation.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame with scalar values only
        """
        if df.empty:
            return df
            
        print(f"🧹 Ensuring clean DataFrame (shape: {df.shape})")
        
        # Create a copy to avoid modifying the original
        clean_df = df.copy()
        
        # Iterate through each cell and ensure it's a scalar value
        for row_idx in range(clean_df.shape[0]):
            for col_idx in range(clean_df.shape[1]):
                cell_value = clean_df.iloc[row_idx, col_idx]
                
                # Check if the cell contains a pandas object
                if isinstance(cell_value, (pd.Series, pd.DataFrame)):
                    print(f"   ⚠️ Found pandas object in cell [{row_idx}, {col_idx}]: {type(cell_value)}")
                    # Convert to string representation or extract scalar value
                    if isinstance(cell_value, pd.Series):
                        if len(cell_value) == 1:
                            clean_df.iloc[row_idx, col_idx] = cell_value.iloc[0]
                        else:
                            clean_df.iloc[row_idx, col_idx] = str(cell_value.iloc[0]) if len(cell_value) > 0 else ""
                    else:
                        clean_df.iloc[row_idx, col_idx] = str(cell_value)
                
                # Also check for other problematic types
                elif hasattr(cell_value, '__iter__') and not isinstance(cell_value, (str, int, float, type(None))):
                    if hasattr(cell_value, '__len__') and len(cell_value) > 0:
                        # For lists, tuples, etc., take the first element or join them
                        try:
                            if isinstance(cell_value, (list, tuple)) and len(cell_value) == 1:
                                clean_df.iloc[row_idx, col_idx] = str(cell_value[0])
                            else:
                                clean_df.iloc[row_idx, col_idx] = str(cell_value)
                        except:
                            clean_df.iloc[row_idx, col_idx] = str(cell_value)
                    else:
                        clean_df.iloc[row_idx, col_idx] = ""
        
        # Ensure all values are converted to appropriate string types
        for col in clean_df.columns:
            clean_df[col] = clean_df[col].astype(str)
        
        print(f"   ✅ DataFrame cleaned successfully")
        return clean_df
    
    def _are_tables_compatible(self, main_df: pd.DataFrame, candidate_df: pd.DataFrame) -> bool:
        """
        Check if two tables can be concatenated (same structure).
        
        Args:
            main_df (pd.DataFrame): Main transaction table
            candidate_df (pd.DataFrame): Candidate table for concatenation
            
        Returns:
            bool: True if tables are compatible for concatenation
        """
        # Both should be non-empty
        if main_df.empty or candidate_df.empty:
            return False
        
        # Check if candidate table is actually transaction data (has date-like patterns)
        # Look for date patterns in any column, not just the first one
        total_date_like_count = 0
        total_cells = 0
        
        for col_idx in range(min(3, candidate_df.shape[1])):  # Check first 3 columns
            col_data = candidate_df.iloc[:, col_idx].astype(str)
            date_like_count = sum(1 for val in col_data 
                                if any(pattern in val.lower() for pattern in ['-', '/', '2024', '2023', '2022', '2021']))
            total_date_like_count += date_like_count
            total_cells += len(col_data)
        
        # If less than 15% of first few columns have date-like patterns, probably not transaction data
        date_pattern_ratio = total_date_like_count / total_cells if total_cells > 0 else 0
        if date_pattern_ratio < 0.15:
            return False
        
        # Allow some flexibility in column count for bank statements
        # Continuation tables often have slightly different column structures
        main_cols = main_df.shape[1]
        candidate_cols = candidate_df.shape[1]
        col_diff = abs(main_cols - candidate_cols)
        
        # Allow up to 2 column difference (common in bank statements)
        if col_diff > 2:
            return False
        
        # If candidate table has header row that should be skipped, analyze data rows
        if candidate_df.shape[0] > 1:
            # Use last few rows to avoid headers
            main_sample = main_df.tail(3).astype(str)
            candidate_sample = candidate_df.tail(3).astype(str)
            
            # Check compatibility based on data patterns in overlapping columns
            min_cols = min(main_cols, candidate_cols)
            compatibility_score = 0
            
            for col_idx in range(min_cols):
                main_col = main_sample.iloc[:, col_idx]
                candidate_col = candidate_sample.iloc[:, col_idx]
                
                # Check if columns have similar characteristics
                main_has_numbers = any(any(c.isdigit() for c in str(val)) for val in main_col)
                candidate_has_numbers = any(any(c.isdigit() for c in str(val)) for val in candidate_col)
                
                main_has_dates = any(any(pattern in str(val).lower() for pattern in ['-', '/']) for val in main_col)
                candidate_has_dates = any(any(pattern in str(val).lower() for pattern in ['-', '/']) for val in candidate_col)
                
                # Give points for similar patterns
                if main_has_numbers == candidate_has_numbers:
                    compatibility_score += 1
                if main_has_dates == candidate_has_dates:
                    compatibility_score += 1
            
            # Compatible if more than 50% of overlapping columns have similar patterns
            return compatibility_score >= (min_cols * 0.5)
        
        return True
    
    def _remove_header_rows(self, df: pd.DataFrame, main_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove header rows from continuation tables.
        
        Args:
            df (pd.DataFrame): Table that might have header rows
            main_df (pd.DataFrame): Main table for reference
            
        Returns:
            pd.DataFrame: Table with header rows removed
        """
        if df.empty or df.shape[0] <= 1:
            return df
        
        # Convert to string for comparison
        df_str = df.astype(str)
        main_str = main_df.astype(str)
        
        # Check if first row looks like headers
        first_row = df_str.iloc[0]
        
        # If first row contains many non-numeric values while data rows have numbers, it's likely a header
        first_row_numeric_count = sum(1 for val in first_row if any(c.isdigit() for c in val))
        
        if df.shape[0] > 2:
            second_row = df_str.iloc[1]
            second_row_numeric_count = sum(1 for val in second_row if any(c.isdigit() for c in val))
            
            # If second row is much more numeric than first row, first row is likely header
            if second_row_numeric_count > first_row_numeric_count + 1:
                return df.iloc[1:].reset_index(drop=True)
        
        return df
    
    def _is_balance_row(self, row_text: str) -> bool:
        """
        Check if a row contains balance-related information.
        
        Args:
            row_text (str): Row text to check
            
        Returns:
            bool: True if row contains balance information
        """
        balance_keywords = [
            'opening balance', 'closing balance', 'balance b/f', 'balance c/f',
            'balance brought forward', 'balance carried forward', 'total balance',
            'balance total', 'net balance'
        ]
        row_text_lower = str(row_text).lower()
        return any(keyword in row_text_lower for keyword in balance_keywords)
    
    def _extract_and_set_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract headers from the table and set them as column names, then remove header rows.
        
        Args:
            df (pd.DataFrame): Main transaction table
            
        Returns:
            pd.DataFrame: Table with proper column names and header rows removed
        """
        if df.empty or df.shape[0] <= 1:
            return df
        
        cleaned_df = df.copy()
        header_candidates = []
        rows_to_remove = []
        
        # Check first few rows for header patterns
        for i in range(min(3, len(df))):
            row = df.iloc[i].astype(str)
            
            # Header indicators - banking terms that suggest this is a header row
            header_indicators = [
                'date', 'transaction', 'description', 'amount', 'balance', 'debit', 'credit',
                'reference', 'particulars', 'details', 'value', 'withdrawal', 'deposit',
                'txn', 'ref', 'chq', 'cheque', 'narration', 'opening', 'closing'
            ]
            
            row_text = ' '.join(row.values).lower()
            header_word_count = sum(1 for indicator in header_indicators if indicator in row_text)
            
            # Count numbers in the row
            number_count = sum(1 for val in row if any(c.isdigit() for c in val))
            
            # If row contains header words and few/no numbers, it's likely a header
            if header_word_count >= 2 and number_count <= 1:
                header_candidates.append((i, row.values.tolist()))
                rows_to_remove.append(i)
            elif i == 0 and header_word_count >= 1 and number_count == 0:
                # First row with header words and no numbers is likely a header
                header_candidates.append((i, row.values.tolist()))
                rows_to_remove.append(i)
        
        # Use the best header row to set column names
        if header_candidates:
            # Choose the header with most banking keywords
            best_header = None
            best_score = 0
            
            for row_idx, header_values in header_candidates:
                header_text = ' '.join(str(val).lower() for val in header_values)
                banking_keywords = [
                    'date', 'transaction', 'description', 'amount', 'balance', 'debit', 'credit',
                    'reference', 'particulars', 'details', 'narration', 'withdrawal', 'deposit'
                ]
                score = sum(1 for keyword in banking_keywords if keyword in header_text)
                
                if score > best_score:
                    best_score = score
                    best_header = header_values
            
            # Set column names from the best header
            if best_header:
                # Clean up header names
                clean_headers = []
                for header in best_header:
                    if pd.isna(header) or str(header).strip() in ['', 'nan', 'None']:
                        clean_headers.append(f'Column_{len(clean_headers)}')
                    else:
                        # Clean the header name
                        clean_name = str(header).strip()
                        # Remove common artifacts
                        clean_name = clean_name.replace('Account Statement', '').strip()
                        if clean_name == '' or clean_name.lower() in ['nan', 'none']:
                            clean_name = f'Column_{len(clean_headers)}'
                        clean_headers.append(clean_name)
                
                # Ensure we have the right number of columns
                while len(clean_headers) < df.shape[1]:
                    clean_headers.append(f'Column_{len(clean_headers)}')
                
                cleaned_df.columns = clean_headers[:df.shape[1]]
        
        # Remove identified header rows
        if rows_to_remove:
            # Check each row before removing
            final_rows_to_remove = []
            for idx in rows_to_remove:
                row_text = ' '.join(str(val) for val in df.iloc[idx])
                # Only remove if not a balance row
                if not self._is_balance_row(row_text):
                    final_rows_to_remove.append(idx)
            
            if final_rows_to_remove:
                cleaned_df = cleaned_df.drop(final_rows_to_remove).reset_index(drop=True)
        
        return cleaned_df
    
    def _is_duplicate_row(self, row1: pd.Series, row2: pd.Series, column_types: Dict[str, str]) -> bool:
        """
        Check if two rows represent the same transaction.
        
        Args:
            row1 (pd.Series): First row to compare
            row2 (pd.Series): Second row to compare
            column_types (dict): Column type mapping
            
        Returns:
            bool: True if rows represent the same transaction
        """
        # Convert rows to string for comparison
        row1_str = row1.astype(str)
        row2_str = row2.astype(str)
        
        # Must match exactly on date
        date_cols = [col for col, type_ in column_types.items() if type_ == 'date']
        if date_cols:
            date_col = date_cols[0]
            if row1_str[date_col].strip() != row2_str[date_col].strip():
                return False
        
        # Must match exactly on reference number if present
        ref_cols = [col for col, type_ in column_types.items() if type_ == 'reference']
        for ref_col in ref_cols:
            ref1 = ''.join(c for c in row1_str[ref_col] if c.isalnum())
            ref2 = ''.join(c for c in row2_str[ref_col] if c.isalnum())
            if ref1 and ref2 and ref1 != ref2:  # If both have ref numbers and they're different
                return False
        
        # Must match exactly on amount
        amount_cols = [col for col, type_ in column_types.items() 
                      if type_ in ['amount_dr', 'amount_cr', 'balance']]
        for amount_col in amount_cols:
            # Clean amounts for comparison
            amt1 = self._clean_amount(row1_str[amount_col])
            amt2 = self._clean_amount(row2_str[amount_col])
            if amt1 is not None and amt2 is not None and abs(amt1 - amt2) > 0.01:
                return False
        
        # Must match on description (ignoring case and whitespace)
        desc_cols = [col for col, type_ in column_types.items() if type_ == 'description']
        if desc_cols:
            desc_col = desc_cols[0]
            desc1 = ''.join(c.lower() for c in row1_str[desc_col] if not c.isspace())
            desc2 = ''.join(c.lower() for c in row2_str[desc_col] if not c.isspace())
            if desc1 != desc2:
                return False
        
        return True
    
    def _remove_irrelevant_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows that are definitely not transactions (headers, footers, etc).
        Only removes rows that are 100% certain to be non-transactional.
        
        Args:
            df (pd.DataFrame): DataFrame to clean
            
        Returns:
            pd.DataFrame: DataFrame with irrelevant rows removed
        """
        if df.empty:
            return df
        
        cleaned_df = df.copy()
        rows_to_remove = []
        
        # Only remove rows that are definitely not transactions
        non_transaction_phrases = [
            'statement of account',
            'page',
            'continued from previous page',
            'continued on next page',
            'statement continued',
            'total',
            'subtotal',
            'grand total',
            'balance carried forward',
            'balance brought forward',
            'this is a computer generated statement'
        ]
        
        for i in range(len(df)):
            row = df.iloc[i].astype(str)
            row_text = ' '.join(str(val).lower().strip() for val in row if pd.notna(val))
            
            # Only remove if the entire row matches one of the non-transaction phrases exactly
            if any(phrase in row_text for phrase in non_transaction_phrases):
                # Additional check: if row has a valid date and amount, keep it
                has_date = any(
                    bool(re.search(r'\d{2}[-/]\d{2}[-/]\d{2,4}', str(val)))
                    for val in row
                )
                has_amount = any(
                    bool(re.search(r'\d+[,.]?\d*', str(val)))
                    for val in row
                )
                
                # If it has both date and amount, it's probably a transaction - keep it
                if not (has_date and has_amount):
                        rows_to_remove.append(i)
        
        # Remove identified non-transaction rows
        if rows_to_remove:
            cleaned_df = cleaned_df.drop(rows_to_remove).reset_index(drop=True)
        
        return cleaned_df
    
    def _realign_continuation_table(self, main_df: pd.DataFrame, cont_df: pd.DataFrame) -> pd.DataFrame:
        """
        Realign a continuation table that might have column shifts.
        
        Args:
            main_df (pd.DataFrame): Main transaction table for reference
            cont_df (pd.DataFrame): Continuation table that might be misaligned
            
        Returns:
            pd.DataFrame: Realigned continuation table
        """
        if cont_df.empty:
            return cont_df
            
        # Create a copy to avoid modifying original
        realigned_df = cont_df.copy()
        
        # Check for potential misalignment by comparing data patterns
        def get_column_pattern(df, col_idx):
            """Get pattern characteristics of a column"""
            col_data = df.iloc[:, col_idx].astype(str)
            has_dates = any(any(pattern in str(val).lower() for pattern in ['-', '/']) for val in col_data)
            has_numbers = any(any(c.isdigit() for c in str(val)) for val in col_data)
            has_currency = any(any(c in str(val) for c in ['₹', '$', '€', '£', '.', ',']) for val in col_data)
            return {'dates': has_dates, 'numbers': has_numbers, 'currency': has_currency}
        
        # Get patterns for main table columns
        main_patterns = [get_column_pattern(main_df, i) for i in range(main_df.shape[1])]
        cont_patterns = [get_column_pattern(realigned_df, i) for i in range(realigned_df.shape[1])]
        
        # Check if columns are shifted left
        shift_left = 0
        max_shift = min(3, realigned_df.shape[1] - 1)  # Limit maximum shift
        
        for shift in range(max_shift):
            matches = 0
            for i in range(len(main_patterns) - shift):
                if i + shift < len(cont_patterns):
                    main_pat = main_patterns[i]
                    cont_pat = cont_patterns[i + shift]
                    if main_pat == cont_pat:
                        matches += 1
            
            if matches > len(main_patterns) * 0.5:  # More than 50% match
                shift_left = shift
                break
        
        # Check if columns are shifted right
        shift_right = 0
        for shift in range(max_shift):
            matches = 0
            for i in range(shift, len(main_patterns)):
                if i - shift < len(cont_patterns):
                    main_pat = main_patterns[i]
                    cont_pat = cont_patterns[i - shift]
                    if main_pat == cont_pat:
                        matches += 1
            
            if matches > len(main_patterns) * 0.5:  # More than 50% match
                shift_right = shift
                break
        
        # Apply the shift that gives better alignment
        if shift_left > 0:
            # Shift columns left
            realigned_df = realigned_df.iloc[:, shift_left:].reset_index(drop=True)
            # Add empty columns at the end if needed
            while realigned_df.shape[1] < main_df.shape[1]:
                realigned_df[f'temp_col_{realigned_df.shape[1]}'] = pd.NA
        elif shift_right > 0:
            # Add empty columns at the start
            for i in range(shift_right):
                realigned_df.insert(0, f'temp_col_{i}', pd.NA)
            # Trim excess columns if any
            if realigned_df.shape[1] > main_df.shape[1]:
                realigned_df = realigned_df.iloc[:, :main_df.shape[1]]
        
        return realigned_df

    def _get_column_type(self, col_name: str) -> str:
        """
        Determine the type of a column based on its name.
        
        Args:
            col_name (str): Column name to analyze
            
        Returns:
            str: Column type ('date', 'amount_dr', 'amount_cr', 'balance', 'reference', 'description', 'other')
        """
        col_lower = str(col_name).lower()
        
        # Date columns
        if any(term in col_lower for term in ['date', 'dt', 'value']):
            return 'date'
            
        # Debit/Withdrawal columns
        if any(term in col_lower for term in ['dr', 'debit', 'withdrawal', 'paid', 'outward']):
            return 'amount_dr'
            
        # Credit/Deposit columns
        if any(term in col_lower for term in ['cr', 'credit', 'deposit', 'received', 'inward']):
            return 'amount_cr'
            
        # Balance columns
        if 'balance' in col_lower:
            return 'balance'
            
        # Reference number columns
        if any(term in col_lower for term in ['ref', 'cheque', 'chq', 'no.', 'number']):
            return 'reference'
            
        # Description/Remarks columns
        if any(term in col_lower for term in ['desc', 'narration', 'remark', 'particular']):
            return 'description'
            
        return 'other'

    def _clean_amount(self, amount_str: str) -> Optional[float]:
        """
        Clean and convert amount string to float, handling various formats.
        
        Args:
            amount_str (str): Amount string to clean
            
        Returns:
            float or None: Cleaned amount or None if invalid
        """
        if pd.isna(amount_str) or not str(amount_str).strip():
            return None
            
        try:
            # Convert to string and clean basic artifacts
            amount = str(amount_str).strip()
            
            # Handle debit/credit indicators
            is_debit = False
            if '(dr)' in amount.lower() or '(debit)' in amount.lower():
                is_debit = True
                amount = amount.lower().replace('(dr)', '').replace('(debit)', '')
            elif '(cr)' in amount.lower() or '(credit)' in amount.lower():
                amount = amount.lower().replace('(cr)', '').replace('(credit)', '')
            
            # Remove any remaining parentheses and extra spaces
            amount = amount.replace('(', '').replace(')', '').strip()
            
            # Remove currency symbols and commas
            for symbol in ['₹', '$', '€', '£', ',']:
                amount = amount.replace(symbol, '')
            
            # Convert to float
            value = float(amount.strip())
            
            # Apply debit sign if needed
            return -value if is_debit else value
            
        except:
            return None

    def _fix_column_alignment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix column alignment issues in the dataframe.
        Specifically handles cases where columns shift left after certain rows.
        
        Args:
            df (pd.DataFrame): DataFrame to fix
            
        Returns:
            pd.DataFrame: Fixed DataFrame
        """
        if df.empty:
            return df
            
        fixed_df = df.copy()
        
        # Identify column types
        column_types = {col: self._get_column_type(col) for col in fixed_df.columns}
        
        # Find numeric columns
        numeric_cols = [col for col, type_ in column_types.items() 
                       if type_ in ['amount_dr', 'amount_cr', 'balance']]
        
        # Get column indices for key columns
        cols = list(fixed_df.columns)
        ref_cols = [i for i, col in enumerate(cols) if column_types[col] == 'reference']
        desc_cols = [i for i, col in enumerate(cols) if column_types[col] == 'description']
        dr_cols = [i for i, col in enumerate(cols) if column_types[col] == 'amount_dr']
        cr_cols = [i for i, col in enumerate(cols) if column_types[col] == 'amount_cr']
        balance_cols = [i for i, col in enumerate(cols) if column_types[col] == 'balance']
        
        # Use first found column of each type
        ref_idx = ref_cols[0] if ref_cols else None
        desc_idx = desc_cols[0] if desc_cols else None
        dr_idx = dr_cols[0] if dr_cols else None
        cr_idx = cr_cols[0] if cr_cols else None
        balance_idx = balance_cols[0] if balance_cols else None
        
        # Detect rows where alignment shifts
        rows_to_fix = []
        for idx, row in fixed_df.iterrows():
            is_misaligned = False
            
            # Check multiple conditions for misalignment
            if ref_idx is not None and desc_idx is not None:
                ref_val = str(row[cols[ref_idx]])
                desc_val = str(row[cols[desc_idx]])
                
                # Condition 1: Reference number contains transaction-like text
                if any(pattern in ref_val.upper() for pattern in ['NEFT', 'IMPS', 'INFT', 'BIL/', '/IMPS/']):
                    is_misaligned = True
                
                # Condition 2: Description column contains only numbers
                if desc_val and desc_val.strip():
                    cleaned_val = desc_val.replace(',', '').replace('.', '').replace(' ', '')
                    if cleaned_val.isdigit():
                        is_misaligned = True
                
                # Condition 3: Check for empty debit/credit with numbers in description
                if dr_idx and cr_idx:
                    dr_val = str(row[cols[dr_idx]])
                    cr_val = str(row[cols[cr_idx]])
                    if (dr_val.strip() in ['', 'nan', 'None'] and 
                        cr_val.strip() in ['', 'nan', 'None'] and
                        any(c.isdigit() for c in desc_val)):
                        is_misaligned = True
            
            if is_misaligned:
                rows_to_fix.append(idx)
        
        if rows_to_fix:
            # Fix each misaligned row
            for idx in rows_to_fix:
                row = fixed_df.iloc[idx].copy()
                
                # Store original values
                if ref_idx is not None and desc_idx is not None:
                    # Move description to correct column
                    fixed_df.at[idx, cols[desc_idx]] = row[cols[ref_idx]]
                    fixed_df.at[idx, cols[ref_idx]] = None
                    
                    # Handle numeric columns
                    if dr_idx and cr_idx and balance_idx:
                        # Get values
                        dr_val = self._clean_amount(row[cols[desc_idx]])  # Original desc now contains dr
                        cr_val = self._clean_amount(row[cols[dr_idx]])    # Original dr now contains cr
                        bal_val = self._clean_amount(row[cols[cr_idx]])   # Original cr now contains balance
                        
                        # Set values in correct columns with proper formatting
                        fixed_df.at[idx, cols[dr_idx]] = f"{abs(dr_val):,.2f}" if dr_val and dr_val < 0 else None
                        fixed_df.at[idx, cols[cr_idx]] = f"{cr_val:,.2f}" if cr_val and cr_val > 0 else None
                        fixed_df.at[idx, cols[balance_idx]] = f"{bal_val:,.2f}" if bal_val is not None else None
        
        # Ensure consistent decimal formatting for all numeric columns
        for col in numeric_cols:
            fixed_df[col] = fixed_df[col].apply(lambda x: 
                f"{abs(self._clean_amount(x)):,.2f}" if self._clean_amount(x) is not None else None
            )
        
        # Remove any trailing commas in all columns
        for col in fixed_df.columns:
            fixed_df[col] = fixed_df[col].apply(
                lambda x: str(x).rstrip(',') if pd.notna(x) else x
            )
        
        return fixed_df

    def _extract_header_info(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Extract header information from a table.
        
        Args:
            df (pd.DataFrame): Table to analyze
            
        Returns:
            dict: Extracted header information
        """
        headers = {}
        
        # Convert all text to string and lowercase for matching
        df_text = df.astype(str).apply(lambda x: x.str.lower())
        
        # Look for account number
        account_patterns = ['account no', 'a/c no', 'account number']
        for pattern in account_patterns:
            for col in df_text.columns:
                col_data = df_text[col].astype(str)
                for idx, val in enumerate(col_data):
                    if pattern in val:
                        # Look in the same row, next column
                        if col < len(df.columns) - 1:
                            acc_val = str(df.iloc[idx, col + 1])
                            # Clean and validate account number
                            acc_val = ''.join(c for c in acc_val if c.isdigit())
                            if len(acc_val) >= 8:  # Most bank account numbers are at least 8 digits
                                headers['account_number'] = acc_val
                                break
        
        # Look for IFSC code
        ifsc_patterns = ['ifsc', 'branch code']
        for pattern in ifsc_patterns:
            for col in df_text.columns:
                col_data = df_text[col].astype(str)
                for idx, val in enumerate(col_data):
                    if pattern in val:
                        # Look in the same row, next column
                        if col < len(df.columns) - 1:
                            ifsc_val = str(df.iloc[idx, col + 1])
                            # IFSC codes are 11 characters
                            if len(ifsc_val) == 11 and ifsc_val[:4].isalpha() and ifsc_val[4:].isdigit():
                                headers['ifsc_code'] = ifsc_val
                                break
        
        return headers

    def _process_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a table DataFrame to clean data without removing any rows.
        
        Args:
            df (pd.DataFrame): Raw table from img2table
            
        Returns:
            pd.DataFrame: Processed table with all rows preserved
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Clean text without removing rows
        processed_df = self._clean_dataframe_text(processed_df)
        
        return processed_df

    def extract_bank_statement_table(self, pdf_path: str, cleanup_temp: bool = True) -> Optional[pd.DataFrame]:
        try:
            print(f"🏦 Processing bank statement: {os.path.basename(pdf_path)}")
            
            # Extract all tables from PDF
            result = self.table_extractor.process_document(pdf_path)
            
            if not result['extracted_tables']:
                print("❌ No tables found in the PDF")
                return None
            
            print(f"📊 Found {len(result['extracted_tables'])} table(s) - analyzing for bank statement...")
            
            # Load extracted tables and analyze them
            excel_file = None
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            # Try to find the Excel file in the expected directory
            search_directories = [
                self.table_extractor.output_dir,
                os.path.join(self.table_extractor.output_dir, "extracted_tables"),
                self.output_dir,
                os.path.join(self.table_extractor.output_dir, "temp_ocr_processing")
            ]
            
            # More flexible file matching - try different naming patterns
            possible_names = [
                base_name,
                base_name.replace("_readable", ""),
                base_name.replace("_reconstructed", ""),
                base_name.replace("_temp", ""),
                os.path.splitext(base_name)[0]
            ]
            
            for search_dir in search_directories:
                if os.path.exists(search_dir):
                    for file in os.listdir(search_dir):
                        if file.endswith('.xlsx'):
                            if any(name == file.replace('.xlsx', '') for name in possible_names):
                                excel_file = os.path.join(search_dir, file)
                                print(f"   ✅ Found Excel file (exact match): {excel_file}")
                                break
                            elif any(name in file for name in possible_names):
                                excel_file = os.path.join(search_dir, file)
                                print(f"   ✅ Found Excel file (partial match): {excel_file}")
                                break
                    if excel_file:
                        break
            
            if not excel_file or not os.path.exists(excel_file):
                print("❌ Could not find extracted tables Excel file")
                return None
            
            # Read all sheets and analyze
            excel_data = pd.ExcelFile(excel_file)
            tables_analysis = []
            
            print("\n📋 Raw tables extracted from img2table:")
            for sheet_name in excel_data.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                print(f"\n   📑 Sheet: {sheet_name}")
                print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                print("   Raw data:")
                print(df.to_string())
                
                # Only clean text, no row removal
                df = self._clean_dataframe_text(df)
                print("   After cleaning:")
                print(df.to_string())
                
                # Find corresponding table info
                table_info = None
                for table in result['extracted_tables']:
                    if table['sheet_name'] == sheet_name:
                        table_info = table
                        break
                
                if table_info is None:
                    continue
                
                is_transaction, confidence = self._is_transaction_table(df, table_info)
                print(f"   Transaction table: {is_transaction} (confidence: {confidence})")
                
                tables_analysis.append({
                    'sheet_name': sheet_name,
                    'dataframe': df,
                    'table_info': table_info,
                    'is_transaction': is_transaction,
                    'confidence': confidence,
                    'page_number': table_info['page_number'],
                    'table_index': table_info['table_index']
                })
                
            # Look for header information in non-transaction tables
            header_info = {}
            for table in tables_analysis:
                if not table['is_transaction']:
                    table_headers = self._extract_header_info(table['dataframe'])
                    header_info.update(table_headers)
            
            if header_info:
                print("\n📋 Found header information in tables:")
                for key, value in header_info.items():
                    print(f"   ✅ {key}: {value}")
            
            # Find the main transaction table (highest confidence)
            transaction_tables = [t for t in tables_analysis if t['is_transaction']]
            
            if not transaction_tables:
                print("❌ No transaction tables identified")
                return None
            
            # Sort by confidence (highest first)
            transaction_tables.sort(key=lambda x: x['confidence'], reverse=True)
            main_table = transaction_tables[0]
            
            print(f"\n✅ Main transaction table: {main_table['sheet_name']} "
                  f"({main_table['dataframe'].shape[0]}×{main_table['dataframe'].shape[1]})")
            
            # Start with the main table
            consolidated_df = main_table['dataframe'].copy()
            
            # Remove extraneous columns before processing
            consolidated_df = self._remove_extraneous_columns(consolidated_df)
            
            # Only set column names from first row if it looks like a header
            if not consolidated_df.empty and consolidated_df.shape[0] > 0:
                first_row = consolidated_df.iloc[0]
                first_row_text = ' '.join(str(val).lower() for val in first_row)
                header_keywords = ['date', 'transaction', 'description', 'amount', 'balance', 'debit', 'credit']
                if any(keyword in first_row_text for keyword in header_keywords):
                    print("   ✅ Found header row:")
                    print("   ", first_row.to_list())
                    consolidated_df.columns = first_row
                    consolidated_df = consolidated_df.iloc[1:].reset_index(drop=True)
            
            # Look for continuation tables
            potential_continuations = []
            for table in tables_analysis:
                if (table['sheet_name'] != main_table['sheet_name'] and 
                    table['is_transaction'] and
                    self._are_tables_compatible(main_table['dataframe'], table['dataframe'])):
                    potential_continuations.append(table)
            
            # Sort continuation tables by page and table order
            potential_continuations.sort(key=lambda x: (x['page_number'], x['table_index']))
            
            # Concatenate compatible tables
            for continuation in potential_continuations:
                print(f"\n📋 Processing continuation table: {continuation['sheet_name']}")
                cont_df = continuation['dataframe'].copy()
                print("   Raw data:")
                print(cont_df.to_string())
                
                if not cont_df.empty:
                    # Remove extraneous columns from continuation table
                    cont_df = self._remove_extraneous_columns(cont_df)
                    
                    # Smart column alignment after cleaning
                    cont_df = self._smart_align_continuation_table(consolidated_df, cont_df)
                    
                    # Only remove first row if it matches column names
                    if not cont_df.empty and cont_df.shape[0] > 0:
                        first_row = cont_df.iloc[0]
                        if all(str(first_row[col]).lower() == str(col).lower() for col in cont_df.columns):
                            print("   🔧 Removing header row:")
                            print("   ", first_row.to_list())
                            cont_df = cont_df.iloc[1:].reset_index(drop=True)
                    
                    print("   After processing:")
                    print(cont_df.to_string())
                    
                    # Ensure both DataFrames are properly cleaned before concatenation
                    consolidated_df = self._ensure_clean_dataframe(consolidated_df)
                    cont_df = self._ensure_clean_dataframe(cont_df)
                    
                    # Concatenate tables
                    consolidated_df = pd.concat([consolidated_df, cont_df], ignore_index=True)
                    print(f"   ➕ Added {cont_df.shape[0]} rows")
            
            # Final cleanup without removing any rows
            if not consolidated_df.empty:
                # Ensure final DataFrame is completely clean
                consolidated_df = self._ensure_clean_dataframe(consolidated_df)
                
                # Clean any remaining Unicode artifacts
                consolidated_df = self._clean_dataframe_text(consolidated_df)
                
                # Reset index
                consolidated_df = consolidated_df.reset_index(drop=True)
            
            # print(f"\n🎉 Final consolidated bank statement: {consolidated_df.shape[0]}×{consolidated_df.shape[1]}")
            # print("Final data:")
            # print(consolidated_df.to_string())
            
            # Save consolidated table
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_file = os.path.join(self.output_dir, f"{base_name}_bank_statement.xlsx")
            consolidated_df.to_excel(output_file, index=False)
            print(f"💾 Saved to: {output_file}")
            
            return consolidated_df
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            if cleanup_temp and 'consolidated_df' in locals():
                self.table_extractor.cleanup_all_temp_files()
    
    def process_multiple_files(self, pdf_files: List[str], cleanup_temp: bool = True) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Process multiple PDF files and extract bank statements from each.
        
        Args:
            pdf_files (list): List of PDF file paths
            cleanup_temp (bool): Whether to clean up temporary files after each file
            
        Returns:
            dict: Dictionary mapping file names to extracted DataFrames
        """
        results = {}
        
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                file_name = os.path.basename(pdf_file)
                print(f"\n{'='*60}")
                print(f"Processing: {file_name}")
                print(f"{'='*60}")
                
                bank_statement = self.extract_bank_statement_table(pdf_file, cleanup_temp)
                results[file_name] = bank_statement
                
                if bank_statement is not None:
                    print(f"✅ Success: {bank_statement.shape[0]} transactions extracted")
                else:
                    print(f"❌ Failed to extract bank statement")
            else:
                print(f"⚠️ File not found: {pdf_file}")
                results[os.path.basename(pdf_file)] = None
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Initialize the bank statement extractor
    extractor = BankStatementExtractor(save_reconstructed_pdfs=True)
    
    # Test with a single file
    # test_file = "BankStatements SK2/axis_bank__statement_for_september_2024_unlocked.pdf"
    # test_file = "BankStatements SK2/hdfc.pdf"
    for file in os.listdir("BankStatements SK2"):
        if file.endswith(".pdf"):
            test_file = os.path.join("BankStatements SK2", file)
            print(f"Processing: {file}")
            if os.path.exists(test_file):
                print("🧪 Testing Bank Statement Extractor")
                print("="*50)
                
                bank_statement = extractor.extract_bank_statement_table(test_file)
                
                if bank_statement is not None:
                    print(f"\n📊 Final Bank Statement Summary:")
                    print(f"   Shape: {bank_statement.shape[0]} rows × {bank_statement.shape[1]} columns")
                    print(f"   Columns: {list(bank_statement.columns)}")
                    
                    final_bs = "BankStatements_Results"
                    os.makedirs(final_bs, exist_ok=True)
                    bank_statement.to_csv(os.path.basename(test_file).replace(".pdf", ".csv"), index=False)
                    # Show first few rows
                    print(f"\n📋 First 5 rows:")
                    print(bank_statement.head().to_string())
                    
                    # Show data types
                    print(f"\n🔍 Column info:")
                    for i, col in enumerate(bank_statement.columns):
                        sample_values = bank_statement[col].dropna().head(3).tolist()
                        print(f"   {i}: {col} - {sample_values}")
                
            else:
                print(f"❌ Test file not found: {test_file}")
                print("Available files in BankStatements SK2:")
                try:
                    for file in os.listdir("BankStatements SK2"):
                        if file.endswith(".pdf"):
                            print(f"  - {file}")
                except:
                    print("  Could not list directory") 