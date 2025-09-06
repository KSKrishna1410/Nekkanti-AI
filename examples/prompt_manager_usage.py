#!/usr/bin/env python3
"""
Example usage of the PromptManager for template-based prompt handling
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.prompt_manager import get_prompt_manager, render_prompt

def main():
    """Demonstrate PromptManager usage"""
    
    # Get the global prompt manager instance
    pm = get_prompt_manager()
    
    print("📋 Available templates:")
    templates = pm.list_templates()
    for template in templates:
        print(f"  - {template}")
    
    print("\n" + "="*50)
    
    # Example 1: Invoice extraction
    sample_invoice_text = """
    INVOICE
    Invoice No: INV-2024-001
    Date: 2024-01-15
    
    From: ABC Corp
    To: XYZ Ltd
    
    Items:
    1. Widget A - Qty: 10 - Rate: 100 - Amount: 1000
    2. Widget B - Qty: 5 - Rate: 200 - Amount: 1000
    
    Total: 2000
    """
    
    try:
        print("🧾 Invoice Extraction Example:")
        invoice_prompt = render_prompt('invoice_extraction.jinja', text=sample_invoice_text)
        print("Prompt generated successfully!")
        print(f"Prompt length: {len(invoice_prompt)} characters")
        print(f"First 200 characters:\n{invoice_prompt[:200]}...")
        
    except Exception as e:
        print(f"❌ Error with invoice template: {e}")
    
    
    print("\n✅ All examples completed!")

if __name__ == "__main__":
    main()