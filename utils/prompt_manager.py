"""
Prompt Manager for handling Jinja2 templates and prompt rendering
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from jinja2 import Environment, FileSystemLoader, Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages Jinja2 templates for prompts"""
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the prompt manager
        
        Args:
            templates_dir: Directory containing template files. If None, uses default.
        """
        if not JINJA2_AVAILABLE:
            raise ImportError("Jinja2 is required for PromptManager. Install with: pip install Jinja2")
        
        # Set default templates directory
        if templates_dir is None:
            # Get the project root directory
            current_file = Path(__file__).resolve()
            project_root = current_file.parents[1]  # Go up to project root
            templates_dir = project_root / "templates" / "prompts"
        
        self.templates_dir = Path(templates_dir)
        
        # Ensure templates directory exists
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters if needed
        self._setup_custom_filters()
        
        logger.info(f"PromptManager initialized with templates directory: {self.templates_dir}")
    
    def _setup_custom_filters(self):
        """Setup custom Jinja2 filters"""
        
        def safe_string(value):
            """Safely convert value to string"""
            return str(value) if value is not None else ""
        
        def truncate_text(text, max_length=1000):
            """Truncate text to max length"""
            text = str(text)
            if len(text) <= max_length:
                return text
            return text[:max_length] + "..."
        
        self.env.filters['safe_string'] = safe_string
        self.env.filters['truncate'] = truncate_text
    
    def render_template(self, template_name: str, **kwargs) -> str:
        """
        Render a template with given variables
        
        Args:
            template_name: Name of the template file (with .jinja extension)
            **kwargs: Variables to pass to the template
            
        Returns:
            Rendered template as string
            
        Raises:
            FileNotFoundError: If template file doesn't exist
            Exception: If template rendering fails
        """
        try:
            template = self.env.get_template(template_name)
            rendered = template.render(**kwargs)
            
            logger.debug(f"Rendered template '{template_name}' with {len(kwargs)} variables")
            return rendered
            
        except Exception as e:
            logger.error(f"Failed to render template '{template_name}': {e}")
            raise
    
    def get_template_path(self, template_name: str) -> Path:
        """Get the full path to a template file"""
        return self.templates_dir / template_name
    
    def template_exists(self, template_name: str) -> bool:
        """Check if a template file exists"""
        return self.get_template_path(template_name).exists()
    
    def list_templates(self) -> list:
        """List all available template files"""
        if not self.templates_dir.exists():
            return []
        
        return [f.name for f in self.templates_dir.iterdir() 
                if f.is_file() and f.suffix == '.jinja']
    
    def render_from_string(self, template_string: str, **kwargs) -> str:
        """
        Render a template from string (useful for dynamic templates)
        
        Args:
            template_string: Template content as string
            **kwargs: Variables to pass to the template
            
        Returns:
            Rendered template as string
        """
        try:
            template = self.env.from_string(template_string)
            return template.render(**kwargs)
        except Exception as e:
            logger.error(f"Failed to render template from string: {e}")
            raise
    
    def create_template_file(self, template_name: str, content: str):
        """
        Create a new template file
        
        Args:
            template_name: Name of the template file (should end with .jinja)
            content: Template content
        """
        if not template_name.endswith('.jinja'):
            template_name += '.jinja'
        
        template_path = self.get_template_path(template_name)
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Created template file: {template_path}")


# Global instance for easy access
_prompt_manager = None

def get_prompt_manager() -> PromptManager:
    """Get the global PromptManager instance"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def render_prompt(template_name: str, **kwargs) -> str:
    """Convenience function to render a prompt template"""
    return get_prompt_manager().render_template(template_name, **kwargs)