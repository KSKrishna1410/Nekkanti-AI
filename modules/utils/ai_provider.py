#!/usr/bin/env python3
"""
AI Provider utility for Azure OpenAI services
"""

import os
import json
import logging
from typing import Dict, Any, Tuple

from openai import AzureOpenAI

logger = logging.getLogger(__name__)


class TokenUsage:
    def __init__(self):
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.provider: str = ""
        self.model: str = ""

    def accumulate(self, other_usage: 'TokenUsage'):
        """Accumulate token usage from another TokenUsage object."""
        self.input_tokens += other_usage.input_tokens
        self.output_tokens += other_usage.output_tokens
        self.total_cost_usd += other_usage.total_cost_usd
        if not self.provider:
            self.provider = other_usage.provider
        if not self.model:
            self.model = other_usage.model


class AIProviderClient:
    def __init__(self):
        # Azure OpenAI configuration
        self.endpoint = "https://usha-mewxgriq-eastus2.cognitiveservices.azure.com/"
        self.model_name = "gpt-4o-mini"
        self.deployment = "gpt-4o-mini"
        self.api_version = "2024-12-01-preview"
        
        # Get API key from environment
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable not set")
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=api_key,
            api_version=self.api_version
        )
        
        logger.info("✅ Using Azure OpenAI GPT-4o-mini provider")
    
    def generate_response(self, prompt: str) -> Tuple[str, TokenUsage]:
        """Generate AI response using Azure OpenAI"""
        token_usage = TokenUsage()
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,  # Use deployment name for Azure OpenAI
                messages=[
                    {"role": "system", "content": "You are an expert document data extraction system. Extract structured information and return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=8192
            )
            
            # Calculate token usage
            if response.usage:
                token_usage.input_tokens = response.usage.prompt_tokens
                token_usage.output_tokens = response.usage.completion_tokens
                # GPT-4o-mini pricing for Azure (approximate)
                token_usage.total_cost_usd = (
                    token_usage.input_tokens * 0.000150 / 1000 +  # GPT-4o-mini input pricing
                    token_usage.output_tokens * 0.000600 / 1000   # GPT-4o-mini output pricing
                )
                token_usage.provider = "azure-openai"
                token_usage.model = "gpt-4o-mini"
            
            logger.info(f"Azure OpenAI usage - Input: {token_usage.input_tokens}, Output: {token_usage.output_tokens}, Cost: ${token_usage.total_cost_usd:.4f}")
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            return response_text, token_usage
            
        except Exception as e:
            logger.error(f"Azure OpenAI generation error: {e}")
            return "", token_usage