import os
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from .modelnames import get_available_models, get_model_config

class AIAssistant:
    def __init__(self):
        load_dotenv()
        self.available_models = get_available_models()
        if not self.available_models:
            raise ValueError("No LLM API keys found in environment variables")
            
        # Use the first available model by default
        self.model_name = next(iter(self.available_models))
        self._update_client()
        
    def _update_client(self):
        """Update the OpenAI client with the current model configuration."""
        config = get_model_config(self.model_name)
        if not config:
            raise ValueError(f"Model {self.model_name} is not available")
            
        self.client = OpenAI(
            api_key=os.getenv(config.api_key_env),
            base_url=config.base_url
        )
        self.model = config.model_name
        
    def set_model(self, model_name: str):
        """Set the model to use for this assistant."""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} is not available")
        self.model_name = model_name
        self._update_client()
        
    def get_available_model_names(self) -> List[str]:
        """Get a list of available model names."""
        return list(self.available_models.keys())
        
    def get_explanation(self, current_line: str, context_lines: List[str], 
                       reading_time: float) -> str:
        """Get an explanation for a difficult passage."""
        if not context_lines:
            return "No context available for explanation."
            
        # Prepare the prompt
        prompt = f"""I've been reading the following text and spent {reading_time:.1f} seconds on this line:
"{current_line}"

Here's the context (previous lines):
{chr(10).join(context_lines)}

Please provide a clear explanation of what this passage means, focusing on any complex concepts or ideas.
Keep the explanation concise and easy to understand."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful reading assistant that explains difficult passages in a clear and concise way."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error getting AI explanation: {e}")
            return "Sorry, I couldn't generate an explanation at this time."
            
    def get_summary(self, text: str) -> str:
        """Get a summary of a longer passage."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful reading assistant that provides concise summaries."},
                    {"role": "user", "content": f"Please provide a brief summary of the following text:\n\n{text}"}
                ],
                max_tokens=1024
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error getting summary: {e}")
            return "Sorry, I couldn't generate a summary at this time." 