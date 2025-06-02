import os
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from .modelnames import get_available_models, get_model_config
import logging
class AIAssistant:
    def __init__(self):
        load_dotenv()
        self.available_models = get_available_models()
        if not self.available_models:
            raise ValueError("No LLM API keys found in environment variables")
            
        # Use the first available model by default
        self.model_name = next(iter(self.available_models))
        self._update_client()
        
        # Conversation history for better context
        self.conversation_history = []
        
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
        
    def add_to_history(self, context: dict):
        """Add interaction context to conversation history."""
        self.conversation_history.append(context)
        # Keep only recent history (last 5 interactions)
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]
    
    def get_history_context(self) -> str:
        """Get conversation history as context string."""
        if not self.conversation_history:
            return ""
        
        history_parts = ["Previous reading session context:"]
        for i, ctx in enumerate(self.conversation_history):
            history_parts.append(f"- Page {ctx.get('page', 0) + 1}: {ctx.get('assistance', 'N/A')}")
        
        return "\n".join(history_parts) + "\n\n"
        
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
            
    def get_reading_assistance(self, page_line_times: dict, current_page: int, 
                              pdf_document, max_lines: int = 5) -> str:
        """Get brief explanation of recent reading based on line times and reading patterns.
        
        Args:
            page_line_times: Dictionary mapping page -> line_text -> reading_time
            current_page: Current page number
            pdf_document: PDF document instance to get line text and context
            max_lines: Maximum number of recent lines to analyze
        """
        try:
            # Collect all lines with timing data across all pages, ordered by time
            all_timed_lines = []
            
            for page_num, lines_dict in page_line_times.items():
                for line_text, reading_time in lines_dict.items():
                    # Only include lines that were actually read (have positive time)
                    if reading_time > 0.1:  # Filter out very brief glances
                        all_timed_lines.append({
                            'page': page_num,
                            'text': line_text,
                            'time': reading_time
                        })
            
            if not all_timed_lines:
                return "No reading data available yet. Start reading and try again."
                
            # Sort by reading time (descending) to prioritize lines that took longer to read
            all_timed_lines.sort(key=lambda x: x['time'], reverse=True)
            
            # Take the most recently read lines (the ones that took significant time)
            recent_lines = all_timed_lines[:max_lines]
            
            # Also get the last few lines in reading order (by page/position)
            # Get current position context
            context_lines = []
            if pdf_document and current_page is not None:
                try:
                    # Get recent context from current page
                    page_text = pdf_document.get_page_text(current_page)
                    if page_text:
                        # Split into lines and take last few
                        lines = [line.strip() for line in page_text.split('\n') if line.strip()]
                        context_lines = lines[-3:]  # Last 3 lines for context
                except Exception:
                    pass
            
            # Build the prompt with history context
            prompt_parts = []
            
            # Add conversation history if available
            history_context = self.get_history_context()
            if history_context:
                prompt_parts.append(history_context)
            
            if recent_lines:
                prompt_parts.append("I've been reading the following text. Here are the lines I spent the most time on:")
                for i, line_data in enumerate(recent_lines):
                    prompt_parts.append(f"{i+1}. (Page {line_data['page']+1}, {line_data['time']:.1f}s): \"{line_data['text'][:100]}{'...' if len(line_data['text']) > 100 else ''}\"")
            
            if context_lines:
                prompt_parts.append("\nHere's my current reading context:")
                for line in context_lines:
                    prompt_parts.append(f"• {line[:100]}{'...' if len(line) > 100 else ''}")
            
            prompt_parts.append("\nPlease provide a brief, helpful explanation focusing on:")
            prompt_parts.append("1. Key concepts or difficult ideas from the lines I spent time on")
            prompt_parts.append("2. How these concepts connect to the overall context")
            if history_context:
                prompt_parts.append("3. Any connections to previous topics we've discussed")
            prompt_parts.append("4. Any clarification that might help my understanding")
            prompt_parts.append("\nKeep the explanation concise and actionable (2-3 sentences).")
            
            prompt = "\n".join(prompt_parts)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful reading assistant that provides brief, focused explanations to help readers understand difficult passages. You maintain awareness of the reading session context and build upon previous interactions. Focus on clarity and practical insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            response = response.choices[0].message.content.strip()
            logging.info(f"Reading assistance: {response}")
            return response
            
        except Exception as e:
            print(f"Error getting reading assistance: {e}")
            return "Sorry, I couldn't generate reading assistance at this time." 