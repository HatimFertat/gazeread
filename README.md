# ReadGaze

An intelligent PDF reader that tracks your eye movements to enhance your reading experience. ReadGaze combines eye tracking technology with AI assistance to help you understand difficult passages and maintain optimal reading habits.

## Features

- Eye tracking integration for reading progress monitoring
- PDF document viewing with large, readable text
- Automatic scrolling based on eye position
- AI-powered assistance for difficult passages
- Reading time tracking and analysis

## Requirements

- Python 3.9 or higher
- Tobii Eye Tracker (compatible with tobii-research library)
- OpenAI API key (for AI assistance)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/readgaze.git
cd readgaze
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Connect your Tobii Eye Tracker
2. Run the application:
```bash
python -m readgaze
```
3. Open a PDF file through the application interface
4. Start reading - the application will track your progress and provide assistance when needed

## Project Structure

- `readgaze/` - Main package directory
  - `gui/` - GUI components and main window
  - `eye_tracking/` - Eye tracking integration
  - `pdf/` - PDF handling and text extraction
  - `ai/` - AI assistance integration
  - `utils/` - Utility functions and helpers

## License

MIT License
