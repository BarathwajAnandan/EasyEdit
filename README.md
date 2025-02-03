# Easy EdIT

**Easy EdIT** is an interactive image processing and analysis platform that lets you edit images using natural language. Built with Python, it combines Streamlit's user interface with OpenCV and NumPy for image manipulation, powered by AI agents that understand your intent and execute the appropriate operations.

## Why Easy EdIT?

- 🎯 **Simplicity First** - Edit images without being an OpenCV expert. Just describe what you want to do in plain English.
- 🔒 **Privacy Focused** - Process images locally instead of uploading to multiple online services.
- 📖 **Documentation Made Easy** - Bypass the complexity of OpenCV/NumPy documentation by using natural language queries.
- 🌟 **Open Source** - Free, transparent, and community-driven development.

## Key Features

### Natural Language Image Processing
- Edit images by describing what you want in plain English
- Intelligent AI agents translate your requests into precise OpenCV/NumPy operations
- Version control system for your edits (similar to git)

### Interactive Interface
- Real-time image preview and side-by-side comparison
- Click-to-get coordinates for precise editing
- Detailed processing history with undo capabilities
- Download processed images in PNG, JPG, or PDF formats

### Supported Operations
- Current supports most one line edits. 
- **Image Editing**: Resize, blur, draw shapes, and more
- **Image Analysis**: Get dimensions, pixel values, and other properties
- Full list of tested operations in ( Need help to add more operations and test them):
  - `functions.txt` (editing operations)
  - `analysis_functions.txt` (analysis operations)

**Note:** Currently, multi-line code edits, such as Hough Circle detection, are not supported.
## Getting Started

### Prerequisites
- Python 3.7+
- Git

### Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/BarathwajAnandan/EasyEdit.git
    cd EasyEdit
    ```

2. **Set Up Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Configure API Keys**

   Create a `.env` file with your API keys:
   ```ini
   GROQ_API_KEY=your_groq_api_key_here    # from console.groq.com
   SNOVA_API_KEY=your_snova_api_key_here  # from cloud.sambanova.ai/apis
   ```

   Get your API keys from:
   - [Groq Console](https://console.groq.com/api-keys)
   - [Sambanova Cloud](https://cloud.sambanova.ai/apis)

4. **Launch the App**

    Start the app from the project root with:

    ```bash
    streamlit run swarm_ui.py
    ```

    This will launch the Easy EdIT interface in your default web browser and local host automatically.

## How It Works

Easy EdIT uses a pipeline of specialized AI agents to:
1. Parse your natural language query
2. Determine the appropriate image processing operations
3. Execute the operations using OpenCV/NumPy
4. Provide human-friendly feedback

## Project Structure

```
.
├── swarm/
│   ├── swarm_ui.py          # Streamlit-based UI for image upload, display, and query interaction
│   ├── clients.py           # API key configuration and provider selection
│   ├── main.py              # Core image processing logic and integration with AI agents
│   └── agents.py            # Definitions of agents for query parsing, function parameter construction, and result interpretation
├── .env                     # Environment configuration file (API keys, etc.)
```

## Contributing

We welcome contributions! Whether it's adding new features, improving documentation, or reporting bugs, please feel free to:
- Open an issue
- Submit a pull request
- Help test and improve our AI agents

## License

[MIT License](LICENSE)

## Acknowledgements

Built with:
- [OpenAI Swarm](https://github.com/openai/swarm)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [Sambanova](https://cloud.sambanova.ai/) for LLM support
- [Groq](https://console.groq.com/) for LLM support

This project leverages the power of [OpenAI Swarm](https://github.com/openai/swarm), [Streamlit](https://streamlit.io/), [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/), [Sambanova](https://cloud.sambanova.ai/), [Groq](https://console.groq.com/), and AI agents for an interactive and intelligent image processing experience. 