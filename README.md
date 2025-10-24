# Voice AI Agent with KaniTTS

A real-time voice AI assistant built with LiveKit Agents framework, featuring speech-to-text, language processing, and text-to-speech capabilities.

## Features

- **Speech-to-Text**: Deepgram STT with Flux General EN model
- **Language Model**: OpenAI GPT-4o-mini for natural conversation
- **Text-to-Speech**: OpenAI-compatible server using KaniTTS (https://github.com/nineninesix-ai/kanitts-vllm)
- **Voice Activity Detection**: Silero VAD for accurate speech detection
- **Turn Detection**: Multilingual turn detection for natural conversations
- **Noise Cancellation**: Background voice cancellation (BVC) for clear audio

## Prerequisites

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) package manager
- LiveKit Cloud account ([sign up for free](https://cloud.livekit.io/))
- API keys for:
  - LiveKit (API key, secret, and URL)
  - Deepgram (for STT)
  - OpenAI (for LLM)

## Installation

### 1. Install LiveKit CLI

**macOS:**
```bash
brew install livekit-cli
```

**Linux:**
```bash
curl -sSL https://get.livekit.io/cli | bash
```

**Windows:**
```bash
winget install LiveKit.LiveKitCLI
```

Then authenticate with LiveKit Cloud:
```bash
lk cloud auth
```

### 2. Clone and Setup

```bash
git clone git@github.com:nineninesix-ai/livekit-agent.git
cd livekit-agent
```

### 3. Install Dependencies

```bash
uv sync
```

This will install all required dependencies from [pyproject.toml](pyproject.toml):
- `livekit-agents` with Deepgram, OpenAI, Silero, and turn-detector plugins
- `livekit-plugins-noise-cancellation` for audio processing
- `python-dotenv` for environment variable management

### 4. Configure Environment Variables

Create a `.env.local` file in the project root with the following variables:

```env
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
LIVEKIT_URL=wss://your-project.livekit.cloud
DEEPGRAM_API_KEY=your_deepgram_api_key
OPENAI_API_KEY=your_openai_api_key
```

**Quick setup with LiveKit CLI:**
```bash
lk app env -w
```

This automatically generates LiveKit credentials in `.env.local`.

### 5. Download Model Files

Download required model files (VAD, turn detection, etc.):

```bash
uv run agent.py download-files
```

## Usage

### Development Mode

Run the agent in development mode (connects to LiveKit Cloud):

```bash
uv run agent.py dev
```

This starts the agent and connects it to your LiveKit server. You can interact with it via:
- [LiveKit Agents Playground](https://cloud.livekit.io/projects/p_/agents)
- Your own web/mobile frontend
- Telephony integration

### Console Mode

Test the agent locally in your terminal without LiveKit connection:

```bash
uv run agent.py console
```

### Production Mode

Run the agent in production:

```bash
uv run agent.py start
```

## Configuration

### Custom TTS Server

This project uses a custom TTS server instead of the default OpenAI TTS. The configuration in [agent.py:31-36](agent.py#L31-L36) points to `http://localhost:8000/v1`.

**To use the local TTS server:**
1. Ensure your TTS server is running on port 8000
2. The server should be OpenAI-compatible (accepts same API format)

Check our FastAPI implementation here: https://github.com/nineninesix-ai/kanitts-vllm

### Assistant Instructions

Customize the AI assistant's behavior by modifying the instructions in [agent.py:15-18](agent.py#L15-L18):

```python
instructions="""You are a helpful voice AI assistant.
Your responses are concise, to the point, and without any complex formatting..."""
```

### Audio Models Configuration

Current configuration:
- **STT**: Deepgram Flux General EN with 0.4s eager end-of-turn threshold
- **LLM**: GPT-4o-mini
- **VAD**: Silero VAD
- **Turn Detection**: Multilingual model
- **Noise Cancellation**: BVC (recommended for telephony)

## Project Structure

```
livekit-voice-agent/
├── agent.py              # Main agent implementation
├── pyproject.toml        # Python dependencies
├── uv.lock              # Dependency lock file
├── .env.local           # Environment variables (not committed)
└── README.md            # This file
```

## Telephony Applications

For telephony use cases, the agent uses BVC (Background Voice Cancellation) noise cancellation. For even better results with telephone audio, consider using `BVCTelephony`:

```python
room_input_options=RoomInputOptions(
    noise_cancellation=noise_cancellation.BVCTelephony(),
)
```

## Resources

- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [LiveKit Cloud Console](https://cloud.livekit.io/)
- [Deepgram Documentation](https://developers.deepgram.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)

## Troubleshooting

**Agent won't start:**
- Verify all environment variables are set correctly in `.env.local`
- Ensure model files are downloaded: `uv run agent.py download-files`
- Check that your API keys are valid

**No audio in/out:**
- Verify your microphone/speaker permissions
- Check LiveKit room configuration
- Ensure noise cancellation is properly configured

**TTS not working:**
- Ensure your local TTS server is running on port 8000
- Check server logs for errors
- Verify the server implements OpenAI-compatible API

## License

Apache 2. See [LICENSE](LICENSE) file for details.
