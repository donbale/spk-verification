from dotenv import load_dotenv
from stt import WhisperSTT

# Load environment variables from .env file
load_dotenv()

# Initialize and start listening
wSTT = WhisperSTT()
wSTT.listen()
