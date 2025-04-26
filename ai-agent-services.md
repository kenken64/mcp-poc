# Image Analysis and Speech Processing Services

These services enhance our mobile AI agent by providing multi-modal capabilities for processing images and speech input from users.

## 1. Image Analysis Service

The image analysis service processes images uploaded by users and extracts meaningful information that can be used by the LLM to provide more relevant responses.

### Core Architecture

```python
# image_analysis_service.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
import httpx
import os
import base64
import io
import time
import logging
from PIL import Image
import json

# Optional: Import vision libraries based on what's available
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models
class ImageAnalysisRequest(BaseModel):
    image: str  # Base64 encoded image
    analysis_type: Optional[List[str]] = ["all"]  # Options: "ocr", "objects", "faces", "general", "all"
    max_results: Optional[int] = 10

class ImageAnalysisResponse(BaseModel):
    analysis_id: str
    caption: Optional[str] = None
    objects: Optional[List[str]] = None
    text: Optional[str] = None
    faces: Optional[int] = None
    colors: Optional[List[Dict[str, Any]]] = None
    safe_search: Optional[Dict[str, str]] = None
    error: Optional[str] = None
    processing_time: float

# API keys for external services (use environment variables in production)
VISION_API_KEY = os.getenv("VISION_API_KEY")
VISION_API_ENDPOINT = os.getenv("VISION_API_ENDPOINT", "https://api.openai.com/v1/chat/completions")

def decode_image(base64_image: str):
    """Decode base64 image to a format usable by CV libraries"""
    try:
        # Remove potential prefixes like "data:image/jpeg;base64,"
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]
            
        image_bytes = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        raise ValueError(f"Invalid image data: {str(e)}")

async def analyze_with_vision_api(image_base64: str):
    """Analyze image using OpenAI's vision model"""
    try:
        # Prepare the request
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image and provide a detailed caption of what you see. Also list any key objects, text, people, and notable elements."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VISION_API_KEY}"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                VISION_API_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=30.0
            )
            
            if response.status_code != 200:
                logger.error(f"Vision API error: {response.text}")
                return {"error": f"Vision API error: {response.status_code}"}
                
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse the content to extract structured information
            analysis = {
                "caption": content.split("\n\n")[0] if "\n\n" in content else content
            }
            
            # Try to extract objects list if present
            if "Objects:" in content:
                objects_text = content.split("Objects:")[1].split("\n\n")[0]
                objects = [obj.strip().replace("- ", "") for obj in objects_text.split("\n") if obj.strip()]
                analysis["objects"] = objects
                
            # Try to extract text if present
            if "Text:" in content:
                text = content.split("Text:")[1].split("\n\n")[0]
                analysis["text"] = text.strip()
                
            return analysis
            
    except Exception as e:
        logger.error(f"Error in vision API analysis: {e}")
        return {"error": f"Vision analysis failed: {str(e)}"}

def extract_text_with_tesseract(image):
    """Extract text from image using Tesseract OCR"""
    if not TESSERACT_AVAILABLE:
        return "OCR not available"
        
    try:
        # Convert PIL image to cv2 format if needed
        if CV2_AVAILABLE:
            img_array = np.array(image)
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Preprocess for better OCR results
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            processed_img = Image.fromarray(binary)
        else:
            processed_img = image
            
        # Run OCR
        text = pytesseract.image_to_string(processed_img)
        return text.strip()
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return f"OCR failed: {str(e)}"

def detect_faces(image):
    """Detect and count faces in the image"""
    if not CV2_AVAILABLE:
        return 0
        
    try:
        # Convert PIL image to cv2 format
        img_array = np.array(image)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale and detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        return len(faces)
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return 0

def extract_dominant_colors(image, num_colors=5):
    """Extract dominant colors from the image"""
    if not CV2_AVAILABLE:
        return []
        
    try:
        # Convert PIL image to cv2 format
        img_array = np.array(image)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Reshape and convert to float
        pixels = img.reshape(-1, 3).astype(np.float32)
        
        # Define criteria and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Count labels to find dominant colors
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        sorted_counts = counts[sorted_indices]
        sorted_centers = centers[sorted_indices]
        
        # Convert to RGB and calculate percentages
        colors = []
        total_pixels = pixels.shape[0]
        
        for i, (center, count) in enumerate(zip(sorted_centers, sorted_counts)):
            b, g, r = center.astype(int)
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            percentage = (count / total_pixels) * 100
            
            colors.append({
                "color": hex_color,
                "rgb": [int(r), int(g), int(b)],
                "percentage": round(percentage, 2)
            })
            
        return colors
    except Exception as e:
        logger.error(f"Color extraction error: {e}")
        return []

@app.post("/analyze", response_model=ImageAnalysisResponse)
async def analyze_image(request: ImageAnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze image and extract information"""
    start_time = time.time()
    analysis_id = f"img_{int(start_time * 1000)}"
    
    try:
        # Decode image
        image = decode_image(request.image)
        
        # Initialize response
        response = {
            "analysis_id": analysis_id,
            "processing_time": 0
        }
        
        # Determine which analyses to run
        run_all = "all" in request.analysis_type
        run_ocr = run_all or "ocr" in request.analysis_type
        run_objects = run_all or "objects" in request.analysis_type
        run_faces = run_all or "faces" in request.analysis_type
        run_colors = run_all or "colors" in request.analysis_type
        run_general = run_all or "general" in request.analysis_type
        
        # Run the requested analyses
        
        # First check if we should use the Vision API
        if run_general or run_objects:
            vision_results = await analyze_with_vision_api(request.image)
            
            if "error" not in vision_results:
                if "caption" in vision_results:
                    response["caption"] = vision_results["caption"]
                    
                if "objects" in vision_results:
                    response["objects"] = vision_results["objects"]
                    
                if "text" in vision_results:
                    response["text"] = vision_results["text"]
            elif "error" in vision_results:
                # Fall back to local processing if available
                pass
        
        # Run OCR if needed and not already provided by Vision API
        if run_ocr and "text" not in response and TESSERACT_AVAILABLE:
            text = extract_text_with_tesseract(image)
            if text and text != "OCR not available" and not text.startswith("OCR failed"):
                response["text"] = text
        
        # Run face detection if needed
        if run_faces and CV2_AVAILABLE:
            face_count = detect_faces(image)
            response["faces"] = face_count
        
        # Extract colors if needed
        if run_colors and CV2_AVAILABLE:
            colors = extract_dominant_colors(image, num_colors=5)
            if colors:
                response["colors"] = colors
        
        # Calculate processing time
        processing_time = time.time() - start_time
        response["processing_time"] = round(processing_time, 3)
        
        return response
        
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        processing_time = time.time() - start_time
        
        return {
            "analysis_id": analysis_id,
            "error": f"Analysis failed: {str(e)}",
            "processing_time": round(processing_time, 3)
        }

@app.get("/status")
async def service_status():
    """Get service status and capabilities"""
    return {
        "status": "operational",
        "capabilities": {
            "vision_api": bool(VISION_API_KEY),
            "tesseract_ocr": TESSERACT_AVAILABLE,
            "cv2": CV2_AVAILABLE
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Determine the port
    port = int(os.getenv("PORT", "8003"))
    
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### Integration with Context Protocol Server

To integrate the Image Analysis Service with the Context Protocol Server, we need to update the main server to call this service when image messages are received:

```python
# In context_protocol_server.py

async def analyze_image(image_base64: str):
    """Send image to analysis service and return results"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://image-analysis-service:8003/analyze",
                json={"image": image_base64},
                timeout=60.0
            )
            
            if response.status_code != 200:
                logger.error(f"Image analysis service error: {response.status_code}, {response.text}")
                return {"error": f"Image analysis failed with status {response.status_code}"}
                
            return response.json()
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return {"error": str(e)}
```

## 2. Speech Processing Service

The speech processing service handles audio transcription and text-to-speech capabilities for the mobile AI agent.

### Core Architecture

```python
# transcription_service.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
import httpx
import os
import base64
import io
import time
import logging
import tempfile
import uuid
import json

# Optional: Import speech libraries based on what's available
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models
class TranscriptionRequest(BaseModel):
    audio: str  # Base64 encoded audio
    language: Optional[str] = "en-US"
    model: Optional[str] = "default"

class TranscriptionResponse(BaseModel):
    transcription_id: str
    text: str
    confidence: Optional[float] = None
    language: str
    error: Optional[str] = None
    processing_time: float

class TextToSpeechRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"
    language: Optional[str] = "en-US"
    speed: Optional[float] = 1.0
    pitch: Optional[float] = 1.0

class TextToSpeechResponse(BaseModel):
    audio_id: str
    audio: str  # Base64 encoded audio
    duration: Optional[float] = None
    error: Optional[str] = None
    processing_time: float

# API keys for external services (use environment variables in production)
SPEECH_API_KEY = os.getenv("SPEECH_API_KEY")
SPEECH_API_ENDPOINT = os.getenv("SPEECH_API_ENDPOINT")

def decode_audio(base64_audio: str):
    """Decode base64 audio to a format usable by speech libraries"""
    try:
        # Remove potential prefixes
        if "," in base64_audio:
            base64_audio = base64_audio.split(",")[1]
            
        audio_bytes = base64.b64decode(base64_audio)
        return audio_bytes
    except Exception as e:
        logger.error(f"Error decoding audio: {e}")
        raise ValueError(f"Invalid audio data: {str(e)}")

async def transcribe_with_api(audio_bytes: bytes, language: str):
    """Transcribe audio using cloud speech API"""
    if not SPEECH_API_KEY or not SPEECH_API_ENDPOINT:
        return {"error": "Speech API not configured"}
        
    try:
        # Prepare request
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        payload = {
            "audio": audio_b64,
            "language": language,
            "model": "whisper-large-v3"
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {SPEECH_API_KEY}"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                SPEECH_API_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=30.0
            )
            
            if response.status_code != 200:
                logger.error(f"Speech API error: {response.text}")
                return {"error": f"Speech API error: {response.status_code}"}
                
            result = response.json()
            return {
                "text": result.get("text", ""),
                "confidence": result.get("confidence", 0.0)
            }
    except Exception as e:
        logger.error(f"Error in API transcription: {e}")
        return {"error": f"API transcription failed: {str(e)}"}

def transcribe_with_local(audio_bytes: bytes, language: str):
    """Transcribe audio using local speech recognition"""
    if not SR_AVAILABLE or not PYDUB_AVAILABLE:
        return {"error": "Local speech recognition not available"}
        
    try:
        # Create temporary file for the audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            temp_audio.write(audio_bytes)
            
        # Convert to proper format if necessary
        if not temp_audio_path.lower().endswith(".wav"):
            try:
                # Try to detect format based on first few bytes
                audio = AudioSegment.from_file(temp_audio_path)
                audio.export(temp_audio_path, format="wav")
            except Exception as format_error:
                logger.error(f"Format conversion error: {format_error}")
                # Try common formats
                try:
                    audio = AudioSegment.from_mp3(temp_audio_path)
                    audio.export(temp_audio_path, format="wav")
                except:
                    try:
                        audio = AudioSegment.from_ogg(temp_audio_path)
                        audio.export(temp_audio_path, format="wav")
                    except Exception as e:
                        return {"error": f"Unsupported audio format: {str(e)}"}
        
        # Initialize recognizer
        r = sr.Recognizer()
        
        # Load audio file
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = r.record(source)
            
        # Perform speech recognition
        if language.startswith("en"):
            text = r.recognize_google(audio_data, language=language)
        else:
            # Use more general recognizer for other languages
            text = r.recognize_google(audio_data, language=language)
            
        # Clean up temporary file
        os.unlink(temp_audio_path)
        
        return {
            "text": text,
            "confidence": 0.8  # Estimated confidence
        }
    except sr.UnknownValueError:
        return {"error": "Speech could not be understood"}
    except sr.RequestError as e:
        return {"error": f"Recognition service error: {str(e)}"}
    except Exception as e:
        logger.error(f"Local transcription error: {e}")
        return {"error": f"Transcription failed: {str(e)}"}

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    """Transcribe audio to text"""
    start_time = time.time()
    transcription_id = f"tr_{int(start_time * 1000)}"
    
    try:
        # Decode audio
        audio_bytes = decode_audio(request.audio)
        
        # Try cloud API first if configured
        if SPEECH_API_KEY and SPEECH_API_ENDPOINT:
            result = await transcribe_with_api(audio_bytes, request.language)
            
            if "error" not in result:
                processing_time = time.time() - start_time
                return {
                    "transcription_id": transcription_id,
                    "text": result["text"],
                    "confidence": result.get("confidence"),
                    "language": request.language,
                    "processing_time": round(processing_time, 3)
                }
        
        # Fall back to local transcription
        if SR_AVAILABLE and PYDUB_AVAILABLE:
            result = transcribe_with_local(audio_bytes, request.language)
            
            if "error" not in result:
                processing_time = time.time() - start_time
                return {
                    "transcription_id": transcription_id,
                    "text": result["text"],
                    "confidence": result.get("confidence"),
                    "language": request.language,
                    "processing_time": round(processing_time, 3)
                }
            else:
                raise ValueError(result["error"])
        else:
            raise ValueError("No transcription service available")
            
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        processing_time = time.time() - start_time
        
        return {
            "transcription_id": transcription_id,
            "text": "",
            "language": request.language,
            "error": f"Transcription failed: {str(e)}",
            "processing_time": round(processing_time, 3)
        }

@app.post("/text-to-speech", response_model=TextToSpeechResponse)
async def text_to_speech(request: TextToSpeechRequest):
    """Convert text to speech"""
    start_time = time.time()
    audio_id = f"tts_{int(start_time * 1000)}"
    
    # This is a placeholder - implement actual TTS using a library like gTTS, pyttsx3, 
    # or cloud services like Google Cloud TTS, Amazon Polly, etc.
    processing_time = time.time() - start_time
    
    return {
        "audio_id": audio_id,
        "audio": "",  # Would contain base64 encoded audio
        "error": "TTS not implemented yet",
        "processing_time": round(processing_time, 3)
    }

@app.get("/status")
async def service_status():
    """Get service status and capabilities"""
    return {
        "status": "operational",
        "capabilities": {
            "speech_api": bool(SPEECH_API_KEY and SPEECH_API_ENDPOINT),
            "speech_recognition": SR_AVAILABLE,
            "audio_processing": PYDUB_AVAILABLE
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Determine the port
    port = int(os.getenv("PORT", "8004"))
    
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### Integration with Context Protocol Server

To integrate the Speech Processing Service with the Context Protocol Server, we need to update the main server to call this service when audio messages are received:

```python
# In context_protocol_server.py

async def transcribe_audio(audio_base64: str):
    """Send audio to transcription service and return text"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://transcription-service:8004/transcribe",
                json={"audio": audio_base64},
                timeout=60.0
            )
            
            if response.status_code != 200:
                logger.error(f"Transcription service error: {response.status_code}, {response.text}")
                return "Sorry, I couldn't transcribe the audio."
                
            result = response.json()
            
            if "error" in result and result["error"]:
                logger.error(f"Transcription error: {result['error']}")
                return "Sorry, there was an error transcribing the audio."
                
            return result.get("text", "")
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return "Sorry, I couldn't transcribe the audio."
```

## 3. Docker Configuration for These Services

```dockerfile
# Dockerfile.image-service
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV and Tesseract
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY image_analysis_service.py .

CMD ["python", "image_analysis_service.py"]
```

```dockerfile
# Dockerfile.transcription-service
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    libpulse0 \
    libasound2-dev \
    portaudio19-dev \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY transcription_service.py .

CMD ["python", "transcription_service.py"]
```

## 4. Handling Multi-Modal Interactions in the Mobile Client

To fully utilize these services, the mobile client needs to handle multi-modal interactions effectively. Here are examples of how to implement image and audio handling in the React Native app:

### Image Handling in React Native

```javascript
// components/ImageInputHandler.js
import React, { useState } from 'react';
import { View, TouchableOpacity, Image, StyleSheet, ActivityIndicator, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import { Ionicons } from '@expo/vector-icons';
import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';

const ImageInputHandler = ({ onImageCaptured, isLoading }) => {
  const [image, setImage] = useState(null);

  const requestPermissions = async () => {
    const { status: cameraStatus } = await ImagePicker.requestCameraPermissionsAsync();
    const { status: libraryStatus } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    
    if (cameraStatus !== 'granted' || libraryStatus !== 'granted') {
      Alert.alert(
        'Permissions Required',
        'We need camera and photo library permissions to send images.',
        [{ text: 'OK' }]
      );
      return false;
    }
    return true;
  };

  const takePhoto = async () => {
    const hasPermissions = await requestPermissions();
    if (!hasPermissions) return;
    
    try {
      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });
      
      if (!result.cancelled && result.uri) {
        handleSelectedImage(result.uri);
      }
    } catch (error) {
      console.error('Error taking photo:', error);
      Alert.alert('Error', 'Failed to take photo. Please try again.');
    }
  };

  const pickImage = async () => {
    const hasPermissions = await requestPermissions();
    if (!hasPermissions) return;
    
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });
      
      if (!result.cancelled && result.uri) {
        handleSelectedImage(result.uri);
      }
    } catch (error) {
      console.error('Error picking image:', error);
      Alert.alert('Error', 'Failed to select image. Please try again.');
    }
  };

  const handleSelectedImage = async (uri) => {
    try {
      // Display the selected image
      setImage(uri);
      
      // Process the image (compress if needed)
      const fileInfo = await FileSystem.getInfoAsync(uri);
      let processedUri = uri;
      
      // Compress image if larger than 1MB
      if (fileInfo.size > 1024 * 1024) {
        const compressedImage = await manipulateAsync(
          uri,
          [{ resize: { width: 1200 } }], // Resize to reasonable dimensions
          { compress: 0.7, format: SaveFormat.JPEG }
        );
        processedUri = compressedImage.uri;
      }
      
      // Convert to base64
      const base64 = await FileSystem.readAsStringAsync(processedUri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      
      // Call the callback with the base64 image data
      if (onImageCaptured) {
        onImageCaptured(base64);
      }
    } catch (error) {
      console.error('Error processing image:', error);
      Alert.alert('Error', 'Failed to process image. Please try again.');
    }
  };

  return (
    <View style={styles.container}>
      {image && (
        <View style={styles.previewContainer}>
          <Image source={{ uri: image }} style={styles.preview} />
          {isLoading && (
            <View style={styles.loadingOverlay}>
              <ActivityIndicator size="large" color="#ffffff" />
            </View>
          )}
        </View>
      )}
      
      <View style={styles.buttonContainer}>
        <TouchableOpacity 
          style={styles.button} 
          onPress={takePhoto}
          disabled={isLoading}
        >
          <Ionicons name="camera" size={24} color={isLoading ? "#999" : "#007AFF"} />
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={styles.button} 
          onPress={pickImage}
          disabled={isLoading}
        >
          <Ionicons name="images" size={24} color={isLoading ? "#999" : "#007AFF"} />
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginVertical: 10,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 10,
  },
  button: {
    padding: 12,
    borderRadius: 30,
    backgroundColor: '#f0f0f0',
  },
  previewContainer: {
    position: 'relative',
    alignItems: 'center',
    marginBottom: 10,
  },
  preview: {
    width: 200,
    height: 200,
    borderRadius: 10,
  },
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 10,
  },
});

export default ImageInputHandler;
```

### Audio Recording and Processing in React Native

```javascript
// components/AudioInputHandler.js
import React, { useState, useEffect } from 'react';
import { View, TouchableOpacity, Text, StyleSheet, ActivityIndicator } from 'react-native';
import { Audio } from 'expo-av';
import * as FileSystem from 'expo-file-system';
import { Ionicons } from '@expo/vector-icons';

const AudioInputHandler = ({ onAudioRecorded, isLoading }) => {
  const [recording, setRecording] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [permissionStatus, setPermissionStatus] = useState(null);
  
  useEffect(() => {
    let interval;
    
    if (isRecording) {
      interval = setInterval(() => {
        setRecordingDuration(prev => prev + 1);
      }, 1000);
    } else {
      setRecordingDuration(0);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isRecording]);
  
  useEffect(() => {
    (async () => {
      const { status } = await Audio.requestPermissionsAsync();
      setPermissionStatus(status);
    })();
  }, []);
  
  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };
  
  const startRecording = async () => {
    try {
      // Check permissions again
      if (permissionStatus !== 'granted') {
        const { status } = await Audio.requestPermissionsAsync();
        setPermissionStatus(status);
        if (status !== 'granted') {
          alert('We need microphone permissions to record audio.');
          return;
        }
      }
      
      // Set audio mode
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        shouldDuckAndroid: true,
        interruptionModeAndroid: Audio.INTERRUPTION_MODE_ANDROID_DO_NOT_MIX,
        playThroughEarpieceAndroid: false,
      });
      
      // Start recording
      const { recording } = await Audio.Recording.createAsync(
        Audio.RECORDING_OPTIONS_PRESET_HIGH_QUALITY
      );
      
      setRecording(recording);
      setIsRecording(true);
    } catch (error) {
      console.error('Failed to start recording', error);
      alert('Failed to start recording. Please try again.');
    }
  };
  
  const stopRecording = async () => {
    if (!recording) return;
    
    setIsRecording(false);
    
    try {
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      setRecording(null);
      
      // Convert to base64
      const base64Audio = await FileSystem.readAsStringAsync(uri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      
      // Call the callback with the base64 audio data
      if (onAudioRecorded) {
        onAudioRecorded(base64Audio);
      }
    } catch (error) {
      console.error('Failed to stop recording:', error);
      alert('Failed to process recording. Please try again.');
    }
  };
  
  if (permissionStatus !== 'granted') {
    return (
      <View style={styles.container}>
        <Text style={styles.permissionText}>
          Microphone permission is required to record audio.
        </Text>
        <TouchableOpacity 
          style={styles.permissionButton}
          onPress={async () => {
            const { status } = await Audio.requestPermissionsAsync();
            setPermissionStatus(status);
          }}
        >
          <Text style={styles.permissionButtonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }
  
  return (
    <View style={styles.container}>
      {isRecording ? (
        <View style={styles.recordingContainer}>
          <Text style={styles.recordingText}>
            Recording... {formatDuration(recordingDuration)}
          </Text>
          <TouchableOpacity 
            style={[styles.recordButton, styles.recordingActive]}
            onPress={stopRecording}
          >
            <Ionicons name="square" size={24} color="#ffffff" />
          </TouchableOpacity>
        </View>
      ) : (
        <TouchableOpacity 
          style={styles.recordButton}
          onPress={startRecording}
          disabled={isLoading}
        >
          <Ionicons 
            name="mic" 
            size={24} 
            color={isLoading ? "#999999" : "#ffffff"} 
          />
        </TouchableOpacity>
      )}
      
      {isLoading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="small" color="#007AFF" />
          <Text style={styles.loadingText}>Processing audio...</Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    marginVertical: 10,
  },
  recordButton: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#007AFF',
    justifyContent: 'center',
    alignItems: 'center',
  },
  recordingActive: {
    backgroundColor: '#FF3B30',
  },
  recordingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    width: '80%',
  },
  recordingText: {
    color: '#FF3B30',
    fontWeight: 'bold',
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 10,
  },
  loadingText: {
    marginLeft: 10,
    color: '#666666',
  },
  permissionText: {
    marginBottom: 10,
    color: '#666666',
    textAlign: 'center',
  },
  permissionButton: {
    padding: 10,
    backgroundColor: '#007AFF',
    borderRadius: 5,
  },
  permissionButtonText: {
    color: '#ffffff',
  },
});

export default AudioInputHandler;
```

## 5. Security Considerations for Multi-Modal Interactions

When implementing multi-modal interactions, special consideration should be given to security:

1. **Data Privacy**:
   - Implement client-side image and audio processing to minimize data transfer
   - Clear cached files after processing
   - Implement clear data retention policies

2. **Content Moderation**:
   - Consider implementing client-side content filtering
   - Add server-side image moderation to detect inappropriate content
   - Limit maximum audio duration to prevent abuse

3. **Authentication for Media Upload**:
   - Verify user authentication before accepting media uploads
   - Implement rate limiting for media uploads
   - Set reasonable size limits for uploads

4. **Secure Media Storage**:
   - Encrypt sensitive media at rest
   - Implement secure media file cleanup
   - Consider privacy-preserving approaches like federated learning

By implementing these services and properly integrating them with our mobile AI agent architecture, we can create a powerful multi-modal experience that allows users to interact naturally through text, images, and voice.