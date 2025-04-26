# Mobile App AI Agent with Model Context Protocol Server

## System Architecture Overview

```
┌──────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                  │     │                 │     │                 │
│  Mobile Client   │────▶│  Context        │────▶│  LLM Server     │
│  Application     │     │  Protocol       │     │                 │
└──────────────────┘     │  Server         │     └────────┬────────┘
                         │                 │              │
                         └─────────┬───────┘              │
                                   │                      │
                         ┌─────────▼───────┐     ┌────────▼────────┐
                         │                 │     │                 │
                         │  Context        │◀───▶│  Agent          │
                         │  Management     │     │  Orchestrator   │
                         │                 │     │                 │
                         └─────────────────┘     └─────────────────┘
```

## 1. Mobile Client Application

Let's design a mobile application that interacts with our AI agent system. We'll create a cross-platform solution using React Native.

### Mobile App Features

- Real-time chat interface with the AI agent
- Voice input for natural interaction
- Support for sharing images, documents, and location data
- Offline capability with syncing when connection is restored
- Secure authentication and data encryption
- Push notifications for agent updates

### Mobile App Implementation

```javascript
// App.js - Main entry point for the React Native app
import React, { useEffect, useState } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { AppState, Platform } from 'react-native';

// Import screens
import LoginScreen from './screens/LoginScreen';
import ChatScreen from './screens/ChatScreen';
import SettingsScreen from './screens/SettingsScreen';
import OnboardingScreen from './screens/OnboardingScreen';

// Import services
import { initializeSession, reconnectSession } from './services/sessionService';
import { registerPushNotifications } from './services/notificationService';
import { initializeSecureStorage } from './services/secureStorageService';

const Stack = createStackNavigator();

export default function App() {
  const [isReady, setIsReady] = useState(false);
  const [initialRoute, setInitialRoute] = useState('Login');
  const [sessionToken, setSessionToken] = useState(null);
  
  useEffect(() => {
    // Initialize app essentials
    const setupApp = async () => {
      // Initialize secure storage
      await initializeSecureStorage();
      
      // Check for existing session
      const token = await AsyncStorage.getItem('sessionToken');
      if (token) {
        // Validate token and reconnect to session
        try {
          const sessionId = await reconnectSession(token);
          if (sessionId) {
            setSessionToken(sessionId);
            setInitialRoute('Chat');
          }
        } catch (error) {
          console.log('Session expired or invalid', error);
        }
      }
      
      // Check if user has completed onboarding
      const onboardingComplete = await AsyncStorage.getItem('onboardingComplete');
      if (!onboardingComplete && !token) {
        setInitialRoute('Onboarding');
      }
      
      // Register for push notifications
      if (Platform.OS !== 'web') {
        await registerPushNotifications();
      }
      
      setIsReady(true);
    };
    
    setupApp();
    
    // Handle app state changes (background, foreground)
    const subscription = AppState.addEventListener('change', nextAppState => {
      if (nextAppState === 'active' && sessionToken) {
        // App came to foreground, refresh connection
        reconnectSession(sessionToken).catch(console.error);
      }
    });
    
    return () => {
      subscription.remove();
    };
  }, [sessionToken]);
  
  if (!isReady) {
    // Return a loading screen here
    return null;
  }
  
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName={initialRoute}>
        <Stack.Screen name="Onboarding" component={OnboardingScreen} options={{ headerShown: false }} />
        <Stack.Screen name="Login" component={LoginScreen} options={{ headerShown: false }} />
        <Stack.Screen name="Chat" component={ChatScreen} options={{ title: 'AI Assistant' }} />
        <Stack.Screen name="Settings" component={SettingsScreen} options={{ title: 'Settings' }} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
```

### Chat Interface Implementation

```javascript
// screens/ChatScreen.js
import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  FlatList,
  TouchableOpacity,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Image,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system';
import { Audio } from 'expo-av';
import { Ionicons } from '@expo/vector-icons';

// Import services
import { sendMessage, connectToAgentStream } from '../services/chatService';
import { uploadAttachment } from '../services/fileService';

const ChatScreen = ({ navigation, route }) => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [recording, setRecording] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  
  const ws = useRef(null);
  const flatListRef = useRef(null);
  const sessionId = route.params?.sessionId;
  
  useEffect(() => {
    // Connect to WebSocket when component mounts
    if (sessionId) {
      ws.current = connectToAgentStream(sessionId);
      
      ws.current.onopen = () => {
        console.log('WebSocket connection established');
      };
      
      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleAgentResponse(data);
      };
      
      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
      
      ws.current.onclose = () => {
        console.log('WebSocket connection closed');
      };
    }
    
    // Add right header button for settings
    navigation.setOptions({
      headerRight: () => (
        <TouchableOpacity 
          onPress={() => navigation.navigate('Settings')}
          style={{ marginRight: 15 }}
        >
          <Ionicons name="settings-outline" size={24} color="#007AFF" />
        </TouchableOpacity>
      ),
    });
    
    // Request microphone permissions
    (async () => {
      const { status } = await Audio.requestPermissionsAsync();
      if (status !== 'granted') {
        alert('Sorry, we need microphone permissions to make this work!');
      }
    })();
    
    // Clean up WebSocket connection
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [sessionId, navigation]);
  
  const handleAgentResponse = (data) => {
    if (data.type === 'message') {
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          id: Date.now().toString(),
          text: data.content,
          isUser: false,
          timestamp: new Date(),
        },
      ]);
      setIsLoading(false);
    } else if (data.type === 'typing') {
      // Handle typing indicator
    } else if (data.type === 'attachment') {
      // Handle attachment from agent
    }
  };
  
  const handleSendMessage = async () => {
    if (!inputText.trim() && !isRecording) return;
    
    const newMessage = {
      id: Date.now().toString(),
      text: inputText,
      isUser: true,
      timestamp: new Date(),
    };
    
    setMessages((prevMessages) => [...prevMessages, newMessage]);
    setInputText('');
    setIsLoading(true);
    
    try {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        sendMessage(ws.current, {
          type: 'message',
          content: inputText,
          sessionId,
        });
      } else {
        console.error('WebSocket not connected');
        setIsLoading(false);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setIsLoading(false);
    }
  };
  
  const startRecording = async () => {
    try {
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });
      
      const { recording } = await Audio.Recording.createAsync(
        Audio.RECORDING_OPTIONS_PRESET_HIGH_QUALITY
      );
      
      setRecording(recording);
      setIsRecording(true);
    } catch (error) {
      console.error('Failed to start recording', error);
    }
  };
  
  const stopRecording = async () => {
    setIsRecording(false);
    setIsLoading(true);
    
    try {
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      setRecording(null);
      
      // Upload audio file and transcribe
      const audioData = await FileSystem.readAsStringAsync(uri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        sendMessage(ws.current, {
          type: 'audio',
          content: audioData,
          sessionId,
        });
      }
    } catch (error) {
      console.error('Failed to stop recording', error);
      setIsLoading(false);
    }
  };
  
  const handlePickImage = async () => {
    // Request permission first
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      alert('Sorry, we need camera roll permissions to make this work!');
      return;
    }
    
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        quality: 0.8,
        base64: true,
      });
      
      if (!result.cancelled && result.base64) {
        // Send image to context server
        setIsLoading(true);
        
        const imageMessage = {
          id: Date.now().toString(),
          isUser: true,
          timestamp: new Date(),
          isImage: true,
          uri: result.uri,
        };
        
        setMessages((prevMessages) => [...prevMessages, imageMessage]);
        
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
          sendMessage(ws.current, {
            type: 'image',
            content: result.base64,
            sessionId,
          });
        }
      }
    } catch (error) {
      console.error('Error picking image:', error);
    }
  };
  
  const renderMessage = ({ item }) => {
    return (
      <View style={[
        styles.messageBubble,
        item.isUser ? styles.userBubble : styles.agentBubble,
      ]}>
        {item.isImage ? (
          <Image source={{ uri: item.uri }} style={styles.messageImage} />
        ) : (
          <Text style={styles.messageText}>{item.text}</Text>
        )}
      </View>
    );
  };
  
  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
    >
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderMessage}
        keyExtractor={(item) => item.id}
        style={styles.messagesList}
        contentContainerStyle={styles.messagesContent}
        onContentSizeChange={() => flatListRef.current.scrollToEnd({ animated: true })}
      />
      
      {isLoading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="small" color="#007AFF" />
          <Text style={styles.loadingText}>AI is thinking...</Text>
        </View>
      )}
      
      <View style={styles.inputContainer}>
        <TouchableOpacity style={styles.attachButton} onPress={handlePickImage}>
          <Ionicons name="image-outline" size={24} color="#007AFF" />
        </TouchableOpacity>
        
        <TextInput
          style={styles.input}
          value={inputText}
          onChangeText={setInputText}
          placeholder="Type a message..."
          returnKeyType="send"
          onSubmitEditing={handleSendMessage}
          editable={!isRecording}
        />
        
        {!isRecording ? (
          <TouchableOpacity 
            style={styles.micButton} 
            onPress={startRecording}
            onLongPress={startRecording}
          >
            <Ionicons name="mic-outline" size={24} color="#007AFF" />
          </TouchableOpacity>
        ) : (
          <TouchableOpacity 
            style={[styles.micButton, styles.recordingButton]} 
            onPress={stopRecording}
          >
            <Ionicons name="stop" size={24} color="#FF3B30" />
          </TouchableOpacity>
        )}
        
        <TouchableOpacity style={styles.sendButton} onPress={handleSendMessage}>
          <Ionicons name="send" size={24} color="#007AFF" />
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F6F6F6',
  },
  messagesList: {
    flex: 1,
  },
  messagesContent: {
    paddingVertical: 10,
    paddingHorizontal: 15,
  },
  messageBubble: {
    maxWidth: '80%',
    padding: 12,
    borderRadius: 18,
    marginVertical: 5,
  },
  userBubble: {
    backgroundColor: '#007AFF',
    alignSelf: 'flex-end',
    borderBottomRightRadius: 5,
  },
  agentBubble: {
    backgroundColor: '#E5E5EA',
    alignSelf: 'flex-start',
    borderBottomLeftRadius: 5,
  },
  messageText: {
    fontSize: 16,
    color: '#000',
  },
  messageImage: {
    width: 200,
    height: 200,
    borderRadius: 10,
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    backgroundColor: 'rgba(0,0,0,0.05)',
  },
  loadingText: {
    marginLeft: 10,
    fontSize: 14,
    color: '#666',
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 10,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#E5E5EA',
    alignItems: 'center',
  },
  input: {
    flex: 1,
    backgroundColor: '#F2F2F7',
    borderRadius: 20,
    paddingHorizontal: 15,
    paddingVertical: 8,
    marginHorizontal: 8,
    fontSize: 16,
  },
  attachButton: {
    padding: 8,
  },
  micButton: {
    padding: 8,
  },
  recordingButton: {
    backgroundColor: 'rgba(255,59,48,0.1)',
    borderRadius: 20,
  },
  sendButton: {
    padding: 8,
  },
});

export default ChatScreen;
```

### Session and Authentication Service

```javascript
// services/sessionService.js
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL } from '../config';

// Initialize a new session with the context protocol server
export const initializeSession = async (userToken) => {
  try {
    const response = await fetch(`${API_BASE_URL}/session/create`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${userToken}`
      }
    });
    
    if (!response.ok) {
      throw new Error(`Failed to initialize session: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Store session token for reconnecting
    await AsyncStorage.setItem('sessionToken', data.session_token);
    
    // Return session ID to be used for WebSocket connection
    return data.session_id;
  } catch (error) {
    console.error('Error initializing session:', error);
    throw error;
  }
};

// Reconnect to an existing session
export const reconnectSession = async (sessionToken) => {
  try {
    const response = await fetch(`${API_BASE_URL}/session/reconnect`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${sessionToken}`
      }
    });
    
    if (!response.ok) {
      // Token expired or invalid, remove from storage
      if (response.status === 401) {
        await AsyncStorage.removeItem('sessionToken');
      }
      throw new Error(`Failed to reconnect: ${response.status}`);
    }
    
    const data = await response.json();
    return data.session_id;
  } catch (error) {
    console.error('Error reconnecting to session:', error);
    throw error;
  }
};

// End the current session
export const endSession = async (sessionId, sessionToken) => {
  try {
    await fetch(`${API_BASE_URL}/session/${sessionId}/end`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${sessionToken}`
      }
    });
    
    // Clear session token from storage
    await AsyncStorage.removeItem('sessionToken');
  } catch (error) {
    console.error('Error ending session:', error);
  }
};
```

### Chat Service Implementation

```javascript
// services/chatService.js
import { API_BASE_URL, WS_BASE_URL } from '../config';

// Connect to the agent WebSocket stream
export const connectToAgentStream = (sessionId) => {
  const ws = new WebSocket(`${WS_BASE_URL}/agent/stream/${sessionId}`);
  
  // Set up ping interval to keep connection alive
  const pingInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'ping' }));
    }
  }, 30000); // Send ping every 30 seconds
  
  // Clear interval when connection closes
  ws.onclose = () => {
    clearInterval(pingInterval);
  };
  
  return ws;
};

// Send a message to the agent
export const sendMessage = (ws, messageData) => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(messageData));
    return true;
  }
  return false;
};

// Get chat history from the server
export const getChatHistory = async (sessionId, userToken) => {
  try {
    const response = await fetch(`${API_BASE_URL}/chat/${sessionId}/history`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${userToken}`
      }
    });
    
    if (!response.ok) {
      throw new Error(`Failed to get chat history: ${response.status}`);
    }
    
    const data = await response.json();
    return data.messages;
  } catch (error) {
    console.error('Error getting chat history:', error);
    throw error;
  }
};
```

## 2. Context Protocol Server

The Context Protocol Server acts as the central hub for managing communication between the mobile client and the AI components.

```python
# context_protocol_server.py
import asyncio
from fastapi import FastAPI, WebSocket, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
import json
import uuid
import jwt
import datetime
from motor.motor_asyncio import AsyncIOMotorClient

app = FastAPI()

# Configure CORS for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection setup
DATABASE_URL = "mongodb://localhost:27017"
client = AsyncIOMotorClient(DATABASE_URL)
db = client.ai_agent_db

# JWT settings
JWT_SECRET = "your-secret-key"  # Use env variables in production
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_DELTA = datetime.timedelta(days=7)

# Models
class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: str
    created_at: datetime.datetime

class SessionCreate(BaseModel):
    user_id: str
    device_info: Optional[Dict[str, Any]] = None

class SessionResponse(BaseModel):
    session_id: str
    session_token: str

class MessageData(BaseModel):
    type: str
    content: Any
    metadata: Optional[Dict[str, Any]] = None

class ContextItem(BaseModel):
    key: str
    value: Any
    source: str
    timestamp: datetime.datetime = datetime.datetime.now()
    ttl: Optional[int] = None  # Time to live in seconds

# Authentication
async def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != 'bearer':
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = await db.users.find_one({"_id": user_id})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        return user
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Session management
@app.post("/session/create", response_model=SessionResponse)
async def create_session(session_data: SessionCreate, current_user: dict = Depends(get_current_user)):
    # Create a new session
    session_id = str(uuid.uuid4())
    
    # Create session token
    payload = {
        "sub": current_user["_id"],
        "session": session_id,
        "exp": datetime.datetime.utcnow() + JWT_EXPIRATION_DELTA
    }
    session_token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    # Store session in database
    await db.sessions.insert_one({
        "_id": session_id,
        "user_id": current_user["_id"],
        "created_at": datetime.datetime.utcnow(),
        "device_info": session_data.device_info,
        "last_active": datetime.datetime.utcnow()
    })
    
    # Initialize empty context for the session
    await db.contexts.insert_one({
        "session_id": session_id,
        "context": {},
        "history": [],
        "metadata": {
            "created_at": datetime.datetime.utcnow(),
            "user_id": current_user["_id"]
        }
    })
    
    return {"session_id": session_id, "session_token": session_token}

@app.post("/session/reconnect", response_model=dict)
async def reconnect_session(current_user: dict = Depends(get_current_user)):
    # Find the most recent active session for the user
    session = await db.sessions.find_one(
        {"user_id": current_user["_id"]},
        sort=[("last_active", -1)]
    )
    
    if not session:
        raise HTTPException(status_code=404, detail="No session found")
    
    # Update last active time
    await db.sessions.update_one(
        {"_id": session["_id"]},
        {"$set": {"last_active": datetime.datetime.utcnow()}}
    )
    
    return {"session_id": session["_id"]}

@app.post("/session/{session_id}/end")
async def end_session(session_id: str, current_user: dict = Depends(get_current_user)):
    # Verify session belongs to user
    session = await db.sessions.find_one({"_id": session_id, "user_id": current_user["_id"]})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Mark session as ended
    await db.sessions.update_one(
        {"_id": session_id},
        {"$set": {"ended_at": datetime.datetime.utcnow()}}
    )
    
    return {"status": "Session ended"}

# Context management
@app.post("/context/{session_id}/update")
async def update_context(
    session_id: str,
    context_update: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    # Verify session belongs to user
    session = await db.sessions.find_one({"_id": session_id, "user_id": current_user["_id"]})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Update context
    context_doc = await db.contexts.find_one({"session_id": session_id})
    if not context_doc:
        raise HTTPException(status_code=404, detail="Context not found")
    
    # Merge new context items
    updated_context = context_doc["context"].copy()
    for key, value in context_update.items():
        updated_context[key] = {
            "value": value,
            "updated_at": datetime.datetime.utcnow(),
            "source": "client"
        }
    
    # Update in database
    await db.contexts.update_one(
        {"session_id": session_id},
        {"$set": {"context": updated_context}}
    )
    
    return {"status": "Context updated"}

@app.get("/context/{session_id}")
async def get_context(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    # Verify session belongs to user
    session = await db.sessions.find_one({"_id": session_id, "user_id": current_user["_id"]})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get context
    context_doc = await db.contexts.find_one({"session_id": session_id})
    if not context_doc:
        raise HTTPException(status_code=404, detail="Context not found")
    
    # Format context for response
    formatted_context = {}
    for key, item in context_doc["context"].items():
        formatted_context[key] = item["value"]
    
    return {
        "session_id": session_id,
        "context": formatted_context,
        "history": context_doc["history"],
        "metadata": context_doc["metadata"]
    }

# WebSocket connection for real-time agent communication
@app.websocket("/agent/stream/{session_id}")
async def agent_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    # Validate session and user (using the first message for auth)
    try:
        auth_message = await websocket.receive_text()
        auth_data = json.loads(auth_message)
        
        if "token" not in auth_data:
            await websocket.send_json({"error": "Authentication required"})
            await websocket.close()
            return
        
        # Verify token
        try:
            payload = jwt.decode(auth_data["token"], JWT_SECRET, algorithms=[JWT_ALGORITHM])
            user_id = payload.get("sub")
            token_session = payload.get("session")
            
            if not user_id or token_session != session_id:
                raise ValueError("Invalid token")
                
            # Check if session exists
            session = await db.sessions.find_one({"_id": session_id, "user_id": user_id})
            if not session:
                raise ValueError("Session not found")
                
        except (jwt.PyJWTError, ValueError) as e:
            await websocket.send_json({"error": f"Authentication failed: {str(e)}"})
            await websocket.close()
            return
            
        # Send confirmation
        await websocket.send_json({"type": "connection", "status": "authenticated"})
        
        # Main communication loop
        while True:
            try:
                # Receive message from client
                message_text = await websocket.receive_text()
                message_data = json.loads(message_text)
                
                # Update session last active time
                await db.sessions.update_one(
                    {"_id": session_id},
                    {"$set": {"last_active": datetime.datetime.utcnow()}}
                )
                
                # Skip ping messages
                if message_data.get("type") == "ping":
                    continue
                
                # Add message to history
                if message_data.get("type") in ["message", "image", "audio", "file"]:
                    await db.contexts.update_one(
                        {"session_id": session_id},
                        {"$push": {"history": {
                            "role": "user",
                            "content": message_data.get("content"),
                            "type": message_data.get("type"),
                            "timestamp": datetime.datetime.utcnow()
                        }}}
                    )
                
                # Process with LLM and agents
                response = await process_with_llm_and_agents(session_id, message_data)
                
                # Send response back
                await websocket.send_json(response)
                
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format"})
            except Exception as e:
                print(f"WebSocket error: {e}")
                await websocket.send_json({"error": f"Error processing request: {str(e)}"})
    
    except Exception as e:
        print(f"WebSocket connection error: {e}")
    finally:
        # Clean up when WebSocket closes
        try:
            await websocket.close()
        except:
            pass

async def process_with_llm_and_agents(session_id: str, message: Dict[str, Any]):
    """Process user message with LLM and agent orchestrator"""
    
    # Get current context
    context_doc = await db.contexts.find_one({"session_id": session_id})
    if not context_doc:
        return {"error": "Context not found"}
    
    # Determine message type and prepare payload for LLM
    message_type = message.get("type", "message")
    content = message.get("content", "")
    
    # Prepare payload based on message type
    if message_type == "message":
        llm_payload = {
            "context": context_doc,
            "message": content,
            "user_id": context_doc["metadata"]["user_id"]
        }
    elif message_type == "image":
        # For image, we need to process it first (e.g. OCR, image analysis)
        image_analysis = await analyze_image(content)
        llm_payload = {
            "context": context_doc,
            "message": "Analyzing image...",
            "image_analysis": image_analysis,
            "user_id": context_doc["metadata"]["user_id"]
        }
    elif message_type == "audio":
        # For audio, send for transcription first
        transcription = await transcribe_audio(content)
        llm_payload = {
            "context": context_doc,
            "message": transcription,
            "is_transcription": True,
            "user_id": context_doc["metadata"]["user_id"]
        }
    else:
        return {"error": "Unsupported message type"}
    
    # Call LLM service
    try:
        async with httpx.AsyncClient() as client:
            llm_response = await client.post(
                "http://llm-server:8001/generate",
                json=llm_payload,
                timeout=30.0
            )
            
            if llm_response.status_code != 200:
                return {"error": f"LLM service error: {llm_response.text}"}
            
            response_data = llm_response.json()
    except Exception as e:
        return {"error": f"Error calling LLM service: {str(e)}"}
    
    # Call agent orchestrator if needed
    try:
        if "actions" in response_data:
            actions = response_data["actions"]
            for action in actions:
                agent_payload = {
                    "session_id": session_id,
                    "action": action["name"],
                    "parameters": action["parameters"],
                    "context": context_doc
                }
                
                # Call agent orchestrator
                async with httpx.AsyncClient() as client:
                    agent_response = await client.post(
                        "http://agent-orchestrator:8002/agent/execute",
                        json=agent_payload,
                        timeout=30.0
                    )
                    
                    if agent_response.status_code == 200:
                        agent_data = agent_response.json()
                        
                        # Update context with agent results if needed
                        if agent_data.get("context_updates"):
                            await db.contexts.update_one(
                                {"session_id": session_id},
                                {"$set": {f"context.{k}": {
                                    "value": v,
                                    "updated_at": datetime.datetime.utcnow(),
                                    "source": "agent"
                                } for k, v in agent_data["context_updates"].items()}}
                            )
    except Exception as e:
        print(f"Agent orchestration error: {e}")
        # Continue even if agent calls fail
    
    # Add assistant response to history
    await db.contexts.update_one(
        {"session_id": session_id},
        {"$push": {"history": {
            "role": "assistant",
            "content": response_data["response"],
            "timestamp": datetime.datetime.utcnow()
        }}}
    )
    
    # Return formatted response to client
    client_response = {
        "type": "message",
        "content": response_data["response"],
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    
    return client_response

async def analyze_image(image_base64: str):
    """Send image to analysis service and return results"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://image-analysis-service:8003/analyze",
                json={"image": image_base64}
            )
            return response.json()
    except Exception as e:
        print(f"Image analysis error: {e}")
        return {"error": str(e)}

async def transcribe_audio(audio_base64: str):
    """Send audio to transcription service and return text"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://transcription-service:8004/transcribe",
                json={"audio": audio_base64}
            )
            result = response.json()
            return result.get("text", "")
    except Exception as e:
        print(f"Transcription error: {e}")
        return "Sorry, I couldn't transcribe the audio."

# User authentication endpoints
@app.post("/auth/register", response_model=dict)
async def register_user(user_data: UserCreate):
    # Check if user already exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password
    hashed_password = bcrypt.hashpw(user_data.password.encode(), bcrypt.gensalt())
    
    # Create new user
    user_id = str(uuid.uuid4())
    user = {
        "_id": user_id,
        "username": user_data.username,
        "email": user_data.email,
        "password": hashed_password.decode(),
        "created_at": datetime.datetime.utcnow()
    }
    
    await db.users.insert_one(user)
    
    # Generate token
    access_token = create_access_token(user_id)
    
    return {
        "id": user_id,
        "username": user_data.username,
        "email": user_data.email,
        "access_token": access_token
    }

@app.post("/auth/login", response_model=dict)
async def login_user(credentials: dict):
    email = credentials.get("email")
    password = credentials.get("password")
    
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")
    
    # Find user
    user = await db.users.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password
    if not bcrypt.checkpw(password.encode(), user["password"].encode()):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Generate token
    access_token = create_access_token(user["_id"])
    
    return {
        "id": user["_id"],
        "username": user["username"],
        "email": user["email"],
        "access_token": access_token
    }

def create_access_token(user_id: str):
    """Create a new JWT token"""
    payload = {
        "sub": user_id,
        "exp": datetime.datetime.utcnow() + JWT_EXPIRATION_DELTA
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

# Chat history endpoints
@app.get("/chat/{session_id}/history")
async def get_chat_history(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    # Verify session belongs to user
    session = await db.sessions.find_one({"_id": session_id, "user_id": current_user["_id"]})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get context with history
    context_doc = await db.contexts.find_one({"session_id": session_id})
    if not context_doc:
        raise HTTPException(status_code=404, detail="Context not found")
    
    return {"messages": context_doc["history"]}

if __name__ == "__main__":
    import uvicorn
    import bcrypt
    import httpx
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 3. LLM Server

The LLM Server interfaces with different LLM providers and handles model-specific operations.

```python
# llm_server.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
import httpx
import os
import json
import asyncio
import logging
from datetime import datetime

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models
class LLMRequest(BaseModel):
    context: Dict[str, Any]
    message: str
    model: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    image_analysis: Optional[Dict[str, Any]] = None
    is_transcription: Optional[bool] = False
    user_id: Optional[str] = None

class LLMResponse(BaseModel):
    response: str
    model_used: str
    tokens: Dict[str, int]
    actions: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

# API keys would be stored securely in environment variables
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "cohere": os.getenv("COHERE_API_KEY"),
    "local": None  # No key needed for local models
}

MODEL_ENDPOINTS = {
    "gpt-4-turbo": {"provider": "openai", "endpoint": "https://api.openai.com/v1/chat/completions"},
    "gpt-4-vision": {"provider": "openai", "endpoint": "https://api.openai.com/v1/chat/completions"},
    "claude-3-opus": {"provider": "anthropic", "endpoint": "https://api.anthropic.com/v1/messages"},
    "claude-3-sonnet": {"provider": "anthropic", "endpoint": "https://api.anthropic.com/v1/messages"},
    "command-r": {"provider": "cohere", "endpoint": "https://api.cohere.ai/v1/generate"},
    "local-llama": {"provider": "local", "endpoint": "http://localhost:8080/generate"}
}

DEFAULT_MODEL = "gpt-4-turbo"
VISION_MODEL = "gpt-4-vision"

# User preferences cache (in production, this would be in a database)
user_model_preferences = {}

@app.post("/generate", response_model=LLMResponse)
async def generate_response(request: LLMRequest, background_tasks: BackgroundTasks):
    try:
        # Check for user model preference
        model = request.model
        if not model and request.user_id in user_model_preferences:
            model = user_model_preferences[request.user_id]
        if not model:
            model = DEFAULT_MODEL
            
        # Use vision model if there's image analysis
        if request.image_analysis and not request.model:
            model = VISION_MODEL
            
        if model not in MODEL_ENDPOINTS:
            logger.warning(f"Model {model} not found, falling back to {DEFAULT_MODEL}")
            model = DEFAULT_MODEL
            
        provider_info = MODEL_ENDPOINTS[model]
        
        # Prepare the context and message for the model
        formatted_context = format_context_for_model(request.context, model)
        formatted_message = request.message
        
        # Add image analysis if present
        if request.image_analysis:
            formatted_message = augment_with_image_analysis(formatted_message, request.image_analysis, model)
        
        # Add transcription info if needed
        if request.is_transcription:
            formatted_message = f"[Transcribed audio]: {formatted_message}"
        
        # Determine if any tools/agents need to be available
        required_tools = determine_required_tools(formatted_context, formatted_message)
        
        async with httpx.AsyncClient() as client:
            response = await call_model_api(
                client, 
                provider_info["provider"], 
                provider_info["endpoint"], 
                formatted_context, 
                formatted_message, 
                model, 
                request.parameters,
                required_tools
            )
        
        processed_response = process_model_response(response, model, required_tools)
        
        # Background task to improve future interactions
        background_tasks.add_task(
            learn_from_interaction,
            request.context,
            request.message,
            processed_response,
            request.user_id
        )
        
        return processed_response
    
    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calling LLM API: {str(e)}")

async def call_model_api(client, provider, endpoint, context, message, model, parameters, required_tools=None):
    """Call the appropriate LLM API based on the provider"""
    
    # Default parameters
    default_params = {
        "openai": {
            "temperature": 0.7,
            "max_tokens": 1000,
        },
        "anthropic": {
            "temperature": 0.7,
            "max_tokens": 1000,
        },
        "cohere": {
            "temperature": 0.7,
            "max_tokens": 1000,
        },
        "local": {
            "temperature": 0.7,
            "max_tokens": 1000,
        }
    }
    
    # Merge with provided parameters
    model_params = default_params[provider].copy()
    if parameters:
        model_params.update(parameters)
    
    # Model-specific API formatting
    if provider == "openai":
        messages = format_messages_for_openai(context, message)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": model_params["temperature"],
            "max_tokens": model_params["max_tokens"],
        }
        
        # Add tools if needed
        if required_tools:
            payload["tools"] = format_tools_for_openai(required_tools)
            payload["tool_choice"] = "auto"
            
        headers = {"Authorization": f"Bearer {API_KEYS['openai']}"}
        response = await client.post(endpoint, json=payload, headers=headers, timeout=60.0)
        
    elif provider == "anthropic":
        messages = format_messages_for_anthropic(context, message)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": model_params["temperature"],
            "max_tokens": model_params["max_tokens"],
        }
        
        # Add tools if supported
        if required_tools and model in ["claude-3-opus", "claude-3-sonnet"]:
            payload["tools"] = format_tools_for_anthropic(required_tools)
            
        headers = {"x-api-key": API_KEYS['anthropic'], "anthropic-version": "2023-06-01"}
        response = await client.post(endpoint, json=payload, headers=headers, timeout=60.0)
        
    elif provider == "cohere":
        prompt = format_prompt_for_cohere(context, message)
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": model_params["temperature"],
            "max_tokens": model_params["max_tokens"],
        }
        headers = {"Authorization": f"Bearer {API_KEYS['cohere']}"}
        response = await client.post(endpoint, json=payload, headers=headers, timeout=60.0)
        
    elif provider == "local":
        prompt = format_prompt_for_local(context, message)
        payload = {
            "prompt": prompt,
            "temperature": model_params["temperature"],
            "max_tokens": model_params["max_tokens"],
        }
        response = await client.post(endpoint, json=payload, timeout=60.0)
    
    return {"raw_response": response.json(), "provider": provider, "model": model}

def format_context_for_model(context, model):
    """Format the context based on the model's needs"""
    # Extract actual context values from the DB structure
    formatted_context = {}
    
    if "context" in context:
        for key, item in context["context"].items():
            if isinstance(item, dict) and "value" in item:
                formatted_context[key] = item["value"]
            else:
                formatted_context[key] = item
    
    # Add history in a standardized format
    if "history" in context:
        formatted_context["history"] = context["history"]
    
    return formatted_context

def format_messages_for_openai(context, message):
    """Format the context and message for OpenAI's API"""
    messages = []
    
    # Add system prompt with context if available
    system_prompt = """You are a helpful mobile AI assistant. 
You can access contextual information like device data, previous interactions, and user preferences.
For complex tasks, you can use specialized tools when necessary."""
    
    # Add context-specific instructions
    if "user_preferences" in context:
        system_prompt += "\nUser preferences: " + json.dumps(context["user_preferences"])
    
    if "device_info" in context:
        system_prompt += "\nDevice info: " + json.dumps(context["device_info"])
        
    messages.append({"role": "system", "content": system_prompt})
    
    # Add conversation history
    if "history" in context:
        for entry in context["history"][-10:]:  # Get last 10 messages only
            role = "user" if entry["role"] == "user" else "assistant"
            content = entry["content"]
            
            # Check if there's a specific type
            if entry.get("type") == "image" and role == "user":
                # For image messages, create a multimodal content array
                messages.append({
                    "role": role,
                    "content": [
                        {"type": "text", "text": "I'm sending you an image to analyze"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{content}"}}
                    ]
                })
            else:
                messages.append({"role": role, "content": content})
    
    # Add the current message
    messages.append({"role": "user", "content": message})
    
    return messages

def format_messages_for_anthropic(context, message):
    """Format the context and message for Anthropic's API"""
    messages = []
    
    # Add system prompt with context
    system_prompt = """You are a helpful mobile AI assistant. 
You can access contextual information like device data, previous interactions, and user preferences.
For complex tasks, you can use specialized tools when necessary."""
    
    # Add conversation history
    if "history" in context:
        for entry in context["history"][-10:]:  # Get last 10 messages only
            role = "user" if entry["role"] == "user" else "assistant"
            content = entry["content"]
            messages.append({"role": role, "content": content})
    
    # Add the current message
    messages.append({"role": "user", "content": message})
    
    return messages, system_prompt

def format_prompt_for_cohere(context, message):
    """Format the context and message for Cohere's API"""
    prompt = "You are a helpful mobile AI assistant.\n\n"
    
    # Add conversation history
    if "history" in context:
        for entry in context["history"][-10:]:  # Get last 10 messages only
            speaker = "User" if entry["role"] == "user" else "Assistant"
            prompt += f"{speaker}: {entry['content']}\n"
    
    # Add the current message
    prompt += f"User: {message}\nAssistant: "
    return prompt

def format_prompt_for_local(context, message):
    """Format the context and message for local models"""
    # Similar to Cohere format but may need adjustments based on local model
    return format_prompt_for_cohere(context, message)

def format_tools_for_openai(tools):
    """Format tools for OpenAI's API"""
    formatted_tools = []
    
    for tool in tools:
        formatted_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
        }
        formatted_tools.append(formatted_tool)
    
    return formatted_tools

def format_tools_for_anthropic(tools):
    """Format tools for Anthropic's API"""
    formatted_tools = []
    
    for tool in tools:
        formatted_tool = {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"]
        }
        formatted_tools.append(formatted_tool)
    
    return formatted_tools

def determine_required_tools(context, message):
    """Determine which tools might be needed based on context and message"""
    tools = []
    
    # Simple heuristic-based detection - in a real system, 
    # you would use more sophisticated methods
    
    # Check for weather-related queries
    if any(keyword in message.lower() for keyword in ["weather", "temperature", "forecast", "rain"]):
        tools.append({
            "name": "get_weather",
            "description": "Get current weather or forecast for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days for forecast",
                        "default": 1
                    }
                },
                "required": ["location"]
            }
        })
    
    # Check for search-related queries
    if any(keyword in message.lower() for keyword in ["search", "find", "lookup", "information about"]):
        tools.append({
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        })
    
    # Check for location-related queries
    if any(keyword in message.lower() for keyword in ["nearby", "location", "find me", "where is", "directions"]):
        tools.append({
            "name": "location_search",
            "description": "Find places near the user or get directions",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["restaurant", "gas_station", "coffee", "pharmacy", "grocery", "hotel", "any"],
                        "default": "any"
                    },
                    "radius": {
                        "type": "integer",
                        "description": "Search radius in meters",
                        "default": 1000
                    }
                },
                "required": ["query"]
            }
        })
    
    return tools

def process_model_response(response, model, required_tools=None):
    """Process the response from the model"""
    raw_response = response["raw_response"]
    provider = response["provider"]
    
    if provider == "openai":
        content = raw_response["choices"][0]["message"]["content"]
        tokens = {
            "prompt_tokens": raw_response["usage"]["prompt_tokens"],
            "completion_tokens": raw_response["usage"]["completion_tokens"],
            "total_tokens": raw_response["usage"]["total_tokens"]
        }
        
        # Check for tool calls
        actions = []
        message = raw_response["choices"][0]["message"]
        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                function_call = tool_call["function"]
                actions.append({
                    "name": function_call["name"],
                    "parameters": json.loads(function_call["arguments"])
                })
    
    elif provider == "anthropic":
        content = raw_response["content"][0]["text"]
        tokens = {
            "input_tokens": raw_response.get("usage", {}).get("input_tokens", 0),
            "output_tokens": raw_response.get("usage", {}).get("output_tokens", 0),
            "total_tokens": raw_response.get("usage", {}).get("input_tokens", 0) + raw_response.get("usage", {}).get("output_tokens", 0)
        }
        
        # Check for tool calls
        actions = []
        if "tool_use" in raw_response:
            for tool_use in raw_response["tool_use"]:
                actions.append({
                    "name": tool_use["name"],
                    "parameters": tool_use["input"]
                })
    
    elif provider == "cohere":
        content = raw_response["generations"][0]["text"]
        tokens = {
            "prompt_tokens": raw_response.get("meta", {}).get("billed_units", {}).get("input_tokens", 0),
            "completion_tokens": raw_response.get("meta", {}).get("billed_units", {}).get("output_tokens", 0),
            "total_tokens": raw_response.get("meta", {}).get("billed_units", {}).get("total_tokens", 0)
        }
        actions = []  # No native tool calling in Cohere
        
    else:  # Local model
        content = raw_response.get("generated_text", "No response generated")
        tokens = {"total_tokens": 0}
        actions = []
    
    return LLMResponse(
        response=content,
        model_used=model,
        tokens=tokens,
        actions=actions if actions else None,
        metadata={"timestamp": datetime.now().isoformat()}
    )

def augment_with_image_analysis(message, image_analysis, model):
    """Add image analysis information to the message"""
    if not image_analysis:
        return message
    
    if "error" in image_analysis:
        return f"{message}\n\n[Note: There was an error analyzing the attached image: {image_analysis['error']}]"
    
    analysis_text = f"{message}\n\n[Image Analysis: "
    
    if "caption" in image_analysis:
        analysis_text += f"Caption: {image_analysis['caption']}. "
    
    if "objects" in image_analysis:
        objects = image_analysis["objects"]
        if objects:
            analysis_text += f"Objects detected: {', '.join(objects)}. "
    
    if "text" in image_analysis:
        analysis_text += f"Text in image: {image_analysis['text']}. "
    
    if "faces" in image_analysis and image_analysis["faces"]:
        face_count = len(image_analysis["faces"])
        analysis_text += f"{face_count} {'face' if face_count == 1 else 'faces'} detected. "
    
    analysis_text += "]"
    
    return analysis_text

async def learn_from_interaction(context, message, response, user_id):
    """Learn from the interaction to improve future responses"""
    if not user_id:
        return
    
    # Update user model preferences if they consistently use the same model
    if user_id in user_model_preferences:
        # No need to update if it's the same model
        if user_model_preferences[user_id] == response.model_used:
            return
    
    # Set or update the user's preferred model
    user_model_preferences[user_id] = response.model_used
    logger.info(f"Updated user {user_id} model preference to {response.model_used}")

@app.post("/set-user-preference")
async def set_user_preference(data: Dict[str, Any]):
    """Set a user's model preference"""
    user_id = data.get("user_id")
    model = data.get("model")
    
    if not user_id or not model:
        raise HTTPException(status_code=400, detail="User ID and model are required")
    
    if model not in MODEL_ENDPOINTS:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model}")
    
    user_model_preferences[user_id] = model
    
    return {"status": "success", "message": f"Model preference set to {model}"}

@app.get("/models")
async def list_available_models():
    """List all available models"""
    return {"models": list(MODEL_ENDPOINTS.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

## 4. Agent Orchestrator

The Agent Orchestrator manages different AI agents, their capabilities, and their interactions.

```python
# agent_orchestrator.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
import httpx
import importlib
import inspect
import os
import json
import logging
import asyncio
import datetime
import jwt

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# JWT settings for authentication
JWT_SECRET = "your-secret-key"  # Use env variables in production
JWT_ALGORITHM = "HS256"

# Models
class AgentRequest(BaseModel):
    session_id: str
    context: Dict[str, Any]
    action: str
    parameters: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    session_id: str
    action_result: Any
    context_updates: Optional[Dict[str, Any]] = None
    next_actions: Optional[List[Dict[str, Any]]] = None

# Authentication
async def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != 'bearer':
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return {"user_id": user_id}
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")

# Agent registry
class AgentRegistry:
    def __init__(self):
        self.agents = {}
        
    def register_agent(self, agent_name, capabilities, agent_instance):
        """Register an agent with its capabilities"""
        for capability in capabilities:
            self.agents[capability] = {
                "name": agent_name,
                "instance": agent_instance
            }
        logger.info(f"Registered agent {agent_name} with capabilities: {capabilities}")
    
    def get_agent_for_action(self, action):
        """Get the appropriate agent for an action"""
        if action not in self.agents:
            return None
        return self.agents[action]["instance"]
    
    def get_all_capabilities(self):
        """Get all registered capabilities"""
        return list(self.agents.keys())
    
    def get_agents_info(self):
        """Get information about all registered agents"""
        agents_info = {}
        for action, agent_data in self.agents.items():
            agent_name = agent_data["name"]
            if agent_name not in agents_info:
                agent_instance = agent_data["instance"]
                agents_info[agent_name] = {
                    "name": agent_name,
                    "description": getattr(agent_instance, "description", "No description"),
                    "capabilities": []
                }
            
            # Add capability if not already added
            capability = {
                "name": action,
                "description": getattr(agent_data["instance"], f"{action}_description", "No description"),
                "parameters": getattr(agent_data["instance"], f"{action}_parameters", {})
            }
            
            if capability not in agents_info[agent_name]["capabilities"]:
                agents_info[agent_name]["capabilities"].append(capability)
                
        return agents_info

# Singleton registry
registry = AgentRegistry()

# Load agents from the agents directory
def load_agents():
    try:
        agents_dir = os.path.join(os.path.dirname(__file__), "agents")
        if not os.path.exists(agents_dir):
            logger.warning(f"Agents directory not found: {agents_dir}")
            return
            
        for filename in os.listdir(agents_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                try:
                    module_name = filename[:-3]
                    module = importlib.import_module(f"agents.{module_name}")
                    
                    # Look for Agent class in the module
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and hasattr(obj, "capabilities"):
                            try:
                                agent = obj()
                                registry.register_agent(name, agent.capabilities, agent)
                            except Exception as e:
                                logger.error(f"Error instantiating agent {name}: {e}")
                except Exception as e:
                    logger.error(f"Error loading agent module {filename}: {e}")
    except Exception as e:
        logger.error(f"Error in load_agents: {e}")

@app.on_event("startup")
async def startup_event():
    """Load all agents on startup"""
    load_agents()

@app.post("/agent/execute", response_model=AgentResponse)
async def execute_agent_action(
    request: AgentRequest, 
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Execute an agent action"""
    agent = registry.get_agent_for_action(request.action)
    if not agent:
        raise HTTPException(status_code=404, detail=f"No agent found for action: {request.action}")
    
    try:
        # Add user_id to context
        context = request.context.copy()
        if "metadata" not in context:
            context["metadata"] = {}
        context["metadata"]["user_id"] = current_user["user_id"]
        
        # Execute the agent action
        result = await agent.execute(request.action, context, request.parameters or {})
        
        # Check if there are background tasks to run
        if hasattr(result, "background_tasks") and result.background_tasks:
            for task in result.background_tasks:
                background_tasks.add_task(
                    task["function"],
                    *task.get("args", []),
                    **task.get("kwargs", {})
                )
        
        # Prepare the response
        response = AgentResponse(
            session_id=request.session_id,
            action_result=result.get("result"),
            context_updates=result.get("context_updates"),
            next_actions=result.get("next_actions")
        )
        
        # Log the action for monitoring
        logger.info(f"Executed action {request.action} for session {request.session_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error executing agent action {request.action}: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing agent action: {str(e)}")

@app.get("/agent/capabilities")
async def get_capabilities():
    """Get all available agent capabilities"""
    return {
        "capabilities": registry.get_all_capabilities(),
        "agents": registry.get_agents_info()
    }

# Sample agent implementations
# In a real project, these would be in separate files in the agents directory

class SearchAgent:
    """Agent that performs various search operations"""
    
    def __init__(self):
        self.description = "Agent that performs various search operations"
        self.capabilities = ["web_search", "document_search", "knowledge_retrieval"]
        
        # Descriptions and parameters
        self.web_search_description = "Search the web for information"
        self.web_search_parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
        
        self.document_search_description = "Search through internal documents"
        self.document_search_parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string"
                },
                "document_types": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of document types to search"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 10
                }
            },
            "required": ["query"]
        }
        
        self.knowledge_retrieval_description = "Retrieve information from knowledge base"
        self.knowledge_retrieval_parameters = {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to retrieve information about"
                },
                "depth": {
                    "type": "string",
                    "enum": ["basic", "intermediate", "advanced"],
                    "description": "Depth of information",
                    "default": "basic"
                }
            },
            "required": ["topic"]
        }
    
    async def execute(self, action, context, parameters):
        """Execute the specified action with the given parameters"""
        if action == "web_search":
            return await self.web_search(context, parameters)
        elif action == "document_search":
            return await self.document_search(context, parameters)
        elif action == "knowledge_retrieval":
            return await self.knowledge_retrieval(context, parameters)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def web_search(self, context, parameters):
        """Search the web for information"""
        query = parameters.get("query")
        num_results = parameters.get("num_results", 5)
        
        if not query:
            return {
                "result": {"error": "No query provided"},
                "context_updates": None
            }
        
        # In a real implementation, this would call a search API
        try:
            async with httpx.AsyncClient() as client:
                # Example using a web search API
                response = await client.get(
                    "https://api.searchapi.example.com/search",
                    params={"q": query, "num": num_results},
                    headers={"Authorization": f"Bearer {os.getenv('SEARCH_API_KEY')}"}
                )
                
                search_results = response.json()
                
                # Format the search results
                formatted_results = []
                for result in search_results.get("results", []):
                    formatted_results.append({
                        "title": result.get("title", "No title"),
                        "snippet": result.get("snippet", "No snippet"),
                        "url": result.get("url", ""),
                        "source": "web_search"
                    })
                
                return {
                    "result": formatted_results,
                    "context_updates": {
                        "last_search_query": query,
                        "last_search_results": formatted_results,
                        "last_search_time": datetime.datetime.now().isoformat()
                    }
                }
        except Exception as e:
            logger.error(f"Web search error: {e}")
            # For demo purposes, return mock results
            mock_results = [
                {
                    "title": f"Result 1 for {query}",
                    "snippet": f"This is a sample result matching the query {query}",
                    "url": "https://example.com/result1",
                    "source": "mock_data"
                },
                {
                    "title": f"Result 2 for {query}",
                    "snippet": f"Another sample result with information about {query}",
                    "url": "https://example.com/result2",
                    "source": "mock_data"
                }
            ]
            
            return {
                "result": mock_results[:num_results],
                "context_updates": {
                    "last_search_query": query,
                    "last_search_results": mock_results[:num_results],
                    "last_search_time": datetime.datetime.now().isoformat()
                }
            }
    
    async def document_search(self, context, parameters):
        """Search through internal documents"""
        # Implementation would be similar to web_search but focused on internal docs
        # Return mock data for now
        query = parameters.get("query")
        document_types = parameters.get("document_types", [])
        max_results = parameters.get("max_results", 10)
        
        mock_results = [
            {
                "title": f"Document 1: {query} Overview",
                "snippet": f"This internal document contains information about {query}",
                "doc_id": "doc123",
                "source": "internal_knowledge_base"
            },
            {
                "title": f"Document 2: Guide to {query}",
                "snippet": f"A comprehensive guide explaining {query} concepts",
                "doc_id": "doc456",
                "source": "internal_knowledge_base"
            }
        ]
        
        return {
            "result": mock_results[:max_results],
            "context_updates": {
                "last_document_search": {
                    "query": query,
                    "results_count": len(mock_results[:max_results])
                }
            }
        }
    
    async def knowledge_retrieval(self, context, parameters):
        """Retrieve information from knowledge base"""
        topic = parameters.get("topic")
        depth = parameters.get("depth", "basic")
        
        # This would retrieve information from a knowledge base
        mock_knowledge = {
            "topic": topic,
            "information": f"This is {depth} information about {topic}.",
            "sources": ["Knowledge Base Entry 1", "Knowledge Base Entry 2"]
        }
        
        if depth == "intermediate":
            mock_knowledge["information"] += f" Additional details about {topic} at an intermediate level."
        elif depth == "advanced":
            mock_knowledge["information"] += f" Comprehensive technical details about {topic} at an advanced level."
        
        return {
            "result": mock_knowledge,
            "context_updates": {
                f"knowledge_{topic}": mock_knowledge
            }
        }

class WeatherAgent:
    """Agent that provides weather information"""
    
    def __init__(self):
        self.description = "Agent that provides weather information"
        self.capabilities = ["get_weather", "get_forecast"]
        
        # Descriptions and parameters
        self.get_weather_description = "Get current weather for a location"
        self.get_weather_parameters = {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                    "description": "Temperature units"
                }
            },
            "required": ["location"]
        }
        
        self.get_forecast_description = "Get weather forecast for a location"
        self.get_forecast_parameters = {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location"
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days for forecast",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                    "description": "Temperature units"
                }
            },
            "required": ["location"]
        }
    
    async def execute(self, action, context, parameters):
        """Execute the specified action with the given parameters"""
        if action == "get_weather":
            return await self.get_weather(context, parameters)
        elif action == "get_forecast":
            return await self.get_forecast(context, parameters)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def get_weather(self, context, parameters):
        """Get current weather for a location"""
        location = parameters.get("location")
        units = parameters.get("units", "celsius")
        
        if not location:
            return {
                "result": {"error": "No location provided"},
                "context_updates": None
            }
        
        try:
            # In a real implementation, call a weather API
            async with httpx.AsyncClient() as client:
                # Example using OpenWeatherMap or similar API
                # api_key = os.getenv("WEATHER_API_KEY")
                # response = await client.get(
                #     "https://api.openweathermap.org/data/2.5/weather",
                #     params={
                #         "q": location,
                #         "units": "metric" if units == "celsius" else "imperial",
                #         "appid": api_key
                #     }
                # )
                # weather_data = response.json()
                
                # For demo, use mock data
                temp_value = 22 if units == "celsius" else 72
                weather_data = {
                    "location": location,
                    "temperature": {
                        "value": temp_value,
                        "unit": "°C" if units == "celsius" else "°F"
                    },
                    "conditions": "Partly cloudy",
                    "humidity": 65,
                    "wind": {
                        "speed": 10,
                        "unit": "km/h"
                    },
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                return {
                    "result": weather_data,
                    "context_updates": {
                        "last_weather_query": location,
                        "current_weather": weather_data
                    }
                }
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return {
                "result": {"error": f"Failed to get weather for {location}: {str(e)}"},
                "context_updates": None
            }
    
    async def get_forecast(self, context, parameters):
        """Get weather forecast for a location"""
        location = parameters.get("location")
        days = parameters.get("days", 5)
        units = parameters.get("units", "celsius")
        
        if not location:
            return {
                "result": {"error": "No location provided"},
                "context_updates": None
            }
        
        # For demo, use mock forecast data
        base_temp = 22 if units == "celsius" else 72
        forecast = []
        for i in range(days):
            day_offset = datetime.datetime.now() + datetime.timedelta(days=i)
            forecast.append({
                "date": day_offset.strftime("%Y-%m-%d"),
                "temperature": {
                    "min": base_temp - 3 + i,
                    "max": base_temp + 2 + i,
                    "unit": "°C" if units == "celsius" else "°F"
                },
                "conditions": ["Sunny", "Partly cloudy", "Cloudy", "Rainy"][i % 4],
                "precipitation": {
                    "probability": (i * 10) % 100,
                    "unit": "%"
                }
            })
        
        forecast_data = {
            "location": location,
            "forecast": forecast,
            "units": units,
            "generated_at": datetime.datetime.now().isoformat()
        }
        
        return {
            "result": forecast_data,
            "context_updates": {
                "last_forecast_query": location,
                "weather_forecast": forecast_data
            }
        }

class LocationAgent:
    """Agent that handles location-based services"""
    
    def __init__(self):
        self.description = "Agent that handles location-based services"
        self.capabilities = ["location_search", "get_directions"]
        
        # Descriptions and parameters
        self.location_search_description = "Find places near a location"
        self.location_search_parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for (e.g., 'coffee shops')"
                },
                "location": {
                    "type": "string",
                    "description": "Location to search near (e.g., city name)"
                },
                "current_location": {
                    "type": "boolean",
                    "description": "Use device's current location",
                    "default": False
                },
                "radius": {
                    "type": "integer",
                    "description": "Search radius in meters",
                    "default": 1000
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 5
                }
            },
            "required": ["query"]
        }
        
        self.get_directions_description = "Get directions between two locations"
        self.get_directions_parameters = {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "Starting location"
                },
                "destination": {
                    "type": "string",
                    "description": "Ending location"
                },
                "mode": {
                    "type": "string",
                    "enum": ["driving", "walking", "bicycling", "transit"],
                    "default": "driving",
                    "description": "Mode of transportation"
                }
            },
            "required": ["origin", "destination"]
        }
    
    async def execute(self, action, context, parameters):
        """Execute the specified action with the given parameters"""
        if action == "location_search":
            return await self.location_search(context, parameters)
        elif action == "get_directions":
            return await self.get_directions(context, parameters)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def location_search(self, context, parameters):
        """Find places near a location"""
        query = parameters.get("query")
        location = parameters.get("location")
        use_current = parameters.get("current_location", False)
        radius = parameters.get("radius", 1000)
        limit = parameters.get("limit", 5)
        
        if not query:
            return {
                "result": {"error": "No search query provided"},
                "context_updates": None
            }
        
        # Get current location from context if needed
        if use_current and not location:
            if "device_info" in context and "location" in context["device_info"]:
                location = context["device_info"]["location"]
            else:
                return {
                    "result": {"error": "Current location not available"},
                    "context_updates": None
                }
        
        if not location and not use_current:
            return {
                "result": {"error": "No location provided"},
                "context_updates": None
            }
        
        # Mock response for demo
        places = [
            {
                "name": f"{query.title()} Place 1",
                "address": f"123 Main St, {location}",
                "rating": 4.5,
                "distance": 450,
                "open_now": True
            },
            {
                "name": f"{query.title()} Place 2",
                "address": f"456 Oak Ave, {location}",
                "rating": 4.2,
                "distance": 750,
                "open_now": True
            },
            {
                "name": f"{query.title()} Place 3",
                "address": f"789 Pine St, {location}",
                "rating": 4.0,
                "distance": 950,
                "open_now": False
            }
        ]
        
        result = {
            "query": query,
            "location": location,
            "places": places[:limit]
        }
        
        return {
            "result": result,
            "context_updates": {
                "last_location_search": {
                    "query": query,
                    "location": location,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            }
        }
    
    async def get_directions(self, context, parameters):
        """Get directions between two locations"""
        origin = parameters.get("origin")
        destination = parameters.get("destination")
        mode = parameters.get("mode", "driving")
        
        if not origin or not destination:
            return {
                "result": {"error": "Origin and destination are required"},
                "context_updates": None
            }
        
        # Mock directions for demo
        steps = [
            f"Start from {origin}",
            "Head north on Main Street for 0.5 miles",
            "Turn right onto Oak Avenue",
            "Continue for 1.2 miles",
            "Turn left onto Pine Street",
            "Your destination will be on the right"
        ]
        
        directions = {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "distance": "2.3 miles",
            "duration": "12 minutes",
            "steps": steps
        }
        
        return {
            "result": directions,
            "context_updates": {
                "last_directions": {
                    "origin": origin,
                    "destination": destination,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            }
        }

# Register built-in agents on startup
@app.on_event("startup")
async def register_builtin_agents():
    """Register built-in agents on startup"""
    registry.register_agent("SearchAgent", SearchAgent().capabilities, SearchAgent())
    registry.register_agent("WeatherAgent", WeatherAgent().capabilities, WeatherAgent())
    registry.register_agent("LocationAgent", LocationAgent().capabilities, LocationAgent())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
```

## 5. Docker Deployment Setup

```yaml
# docker-compose.yml
version: '3'

services:
  mongodb:
    image: mongo:latest
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin
      - MONGO_INITDB_ROOT_PASSWORD=secretpassword
    networks:
      - agent-network

  context-server:
    build:
      context: ./backend
      dockerfile: Dockerfile.context-server
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=mongodb://admin:secretpassword@mongodb:27017
      - JWT_SECRET=${JWT_SECRET}
      - PYTHONUNBUFFERED=1
    depends_on:
      - mongodb
    networks:
      - agent-network

  llm-server:
    build:
      context: ./backend
      dockerfile: Dockerfile.llm-server
    ports:
      - "8001:8001"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - COHERE_API_KEY=${COHERE_API_KEY}
      - PYTHONUNBUFFERED=1
    networks:
      - agent-network

  agent-orchestrator:
    build:
      context: ./backend
      dockerfile: Dockerfile.agent-orchestrator
    ports:
      - "8002:8002"
    volumes:
      - ./backend/agents:/app/agents
    environment:
      - JWT_SECRET=${JWT_SECRET}
      - PYTHONUNBUFFERED=1
    depends_on:
      - context-server
      - llm-server
    networks:
      - agent-network

  image-analysis-service:
    build:
      context: ./backend
      dockerfile: Dockerfile.image-service
    ports:
      - "8003:8003"
    environment:
      - VISION_API_KEY=${VISION_API_KEY}
      - PYTHONUNBUFFERED=1
    networks:
      - agent-network

  transcription-service:
    build:
      context: ./backend
      dockerfile: Dockerfile.transcription-service
    ports:
      - "8004:8004"
    environment:
      - SPEECH_API_KEY=${SPEECH_API_KEY}
      - PYTHONUNBUFFERED=1
    networks:
      - agent-network

  mobile-app-backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.api-gateway
    ports:
      - "80:80"
    environment:
      - CONTEXT_SERVER_URL=http://context-server:8000
      - LLM_SERVER_URL=http://llm-server:8001
      - AGENT_ORCHESTRATOR_URL=http://agent-orchestrator:8002
      - JWT_SECRET=${JWT_SECRET}
      - PYTHONUNBUFFERED=1
    depends_on:
      - context-server
      - llm-server
      - agent-orchestrator
    networks:
      - agent-network

networks:
  agent-network:
    driver: bridge

volumes:
  mongo_data:
```

## 6. Dockerfile Examples

```dockerfile
# Dockerfile.context-server
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY context_protocol_server.py .

CMD ["python", "context_protocol_server.py"]
```

```dockerfile
# Dockerfile.llm-server
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY llm_server.py .

CMD ["python", "llm_server.py"]
```

```dockerfile
# Dockerfile.agent-orchestrator
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent_orchestrator.py .
RUN mkdir -p agents

CMD ["python", "agent_orchestrator.py"]
```

```dockerfile
# Dockerfile.image-service
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY image_analysis_service.py .

CMD ["python", "image_analysis_service.py"]
```

```dockerfile
# Dockerfile.transcription-service
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY transcription_service.py .

CMD ["python", "transcription_service.py"]
```

```dockerfile
# Dockerfile.api-gateway
FROM nginx:alpine

COPY nginx.conf /etc/nginx/nginx.conf
```

## 7. Mobile Client Implementation Guide

### Setting Up React Native Project

1. Create a new React Native project:

```bash
# Using Expo (recommended for easier setup)
npx create-expo-app MobileAIAgent
cd MobileAIAgent

# Install necessary dependencies
npm install @react-navigation/native @react-navigation/stack
npm install expo-secure-store expo-speech expo-file-system expo-image-picker
npm install @react-native-async-storage/async-storage
npm install react-native-gesture-handler react-native-screens react-native-safe-area-context
npm install axios socket.io-client
```

2. Project Structure:

```
MobileAIAgent/
├── App.js
├── app.json
├── assets/
├── babel.config.js
├── components/
│   ├── ChatBubble.js
│   ├── ChatInput.js
│   ├── MessageList.js
│   └── VoiceRecordButton.js
├── config/
│   └── index.js
├── contexts/
│   ├── AuthContext.js
│   └── AIAgentContext.js
├── hooks/
│   ├── useAudioRecording.js
│   └── useWebSocket.js
├── package.json
├── screens/
│   ├── ChatScreen.js
│   ├── LoginScreen.js
│   ├── OnboardingScreen.js
│   └── SettingsScreen.js
└── services/
    ├── authService.js
    ├── chatService.js
    ├── fileService.js
    ├── notificationService.js
    ├── secureStorageService.js
    └── sessionService.js
```

### Application Config

```javascript
// config/index.js
const DEV = __DEV__;

export const API_BASE_URL = DEV 
  ? 'http://10.0.2.2:80' // Android emulator
  : 'https://your-production-url.com';

export const WS_BASE_URL = DEV 
  ? 'ws://10.0.2.2:8000' 
  : 'wss://your-production-url.com';

export const APP_VERSION = '1.0.0';
```

### Authentication Context

```javascript
// contexts/AuthContext.js
import React, { createContext, useState, useEffect } from 'react';
import * as SecureStore from 'expo-secure-store';
import { login, register, refreshToken } from '../services/authService';

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [isLoading, setIsLoading] = useState(true);
  const [userToken, setUserToken] = useState(null);
  const [user, setUser] = useState(null);
  
  useEffect(() => {
    // Check for stored token on startup
    const bootstrapAsync = async () => {
      try {
        const storedToken = await SecureStore.getItemAsync('userToken');
        if (storedToken) {
          const userData = JSON.parse(await SecureStore.getItemAsync('userData'));
          
          // Verify token is still valid
          const refreshResult = await refreshToken(storedToken);
          if (refreshResult.success) {
            setUserToken(refreshResult.token);
            setUser(userData);
          } else {
            // Token expired, clear storage
            await SecureStore.deleteItemAsync('userToken');
            await SecureStore.deleteItemAsync('userData');
          }
        }
      } catch (e) {
        console.log('Failed to restore authentication state:', e);
      } finally {
        setIsLoading(false);
      }
    };
    
    bootstrapAsync();
  }, []);
  
  const authContext = {
    isLoading,
    userToken,
    user,
    signIn: async (email, password) => {
      setIsLoading(true);
      try {
        const response = await login(email, password);
        if (response.access_token) {
          await SecureStore.setItemAsync('userToken', response.access_token);
          await SecureStore.setItemAsync('userData', JSON.stringify({
            id: response.id,
            email: response.email,
            username: response.username
          }));
          setUserToken(response.access_token);
          setUser({
            id: response.id,
            email: response.email,
            username: response.username
          });
          return { success: true };
        }
        return { success: false, error: 'Authentication failed' };
      } catch (error) {
        return { 
          success: false, 
          error: error.response?.data?.detail || 'Authentication failed' 
        };
      } finally {
        setIsLoading(false);
      }
    },
    signUp: async (username, email, password) => {
      setIsLoading(true);
      try {
        const response = await register(username, email, password);
        if (response.access_token) {
          await SecureStore.setItemAsync('userToken', response.access_token);
          await SecureStore.setItemAsync('userData', JSON.stringify({
            id: response.id,
            email: response.email,
            username: response.username
          }));
          setUserToken(response.access_token);
          setUser({
            id: response.id,
            email: response.email,
            username: response.username
          });
          return { success: true };
        }
        return { success: false, error: 'Registration failed' };
      } catch (error) {
        return { 
          success: false, 
          error: error.response?.data?.detail || 'Registration failed' 
        };
      } finally {
        setIsLoading(false);
      }
    },
    signOut: async () => {
      setIsLoading(true);
      try {
        await SecureStore.deleteItemAsync('userToken');
        await SecureStore.deleteItemAsync('userData');
        setUserToken(null);
        setUser(null);
      } catch (e) {
        console.log('Error signing out:', e);
      } finally {
        setIsLoading(false);
      }
    }
  };
  
  return (
    <AuthContext.Provider value={authContext}>
      {children}
    </AuthContext.Provider>
  );
};
```

### AI Agent Context

```javascript
// contexts/AIAgentContext.js
import React, { createContext, useState, useContext, useRef, useEffect } from 'react';
import { AuthContext } from './AuthContext';
import { initializeSession, reconnectSession } from '../services/sessionService';
import { connectToAgentStream, sendMessage } from '../services/chatService';
import AsyncStorage from '@react-native-async-storage/async-storage';

export const AIAgentContext = createContext();

export const AIAgentProvider = ({ children }) => {
  const { userToken, user } = useContext(AuthContext);
  
  const [sessionId, setSessionId] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  
  const ws = useRef(null);
  
  // Initialize or reconnect session when auth changes
  useEffect(() => {
    if (userToken) {
      const setupSession = async () => {
        try {
          // Check for existing session
          const storedSessionId = await AsyncStorage.getItem('sessionId');
          
          if (storedSessionId) {
            // Try to reconnect
            try {
              const reconnected = await reconnectSession(userToken);
              if (reconnected) {
                setSessionId(storedSessionId);
                // Load cached messages
                const cachedMessages = await AsyncStorage.getItem('messages');
                if (cachedMessages) {
                  setMessages(JSON.parse(cachedMessages));
                }
                return;
              }
            } catch (e) {
              console.log('Failed to reconnect session:', e);
            }
          }
          
          // Initialize new session
          const newSession = await initializeSession(userToken);
          if (newSession) {
            setSessionId(newSession);
            await AsyncStorage.setItem('sessionId', newSession);
          }
        } catch (e) {
          console.log('Error setting up session:', e);
        }
      };
      
      setupSession();
    } else {
      // Clear session when logged out
      setSessionId(null);
      setMessages([]);
      setIsConnected(false);
      if (ws.current) {
        ws.current.close();
        ws.current = null;
      }
    }
  }, [userToken]);
  
  // Connect to WebSocket when sessionId changes
  useEffect(() => {
    if (sessionId && userToken) {
      connectWs();
    }
    
    // Save messages when they change
    if (messages.length > 0) {
      AsyncStorage.setItem('messages', JSON.stringify(messages));
    }
    
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [sessionId, userToken, messages]);
  
  const connectWs = () => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      return;
    }
    
    ws.current = connectToAgentStream(sessionId);
    
    ws.current.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      
      // Send auth message
      ws.current.send(JSON.stringify({
        token: userToken
      }));
    };
    
    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.error) {
          console.error('WebSocket error:', data.error);
          return;
        }
        
        if (data.type === 'message') {
          addMessage({
            id: Date.now().toString(),
            content: data.content,
            isUser: false,
            timestamp: new Date(data.timestamp || Date.now())
          });
          setIsLoading(false);
        }
      } catch (e) {
        console.error('Error processing message:', e);
      }
    };
    
    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };
    
    ws.current.onclose = () => {
      console.log('WebSocket connection closed');
      setIsConnected(false);
    };
  };
  
  const addMessage = (message) => {
    setMessages((prevMessages) => [...prevMessages, message]);
  };
  
  const sendQuery = (text) => {
    if (!sessionId || !isConnected) {
      console.error('Cannot send message: no active session or connection');
      return false;
    }
    
    const newMessage = {
      id: Date.now().toString(),
      content: text,
      isUser: true,
      timestamp: new Date()
    };
    
    addMessage(newMessage);
    setIsLoading(true);
    
    sendMessage(ws.current, {
      type: 'message',
      content: text,
      sessionId
    });
    
    return true;
  };
  
  const sendAudio = async (audioBase64) => {
    if (!sessionId || !isConnected) {
      console.error('Cannot send audio: no active session or connection');
      return false;
    }
    
    setIsLoading(true);
    
    sendMessage(ws.current, {
      type: 'audio',
      content: audioBase64,
      sessionId
    });
    
    return true;
  };
  
  const sendImage = async (imageBase64) => {
    if (!sessionId || !isConnected) {
      console.error('Cannot send image: no active session or connection');
      return false;
    }
    
    const newMessage = {
      id: Date.now().toString(),
      isUser: true,
      timestamp: new Date(),
      isImage: true,
      imageData: imageBase64
    };
    
    addMessage(newMessage);
    setIsLoading(true);
    
    sendMessage(ws.current, {
      type: 'image',
      content: imageBase64,
      sessionId
    });
    
    return true;
  };
  
  const clearConversation = async () => {
    setMessages([]);
    await AsyncStorage.removeItem('messages');
    
    // Optionally, notify the server about clearing context
    if (isConnected && ws.current) {
      sendMessage(ws.current, {
        type: 'clear_context',
        sessionId
      });
    }
  };
  
  return (
    <AIAgentContext.Provider
      value={{
        sessionId,
        isConnected,
        messages,
        isLoading,
        sendQuery,
        sendAudio,
        sendImage,
        clearConversation
      }}
    >
      {children}
    </AIAgentContext.Provider>
  );
};
```

### Main App Structure

```javascript
// App.js
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';

// Screens
import LoginScreen from './screens/LoginScreen';
import ChatScreen from './screens/ChatScreen';
import SettingsScreen from './screens/SettingsScreen';
import OnboardingScreen from './screens/OnboardingScreen';

// Contexts
import { AuthProvider, AuthContext } from './contexts/AuthContext';
import { AIAgentProvider } from './contexts/AIAgentContext';

const Stack = createStackNavigator();

const AppNavigator = () => {
  return (
    <AuthContext.Consumer>
      {({ isLoading, userToken }) => {
        if (isLoading) {
          // Show a loading screen
          return null;
        }
        
        return (
          <Stack.Navigator>
            {userToken ? (
              // User is signed in
              <>
                <Stack.Screen 
                  name="Chat" 
                  component={ChatScreen} 
                  options={{ 
                    title: 'AI Assistant',
                    headerBackTitle: 'Back',
                  }} 
                />
                <Stack.Screen 
                  name="Settings" 
                  component={SettingsScreen} 
                  options={{ title: 'Settings' }} 
                />
              </>
            ) : (
              // User is not signed in
              <>
                <Stack.Screen 
                  name="Onboarding" 
                  component={OnboardingScreen} 
                  options={{ headerShown: false }} 
                />
                <Stack.Screen 
                  name="Login" 
                  component={LoginScreen} 
                  options={{ headerShown: false }} 
                />
              </>
            )}
          </Stack.Navigator>
        );
      }}
    </AuthContext.Consumer>
  );
};

export default function App() {
  return (
    <SafeAreaProvider>
      <StatusBar style="auto" />
      <AuthProvider>
        <AIAgentProvider>
          <NavigationContainer>
            <AppNavigator />
          </NavigationContainer>
        </AIAgentProvider>
      </AuthProvider>
    </SafeAreaProvider>
  );
}
```

## 8. Implementation Steps Guide

1. **Backend Setup**:
   - Implement the Context Protocol Server
   - Set up the LLM Server with model providers
   - Build the Agent Orchestrator with sample agents
   - Create support services for image analysis and speech transcription

2. **Database Configuration**:
   - Set up MongoDB for storing sessions, context, and user data
   - Create appropriate indexes for efficient queries

3. **Docker Deployment**:
   - Create Dockerfiles for each service
   - Configure docker-compose for easy deployment
   - Set up environment variables for API keys and secrets

4. **Mobile App Development**:
   - Create React Native project structure
   - Implement authentication and session management
   - Build the chat interface with support for text, voice, and images
   - Develop offline capabilities and synchronization

5. **Testing and Optimization**:
   - Test each component individually
   - Perform integration testing
   - Optimize for mobile network conditions
   - Implement error handling and recovery mechanisms

## 9. Advanced Features to Consider

1. **Context-Aware Responses**:
   - Use device sensors and location for more relevant responses
   - Incorporate user preferences and history for personalization

2. **Offline Capabilities**:
   - Download lightweight models for basic functionality when offline
   - Queue messages and synchronize when back online

3. **Multi-Modal Interaction**:
   - Support for voice, text, and image inputs
   - Accessibility features for different user needs

4. **Security Enhancements**:
   - End-to-end encryption for sensitive conversations
   - Biometric authentication for the mobile app
   - Data retention policies and user control

5. **Performance Optimizations**:
   - Implement caching strategies for frequent queries
   - Message compression for limited bandwidth
   - Batch processing for multiple requestsRATOR_URL=http://agent-orchest
