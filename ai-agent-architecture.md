# AI Agent with Model Context Protocol Server

## System Architecture

### 1. Context Protocol Server
This is the central component that manages the communication between clients, LLMs, and agent modules.

```python
# context_protocol_server.py
import asyncio
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import json
import uuid

app = FastAPI()

class ContextSession:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.context = {}
        self.history = []
        self.active_agents = []
        self.metadata = {}
        
    def update_context(self, key, value):
        self.context[key] = value
        
    def add_to_history(self, message):
        self.history.append(message)
        
    def get_full_context(self):
        return {
            "session_id": self.session_id,
            "context": self.context,
            "history": self.history,
            "metadata": self.metadata
        }

# Global sessions store
sessions = {}

@app.post("/session/create")
async def create_session():
    session = ContextSession()
    sessions[session.session_id] = session
    return {"session_id": session.session_id}

@app.post("/context/{session_id}/update")
async def update_context(session_id: str, context_update: Dict[str, Any]):
    if session_id not in sessions:
        return {"error": "Session not found"}
    
    for key, value in context_update.items():
        sessions[session_id].update_context(key, value)
    
    return {"status": "Context updated"}

@app.get("/context/{session_id}")
async def get_context(session_id: str):
    if session_id not in sessions:
        return {"error": "Session not found"}
    
    return sessions[session_id].get_full_context()

@app.websocket("/agent/stream/{session_id}")
async def agent_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    if session_id not in sessions:
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return
    
    try:
        while True:
            data = await websocket.receive_text()
            parsed_data = json.loads(data)
            
            # Add message to history
            sessions[session_id].add_to_history(parsed_data)
            
            # Process with LLM and agents
            response = await process_with_llm_and_agents(session_id, parsed_data)
            
            # Send response back
            await websocket.send_json(response)
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

async def process_with_llm_and_agents(session_id: str, message: Dict[str, Any]):
    # This would integrate with the LLM server and agent orchestrator
    # Placeholder implementation
    context = sessions[session_id].get_full_context()
    
    # Here you'd call the LLM server with the context and message
    llm_response = {"response": "This is a placeholder LLM response"}
    
    # Update context with the new information
    sessions[session_id].update_context("last_response", llm_response)
    sessions[session_id].add_to_history({"role": "assistant", "content": llm_response["response"]})
    
    return llm_response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. LLM Server
This component interfaces with different LLM providers and handles model-specific operations.

```python
# llm_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import httpx
import os
import json

app = FastAPI()

class LLMRequest(BaseModel):
    context: Dict[str, Any]
    message: str
    model: str = "default"
    parameters: Optional[Dict[str, Any]] = None

class LLMResponse(BaseModel):
    response: str
    model_used: str
    tokens: Dict[str, int]
    metadata: Optional[Dict[str, Any]] = None

# API keys would be stored securely in environment variables
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "local": None  # No key needed for local models
}

MODEL_ENDPOINTS = {
    "gpt-4": {"provider": "openai", "endpoint": "https://api.openai.com/v1/chat/completions"},
    "claude-3-opus": {"provider": "anthropic", "endpoint": "https://api.anthropic.com/v1/messages"},
    "local-llama": {"provider": "local", "endpoint": "http://localhost:8080/generate"}
}

DEFAULT_MODEL = "gpt-4"

@app.post("/generate", response_model=LLMResponse)
async def generate_response(request: LLMRequest):
    model = request.model if request.model in MODEL_ENDPOINTS else DEFAULT_MODEL
    provider_info = MODEL_ENDPOINTS[model]
    
    # Prepare the context and message for the model
    formatted_context = format_context_for_model(request.context, model)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await call_model_api(
                client, 
                provider_info["provider"], 
                provider_info["endpoint"], 
                formatted_context, 
                request.message, 
                model, 
                request.parameters
            )
        
        processed_response = process_model_response(response, model)
        return processed_response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling LLM API: {str(e)}")

async def call_model_api(client, provider, endpoint, context, message, model, parameters):
    # Model-specific API formatting
    if provider == "openai":
        messages = format_messages_for_openai(context, message)
        payload = {
            "model": model,
            "messages": messages,
            **(parameters or {})
        }
        headers = {"Authorization": f"Bearer {API_KEYS['openai']}"}
        response = await client.post(endpoint, json=payload, headers=headers)
        
    elif provider == "anthropic":
        messages = format_messages_for_anthropic(context, message)
        payload = {
            "model": model,
            "messages": messages,
            **(parameters or {})
        }
        headers = {"x-api-key": API_KEYS['anthropic']}
        response = await client.post(endpoint, json=payload, headers=headers)
        
    elif provider == "local":
        payload = {
            "prompt": format_prompt_for_local(context, message),
            **(parameters or {})
        }
        response = await client.post(endpoint, json=payload)
    
    return response.json()

def format_context_for_model(context, model):
    # Implement context formatting logic for different models
    # This could include different ways of structuring the context based on the model's preferences
    return context

def format_messages_for_openai(context, message):
    # Format history in OpenAI's expected format
    messages = []
    
    # Add system prompt with context if available
    system_prompt = "You are a helpful AI assistant."
    if "system_prompt" in context:
        system_prompt = context["system_prompt"]
    messages.append({"role": "system", "content": system_prompt})
    
    # Add conversation history
    if "history" in context:
        for entry in context["history"]:
            messages.append(entry)
    
    # Add the current message
    messages.append({"role": "user", "content": message})
    
    return messages

def format_messages_for_anthropic(context, message):
    # Similar to OpenAI but formatted for Anthropic's API
    messages = []
    
    # Add conversation history
    if "history" in context:
        for entry in context["history"]:
            role = "user" if entry["role"] == "user" else "assistant"
            messages.append({"role": role, "content": entry["content"]})
    
    # Add the current message
    messages.append({"role": "user", "content": message})
    
    return messages

def format_prompt_for_local(context, message):
    # Format for local models that might use a different prompt structure
    prompt = ""
    
    if "history" in context:
        for entry in context["history"]:
            if entry["role"] == "user":
                prompt += f"User: {entry['content']}\n"
            else:
                prompt += f"Assistant: {entry['content']}\n"
    
    prompt += f"User: {message}\nAssistant: "
    return prompt

def process_model_response(response, model):
    # Extract and format the response from different model providers
    if model.startswith("gpt"):
        content = response["choices"][0]["message"]["content"]
        tokens = {
            "prompt_tokens": response["usage"]["prompt_tokens"],
            "completion_tokens": response["usage"]["completion_tokens"],
            "total_tokens": response["usage"]["total_tokens"]
        }
    elif model.startswith("claude"):
        content = response["content"][0]["text"]
        tokens = {
            "input_tokens": response.get("usage", {}).get("input_tokens", 0),
            "output_tokens": response.get("usage", {}).get("output_tokens", 0)
        }
    else:  # Local model
        content = response["generated_text"]
        tokens = response.get("usage", {"total_tokens": 0})
    
    return LLMResponse(
        response=content,
        model_used=model,
        tokens=tokens,
        metadata={"raw_response": response}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

### 3. Agent Orchestrator
This component manages different AI agents, their capabilities, and their interactions.

```python
# agent_orchestrator.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import httpx
import importlib
import inspect
import os
import json

app = FastAPI()

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

# Agent registry
agent_registry = {}

# Load all agents from the agents directory
def load_agents():
    agents_dir = os.path.join(os.path.dirname(__file__), "agents")
    for filename in os.listdir(agents_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            module = importlib.import_module(f"agents.{module_name}")
            
            # Look for Agent class in the module
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and hasattr(obj, "capabilities"):
                    agent = obj()
                    for capability in agent.capabilities:
                        agent_registry[capability] = agent
                    print(f"Registered agent {name} with capabilities: {agent.capabilities}")

@app.on_event("startup")
async def startup_event():
    load_agents()

@app.post("/agent/execute", response_model=AgentResponse)
async def execute_agent_action(request: AgentRequest, background_tasks: BackgroundTasks):
    if request.action not in agent_registry:
        raise HTTPException(status_code=404, detail=f"No agent found for action: {request.action}")
    
    agent = agent_registry[request.action]
    
    try:
        # Execute the agent action
        result = await agent.execute(request.action, request.context, request.parameters or {})
        
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
        
        # Update context in the context server if needed
        if response.context_updates:
            await update_context_server(request.session_id, response.context_updates)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing agent action: {str(e)}")

async def update_context_server(session_id, context_updates):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://context-server:8000/context/{session_id}/update",
            json=context_updates
        )
        if response.status_code != 200:
            print(f"Error updating context: {response.text}")

@app.get("/agent/capabilities")
async def get_capabilities():
    capabilities = {}
    for action, agent in agent_registry.items():
        agent_name = agent.__class__.__name__
        if agent_name not in capabilities:
            capabilities[agent_name] = {
                "actions": [],
                "description": agent.__class__.__doc__ or "No description"
            }
        capabilities[agent_name]["actions"].append({
            "name": action,
            "description": getattr(agent, f"{action}_description", "No description"),
            "parameters": getattr(agent, f"{action}_parameters", {})
        })
    
    return capabilities

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
```

### 4. Context Management System
This component maintains and manages context across interactions.

```python
# context_manager.py
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import json
import os
import datetime

class ContextItem(BaseModel):
    key: str
    value: Any
    source: str
    timestamp: datetime.datetime = datetime.datetime.now()
    ttl: Optional[int] = None  # Time to live in seconds, None means permanent

class ContextSnapshot(BaseModel):
    session_id: str
    timestamp: datetime.datetime
    items: Dict[str, ContextItem]

class ContextManager:
    def __init__(self, storage_dir="./context_storage"):
        self.storage_dir = storage_dir
        self.active_contexts = {}
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
    
    def create_context(self, session_id):
        """Create a new context for a session"""
        if session_id in self.active_contexts:
            return False
            
        self.active_contexts[session_id] = {}
        return True
    
    def add_to_context(self, session_id, key, value, source, ttl=None):
        """Add or update an item in the context"""
        if session_id not in self.active_contexts:
            self.create_context(session_id)
            
        self.active_contexts[session_id][key] = ContextItem(
            key=key,
            value=value,
            source=source,
            timestamp=datetime.datetime.now(),
            ttl=ttl
        )
        return True
    
    def get_from_context(self, session_id, key):
        """Get an item from the context"""
        if session_id not in self.active_contexts:
            return None
        
        item = self.active_contexts[session_id].get(key)
        if item is None:
            return None
            
        # Check if item has expired
        if item.ttl is not None:
            expiry_time = item.timestamp + datetime.timedelta(seconds=item.ttl)
            if datetime.datetime.now() > expiry_time:
                del self.active_contexts[session_id][key]
                return None
                
        return item.value
    
    def get_full_context(self, session_id):
        """Get the full context for a session"""
        if session_id not in self.active_contexts:
            return {}
            
        # Filter out expired items
        now = datetime.datetime.now()
        context = {}
        for key, item in list(self.active_contexts[session_id].items()):
            if item.ttl is None or (item.timestamp + datetime.timedelta(seconds=item.ttl)) > now:
                context[key] = item.value
            else:
                del self.active_contexts[session_id][key]
                
        return context
    
    def save_snapshot(self, session_id):
        """Save a snapshot of the current context"""
        if session_id not in self.active_contexts:
            return False
            
        snapshot = ContextSnapshot(
            session_id=session_id,
            timestamp=datetime.datetime.now(),
            items=self.active_contexts[session_id]
        )
        
        filename = f"{session_id}_{snapshot.timestamp.strftime('%Y%m%d%H%M%S')}.json"
        path = os.path.join(self.storage_dir, filename)
        
        with open(path, 'w') as f:
            f.write(snapshot.json())
            
        return True
    
    def load_snapshot(self, filepath):
        """Load a context snapshot from a file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                snapshot = ContextSnapshot(**data)
                
            session_id = snapshot.session_id
            self.active_contexts[session_id] = snapshot.items
            return session_id
        except Exception as e:
            print(f"Error loading snapshot: {e}")
            return None
    
    def merge_contexts(self, target_session_id, source_session_id, overwrite=False):
        """Merge source context into target context"""
        if target_session_id not in self.active_contexts or source_session_id not in self.active_contexts:
            return False
            
        target = self.active_contexts[target_session_id]
        source = self.active_contexts[source_session_id]
        
        for key, item in source.items():
            if overwrite or key not in target:
                target[key] = item
                
        return True
    
    def clean_expired_items(self):
        """Remove all expired items from all active contexts"""
        now = datetime.datetime.now()
        for session_id in self.active_contexts:
            for key, item in list(self.active_contexts[session_id].items()):
                if item.ttl is not None and (item.timestamp + datetime.timedelta(seconds=item.ttl)) <= now:
                    del self.active_contexts[session_id][key]
                    
        return True

# Singleton instance
context_manager = ContextManager()
```

### 5. Example Agent Implementation

```python
# agents/search_agent.py
from typing import Dict, List, Any, Optional
import httpx
import json

class SearchAgent:
    """Agent that performs various search operations"""
    
    def __init__(self):
        self.capabilities = ["web_search", "document_search", "knowledge_retrieval"]
        
        # Descriptions and parameters
        self.web_search_description = "Search the web for information"
        self.web_search_parameters = {
            "query": "The search query string",
            "num_results": "Number of results to return (default: 5)"
        }
        
        self.document_search_description = "Search through internal documents"
        self.document_search_parameters = {
            "query": "The search query string",
            "document_types": "List of document types to search",
            "max_results": "Maximum number of results to return"
        }
        
        self.knowledge_retrieval_description = "Retrieve information from knowledge base"
        self.knowledge_retrieval_parameters = {
            "topic": "The topic to retrieve information about",
            "depth": "Depth of information (basic, intermediate, advanced)"
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
        query = parameters.get("query")
        num_results = parameters.get("num_results", 5)
        
        if not query:
            return {
                "result": {"error": "No query provided"},
                "context_updates": None
            }
        
        # In a real implementation, this would call a search API
        # This is a placeholder implementation
        async with httpx.AsyncClient() as client:
            try:
                # Example using DuckDuckGo API
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json"}
                )
                search_results = response.json()
                
                # Process and format results
                formatted_results = self._format_search_results(search_results, num_results)
                
                return {
                    "result": formatted_results,
                    "context_updates": {
                        "last_search_query": query,
                        "last_search_results": formatted_results
                    }
                }
            except Exception as e:
                return {
                    "result": {"error": f"Search failed: {str(e)}"},
                    "context_updates": None
                }
    
    async def document_search(self, context, parameters):
        query = parameters.get("query")
        document_types = parameters.get("document_types", [])
        max_results = parameters.get("max_results", 10)
        
        # This would integrate with a document database or search engine
        # Placeholder implementation
        results = [
            {"title": "Example Document 1", "snippet": "This is a sample document that matches the query", "url": "https://example.com/doc1"},
            {"title": "Example Document 2", "snippet": "Another sample document with relevant information", "url": "https://example.com/doc2"}
        ]
        
        return {
            "result": results[:max_results],
            "context_updates": {
                "last_document_search": {
                    "query": query,
                    "results_count": len(results[:max_results])
                }
            }
        }
    
    async def knowledge_retrieval(self, context, parameters):
        topic = parameters.get("topic")
        depth = parameters.get("depth", "basic")
        
        # This would retrieve information from a knowledge base
        # Placeholder implementation
        knowledge = {
            "topic": topic,
            "information": f"This is {depth} information about {topic}",
            "sources": ["Knowledge Base Entry 1", "Knowledge Base Entry 2"]
        }
        
        return {
            "result": knowledge,
            "context_updates": {
                f"knowledge_{topic}": knowledge
            }
        }
    
    def _format_search_results(self, raw_results, num_results):
        """Format the raw search results into a standardized format"""
        formatted = []
        
        # Process DuckDuckGo results
        if "RelatedTopics" in raw_results:
            for topic in raw_results["RelatedTopics"][:num_results]:
                if "Text" in topic:
                    formatted.append({
                        "title": topic.get("FirstURL", "No URL"),
                        "snippet": topic["Text"],
                        "url": topic.get("FirstURL", "")
                    })
        
        return formatted
```

### 6. Client Integration Example

```javascript
// client_integration.js

// Create a session and initialize the context
async function initializeSession() {
    const response = await fetch('http://your-server/session/create', {
        method: 'POST'
    });
    const data = await response.json();
    return data.session_id;
}

// Connect to the WebSocket stream for real-time interaction
function connectToAgentStream(sessionId) {
    const ws = new WebSocket(`ws://your-server/agent/stream/${sessionId}`);
    
    ws.onopen = () => {
        console.log('Connected to agent stream');
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        // Handle the agent response
        displayResponse(data);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
        console.log('Connection closed');
    };
    
    return ws;
}

// Send a message to the agent
function sendMessage(ws, message) {
    ws.send(JSON.stringify({
        role: 'user',
        content: message
    }));
}

// Display the agent's response
function displayResponse(response) {
    // Implementation depends on your UI
    console.log('Agent response:', response);
}

// Example usage
async function startConversation() {
    const sessionId = await initializeSession();
    const ws = connectToAgentStream(sessionId);
    
    // Wait for connection to establish
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Send an initial message
    sendMessage(ws, 'Hello, I need help with analyzing this dataset.');
}

// Start the conversation
startConversation();
```

## Deployment with Docker Compose

```yaml
# docker-compose.yml
version: '3'

services:
  context-server:
    build:
      context: .
      dockerfile: Dockerfile.context-server
    ports:
      - "8000:8000"
    volumes:
      - ./context_storage:/app/context_storage
    environment:
      - PYTHONUNBUFFERED=1
    networks:
      - agent-network

  llm-server:
    build:
      context: .
      dockerfile: Dockerfile.llm-server
    ports:
      - "8001:8001"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - PYTHONUNBUFFERED=1
    networks:
      - agent-network

  agent-orchestrator:
    build:
      context: .
      dockerfile: Dockerfile.agent-orchestrator
    ports:
      - "8002:8002"
    volumes:
      - ./agents:/app/agents
    environment:
      - PYTHONUNBUFFERED=1
    depends_on:
      - context-server
      - llm-server
    networks:
      - agent-network

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - context-server
      - llm-server
      - agent-orchestrator
    networks:
      - agent-network

networks:
  agent-network:
    driver: bridge
```

## Implementation Steps

1. **Set up the basic infrastructure**:
   - Create the directory structure
   - Set up virtual environments
   - Install dependencies

2. **Build the Context Protocol Server**:
   - Implement the session management
   - Create the context update/retrieve APIs
   - Set up WebSocket for streaming

3. **Implement the LLM Server**:
   - Connect to different LLM providers
   - Handle prompt formatting and response processing
   - Implement caching for efficiency

4. **Create the Agent Orchestrator**:
   - Define the agent interface
   - Implement the agent registry
   - Create the execution pipeline

5. **Develop Basic Agents**:
   - Search agent
   - Document processing agent
   - Tool-using agent

6. **Build the Context Management System**:
   - Implement context storage and retrieval
   - Add context validation and cleaning
   - Set up TTL and expiration mechanisms

7. **Integrate Client-Side Components**:
   - Create a simple UI for testing
   - Implement WebSocket connections
   - Add message formatting and display

8. **Set up Deployment**:
   - Create Docker containers
   - Configure Docker Compose
   - Set up nginx for routing

9. **Testing and Optimization**:
   - Unit tests for each component
   - Integration tests for the full system
   - Performance optimization

## Key Features

- **Stateful Context Management**: Maintains conversation history and relevant context across interactions
- **Multi-Model Support**: Can use different LLMs based on the task requirements
- **Agent-Based Architecture**: Modular design allows adding new capabilities easily
- **Real-Time Streaming**: WebSocket-based communication for responsive interactions
- **Scalable Design**: Components can be deployed and scaled independently
