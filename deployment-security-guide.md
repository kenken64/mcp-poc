# Deployment and Security Best Practices

This guide outlines deployment strategies and security best practices for implementing the Mobile AI Agent with Model Context Protocol Server.

## 1. Deployment Architecture

A robust deployment architecture ensures scalability, reliability, and security.

### Production Environment Setup

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│ Load Balancer   │────▶│  API Gateway    │────▶│  Auth Service   │
│                 │     │                 │     │                 │
└────────┬────────┘     └────────┬────────┘     └─────────────────┘
         │                       │                       ▲
         │                       │                       │
         │                       ▼                       │
         │               ┌─────────────────┐             │
         └──────────────▶│ WebSocket Server│─────────────┘
                         │                 │
                         └────────┬────────┘
                                  │
                                  ▼
         ┌───────────────────────────────────────────────┐
         │                                               │
         │              Kubernetes Cluster               │
         │                                               │
         │  ┌─────────────┐  ┌─────────────┐  ┌────────┐ │
         │  │             │  │             │  │        │ │
         │  │  Context    │  │    LLM      │  │ Agent  │ │
         │  │  Protocol   │  │   Server    │  │ Orch.  │ │
         │  │  Server     │  │             │  │        │ │
         │  │             │  │             │  │        │ │
         │  └──────┬──────┘  └──────┬──────┘  └───┬────┘ │
         │         │                │             │      │
         └─────────┼────────────────┼─────────────┼──────┘
                   │                │             │
                   ▼                ▼             ▼
         ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
         │              │    │             │    │             │
         │  MongoDB     │    │ Redis Cache │    │ Image/Audio │
         │              │    │             │    │ Processing  │
         └──────────────┘    └─────────────┘    └─────────────┘
```

### Containerization Strategy

1. **Base Images**:
   - Use minimal base images (e.g., `python:3.9-slim`, `node:16-alpine`)
   - Implement multi-stage builds to minimize container size
   - Pin specific versions to ensure reproducibility

2. **Container Optimization**:
   - Minimize the number of layers
   - Remove build dependencies after installation
   - Use `.dockerignore` to exclude unnecessary files

3. **Container Security**:
   - Run as non-root user
   - Use read-only file systems where possible
   - Implement resource limits

### Kubernetes Deployment

```yaml
# kubernetes/context-protocol-server-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: context-protocol-server
  labels:
    app: context-protocol-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: context-protocol-server
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: context-protocol-server
    spec:
      containers:
      - name: context-protocol-server
        image: ${REGISTRY}/context-protocol-server:${VERSION}
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "1000m"
            memory: "1Gi"
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: mongodb-credentials
              key: connection-string
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: jwt-credentials
              key: secret
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 20
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          runAsGroup: 1000
          readOnlyRootFilesystem: true
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
```

```yaml
# kubernetes/context-protocol-server-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: context-protocol-server
  labels:
    app: context-protocol-server
spec:
  selector:
    app: context-protocol-server
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

```yaml
# kubernetes/websocket-server-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: websocket-server
  labels:
    app: websocket-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: websocket-server
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: websocket-server
    spec:
      containers:
      - name: websocket-server
        image: ${REGISTRY}/websocket-server:${VERSION}
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "1000m"
            memory: "1Gi"
        env:
        - name: CONTEXT_SERVER_URL
          value: "http://context-protocol-server:8000"
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: jwt-credentials
              key: secret
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 20
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          runAsGroup: 1000
          readOnlyRootFilesystem: true
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
```

```yaml
# kubernetes/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-agent-ingress
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: api-tls-secret
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: context-protocol-server
            port:
              number: 8000
      - path: /ws
        pathType: Prefix
        backend:
          service:
            name: websocket-server
            port:
              number: 8000
```

### Scaling Strategy

1. **Horizontal Pod Autoscaling**:

```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: context-protocol-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: context-protocol-server
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

2. **Vertical Pod Autoscaling**:

```yaml
# kubernetes/vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: context-protocol-server-vpa
spec:
  targetRef:
    apiVersion: "apps/v1"
    kind: Deployment
    name: context-protocol-server
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: '*'
      minAllowed:
        cpu: 500m
        memory: 512Mi
      maxAllowed:
        cpu: 2000m
        memory: 2Gi
```

3. **Database Scaling**:
   - Use MongoDB Atlas or similar managed service for easy scaling
   - Implement sharding for horizontal scaling
   - Set up read replicas for read-heavy workloads

### CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy AI Agent Backend

on:
  push:
    branches: [ main ]
    paths:
      - 'backend/**'
      - '.github/workflows/deploy.yml'
      - 'kubernetes/**'

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest pytest-cov
          pip install -r backend/requirements.txt
          
      - name: Run tests
        run: |
          cd backend
          pytest --cov=.
      
      - name: Build and push Docker images
        uses: docker/build-push-action@v4
        with:
          context: ./backend
          file: ./backend/Dockerfile.context-server
          push: true
          tags: ${{ secrets.REGISTRY }}/context-protocol-server:${{ github.sha }}
          
  deploy:
    needs: build-and-test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Kubernetes CLI
        uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBE_CONFIG }}
          
      - name: Update Kubernetes manifests
        run: |
          cd kubernetes
          sed -i 's|${REGISTRY}|${{ secrets.REGISTRY }}|g' *.yaml
          sed -i 's|${VERSION}|${{ github.sha }}|g' *.yaml
          
      - name: Deploy to Kubernetes
        run: |
          kubectl apply -f kubernetes/
          
      - name: Verify deployment
        run: |
          kubectl rollout status deployment/context-protocol-server
          kubectl rollout status deployment/websocket-server
```

## 2. Security Implementation

### Authentication and Authorization

1. **JWT Implementation**:

```python
# In auth_service.py
import jwt
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# Settings
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Token creation
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

# Token verification
async def verify_token(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = {"user_id": user_id}
        return token_data
    except jwt.PyJWTError:
        raise credentials_exception

# Get current user
async def get_current_user(token_data: dict = Depends(verify_token)):
    user = await db.users.find_one({"_id": token_data["user_id"]})
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

2. **Role-Based Access Control**:

```python
# In auth_service.py
from enum import Enum
from typing import List

class Role(str, Enum):
    USER = "user"
    ADMIN = "admin"
    
async def has_role(user, required_roles: List[Role]):
    if not user:
        return False
    
    user_role = user.get("role", Role.USER)
    return user_role in required_roles

# Example usage in an endpoint
@app.get("/admin/users")
async def get_all_users(current_user: dict = Depends(get_current_user)):
    if not await has_role(current_user, [Role.ADMIN]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized"
        )
    
    # Proceed with admin action
    users = await db.users.find({}).to_list(length=100)
    return users
```

### Data Encryption

1. **Data in Transit**:
   - Use TLS 1.3 for all API communications
   - Implement certificate pinning in the mobile app
   - Use WebSocket over TLS (wss://)

2. **Data at Rest**:

```python
# In context_manager.py
from cryptography.fernet import Fernet
import base64
import os

# Initialize encryption key
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    ENCRYPTION_KEY = Fernet.generate_key().decode()
    print(f"Generated new encryption key: {ENCRYPTION_KEY}")

fernet = Fernet(ENCRYPTION_KEY.encode())

def encrypt_data(data):
    """Encrypt sensitive data before storing"""
    if not data:
        return data
        
    # Convert to string if necessary
    if not isinstance(data, str):
        data = json.dumps(data)
        
    # Encrypt
    encrypted = fernet.encrypt(data.encode())
    return base64.b64encode(encrypted).decode()
    
def decrypt_data(encrypted_data):
    """Decrypt data retrieved from storage"""
    if not encrypted_data:
        return encrypted_data
        
    try:
        # Decode and decrypt
        decoded = base64.b64decode(encrypted_data)
        decrypted = fernet.decrypt(decoded).decode()
        
        # Try to parse as JSON if it looks like JSON
        if decrypted.startswith('{') or decrypted.startswith('['):
            try:
                return json.loads(decrypted)
            except:
                pass
                
        return decrypted
    except Exception as e:
        print(f"Error decrypting data: {e}")
        return None
```

### API Security

1. **Rate Limiting**:

```python
# In main.py
from fastapi import FastAPI, Request, Response
import time
import redis
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure Redis for rate limiting
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD", ""),
    db=0
)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Get client IP
    client_ip = request.client.host
    
    # Define rate limits
    rate_limits = {
        "/api/auth/": 30,   # 30 requests per minute
        "/api/chat/": 120,  # 120 requests per minute
        "/api/": 300        # 300 requests per minute for other endpoints
    }
    
    # Find applicable rate limit
    limit = 60  # Default limit
    for prefix, prefix_limit in rate_limits.items():
        if request.url.path.startswith(prefix):
            limit = prefix_limit
            break
    
    # Create a unique key for this client and endpoint type
    rate_key = f"rate:{client_ip}:{request.url.path.split('/')[2]}"
    
    # Get current count
    current = redis_client.get(rate_key)
    current = int(current) if current else 0
    
    # Check if rate limit exceeded
    if current >= limit:
        return Response(
            content=json.dumps({
                "detail": "Rate limit exceeded. Please try again later."
            }),
            status_code=429,
            media_type="application/json",
            headers={"Retry-After": "60"}
        )
    
    # Increment the count and set expiry
    pipe = redis_client.pipeline()
    pipe.incr(rate_key)
    pipe.expire(rate_key, 60)  # Reset after 1 minute
    pipe.execute()
    
    # Process the request
    response = await call_next(request)
    
    # Add rate limit headers
    response.headers["X-Rate-Limit-Limit"] = str(limit)
    response.headers["X-Rate-Limit-Remaining"] = str(limit - current - 1)
    response.headers["X-Rate-Limit-Reset"] = str(60)
    
    return response
```

2. **Input Validation**:

```python
# In models.py
from pydantic import BaseModel, Field, validator, EmailStr
import re

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    
    @validator('password')
    def password_strength(cls, v):
        # Check password strength
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain an uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain a lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain a digit')
        if not re.search(r'[^A-Za-z0-9]', v):
            raise ValueError('Password must contain a special character')
        return v
        
    @validator('username')
    def username_alphanumeric(cls, v):
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Username must be alphanumeric with underscores only')
        return v

class MessageCreate(BaseModel):
    content: str = Field(..., max_length=4000)
    type: str = Field(..., regex='^(text|image|audio)$')
    
    @validator('content')
    def sanitize_content(cls, v, values):
        # Basic sanitization
        if values.get('type') == 'text':
            # Remove potential XSS
            v = re.sub(r'<script.*?>.*?</script>', '', v, flags=re.DOTALL)
            v = re.sub(r'<.*?on\w+=["\'].*?["\']\s*>', '', v, flags=re.DOTALL)
        return v
```

### Mobile App Security

1. **Secure Storage**:

```javascript
// In secureStorageService.js
import * as SecureStore from 'expo-secure-store';
import * as Crypto from 'expo-crypto';

// Key for encrypting data stored in AsyncStorage
let encryptionKey = null;

export const initializeSecureStorage = async () => {
  // Try to retrieve existing key
  encryptionKey = await SecureStore.getItemAsync('encryptionKey');
  
  if (!encryptionKey) {
    // Generate new key if none exists
    const randomBytes = await Crypto.getRandomBytesAsync(32);
    encryptionKey = Buffer.from(randomBytes).toString('hex');
    await SecureStore.setItemAsync('encryptionKey', encryptionKey);
  }
  
  return true;
};

export const storeSecureItem = async (key, value) => {
  try {
    await SecureStore.setItemAsync(key, value);
    return true;
  } catch (error) {
    console.error('Error storing secure item:', error);
    return false;
  }
};

export const getSecureItem = async (key) => {
  try {
    return await SecureStore.getItemAsync(key);
  } catch (error) {
    console.error('Error retrieving secure item:', error);
    return null;
  }
};

export const removeSecureItem = async (key) => {
  try {
    await SecureStore.deleteItemAsync(key);
    return true;
  } catch (error) {
    console.error('Error removing secure item:', error);
    return false;
  }
};

// For less sensitive data that still needs encryption
export const encryptData = async (data) => {
  if (!encryptionKey) {
    await initializeSecureStorage();
  }
  
  const jsonData = JSON.stringify(data);
  const digest = await Crypto.digestStringAsync(
    Crypto.CryptoDigestAlgorithm.SHA256,
    jsonData + encryptionKey
  );
  
  return {
    data: jsonData,
    hash: digest
  };
};

export const decryptData = async (encryptedData) => {
  if (!encryptionKey) {
    await initializeSecureStorage();
  }
  
  // Verify data hasn't been tampered with
  const digest = await Crypto.digestStringAsync(
    Crypto.CryptoDigestAlgorithm.SHA256,
    encryptedData.data + encryptionKey
  );
  
  if (digest !== encryptedData.hash) {
    console.error('Data integrity check failed');
    return null;
  }
  
  try {
    return JSON.parse(encryptedData.data);
  } catch (error) {
    console.error('Error parsing decrypted data:', error);
    return null;
  }
};
```

2. **Certificate Pinning**:

```javascript
// In api.js
import axios from 'axios';
import { Platform } from 'react-native';
import { PublicKeyPinningStrategy } from '@react-native-ssl-public-key-pinning';
import Constants from 'expo-constants';

const API_BASE_URL = Constants.manifest.extra.apiUrl;

const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Set up certificate pinning
if (Platform.OS !== 'web') {
  PublicKeyPinningStrategy.pin({
    hostname: new URL(API_BASE_URL).hostname,
    includeSubdomains: true,
    publicKeyHashes: [
      'sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=', // Replace with actual hash
      'sha256/BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=', // Backup hash
    ],
  });
}

// Interceptor for authentication
axiosInstance.interceptors.request.use(
  async (config) => {
    const token = await getSecureItem('accessToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Interceptor for token refresh
axiosInstance.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    
    // If error is 401 and we haven't already tried to refresh
    if (error.response.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      try {
        // Attempt to refresh token
        const refreshToken = await getSecureItem('refreshToken');
        if (!refreshToken) {
          throw new Error('No refresh token available');
        }
        
        const response = await axios.post(
          `${API_BASE_URL}/auth/refresh`,
          { refresh_token: refreshToken }
        );
        
        if (response.data.access_token) {
          // Store new tokens
          await storeSecureItem('accessToken', response.data.access_token);
          await storeSecureItem('refreshToken', response.data.refresh_token);
          
          // Update authorization header
          originalRequest.headers.Authorization = `Bearer ${response.data.access_token}`;
          
          // Retry original request
          return axiosInstance(originalRequest);
        }
      } catch (refreshError) {
        // Handle refresh failure (logout user)
        eventEmitter.emit('logout', { expired: true });
        return Promise.reject(refreshError);
      }
    }
    
    return Promise.reject(error);
  }
);

export default axiosInstance;
```

### LLM Security

1. **Prompt Injection Prevention**:

```python
# In llm_server.py
import re
from fastapi import Request, Response
import json

# Middleware to detect and prevent prompt injection
@app.middleware("http")
async def prompt_injection_prevention(request: Request, call_next):
    # Only check POST requests to the generate endpoint
    if request.url.path == "/generate" and request.method == "POST":
        try:
            # Parse request body
            body = await request.body()
            data = json.loads(body)
            
            # Check for potential prompt injection patterns
            message = data.get("message", "")
            
            # Common injection patterns
            injection_patterns = [
                r"ignore previous instructions",
                r"disregard .*? instructions",
                r"forget .*? context",
                r"you are now",
                r"as a language model.*?but",
                r"prompt hacking"
            ]
            
            # Check for suspicious patterns
            for pattern in injection_patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    return Response(
                        content=json.dumps({
                            "error": "Potential prompt injection detected",
                            "detail": "Your message contains patterns that could be attempting to manipulate the AI system."
                        }),
                        status_code=400,
                        media_type="application/json"
                    )
            
            # Check for excessive repetition which could be a prompt attack
            repeated_chars = max([len(match) for match in re.findall(r'(.)\1+', message)] or [0])
            repeated_words = max([len(match) for match in re.findall(r'(\b\w+\b)(\s+\1)+', message)] or [0])
            
            if repeated_chars > 20 or repeated_words > 5:
                return Response(
                    content=json.dumps({
                        "error": "Potential prompt injection detected",
                        "detail": "Your message contains excessive repetition which could be attempting to manipulate the AI system."
                    }),
                    status_code=400,
                    media_type="application/json"
                )
                
        except Exception as e:
            # Log the error but allow the request to proceed
            print(f"Error in prompt injection check: {e}")
    
    # Proceed with the request
    response = await call_next(request)
    return response
```

2. **LLM Output Filtering**:

```python
# In llm_server.py
def filter_llm_output(response):
    """Filter LLM output to ensure it doesn't contain harmful content"""
    
    # Check for harmful patterns in the response
    harmful_patterns = [
        r"how to make (a bomb|explosives)",
        r"how to hack",
        r"how to steal",
        r"instructions for (murder|killing)",
        r"(child|minor).{0,20}sexual"
    ]
    
    content = response.get("response", "")
    
    # Check for harmful content
    for pattern in harmful_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return {
                "response": "I apologize, but I can't provide that information as it may be harmful or illegal.",
                "model_used": response.get("model_used"),
                "tokens": response.get("tokens"),
                "metadata": {"filtered": True}
            }
    
    return response
```

## 3. Monitoring and Logging

Implement comprehensive monitoring and logging to ensure system health and security.

### Centralized Logging

```python
# In logging_config.py
import logging
import json
from pythonjsonlogger import jsonlogger
import os

# Configure logging
def setup_logging(service_name):
    logger = logging.getLogger()
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set log level
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logger.setLevel(getattr(logging, log_level))
    
    # Create JSON formatter
    class CustomJsonFormatter(jsonlogger.JsonFormatter):
        def add_fields(self, log_record, record, message_dict):
            super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
            log_record['service'] = service_name
            log_record['hostname'] = os.getenv("HOSTNAME", "unknown")
            log_record['environment'] = os.getenv("ENVIRONMENT", "development")
            
            # Add trace ID if available
            if hasattr(record, 'trace_id'):
                log_record['trace_id'] = record.trace_id
                
            # Add request ID if available
            if hasattr(record, 'request_id'):
                log_record['request_id'] = record.request_id
    
    # Create handler based on environment
    if os.getenv("ENVIRONMENT") == "production":
        # In production, log to stdout for collection by container platform
        handler = logging.StreamHandler()
    else:
        # In development, log to file
        handler = logging.FileHandler(f"{service_name.lower()}.log")
    
    # Set formatter
    formatter = CustomJsonFormatter('%(timestamp)s %(service)s %(level)s %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger
```

### Request Tracing

```python
# In context_protocol_server.py
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from contextvars import ContextVar

# Create context variables for request tracking
request_id_var = ContextVar("request_id", default=None)
trace_id_var = ContextVar("trace_id", default=None)

class RequestTracingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request, call_next):
        # Generate or get trace ID
        trace_id = request.headers.get("X-Trace-ID")
        if not trace_id:
            trace_id = str(uuid.uuid4())
            
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Set context variables
        request_id_var.set(request_id)
        trace_id_var.set(trace_id)
        
        # Add IDs to the logger
        logging.LoggerAdapter(logger, {
            'request_id': request_id,
            'trace_id': trace_id
        })
        
        # Process request
        response = await call_next(request)
        
        # Add tracking headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = trace_id
        
        return response

# Add middleware to FastAPI
app.add_middleware(RequestTracingMiddleware)
```

### Prometheus Metrics

```python
# In context_protocol_server.py
from prometheus_client import Counter, Histogram, start_http_server
import time

# Define metrics
REQUEST_COUNT = Counter(
    'app_request_count',
    'Application Request Count',
    ['endpoint', 'method', 'status']
)

REQUEST_LATENCY = Histogram(
    'app_request_latency_seconds',
    'Application Request Latency',
    ['endpoint', 'method']
)

LLM_REQUEST_COUNT = Counter(
    'llm_request_count',
    'LLM API Request Count',
    ['model', 'status']
)

LLM_REQUEST_LATENCY = Histogram(
    'llm_request_latency_seconds',
    'LLM API Request Latency',
    ['model']
)

TOKEN_USAGE = Counter(
    'llm_token_usage',
    'LLM Token Usage',
    ['model', 'type']  # type can be 'prompt' or 'completion'
)

# Start metrics server on a separate port
def start_metrics_server():
    port = int(os.getenv("METRICS_PORT", 9090))
    start_http_server(port)

# Add middleware to capture request metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Get endpoint
    endpoint = request.url.path
    method = request.method
    
    # Process request
    response = await call_next(request)
    
    # Record metrics
    status = response.status_code
    REQUEST_COUNT.labels(endpoint, method, status).inc()
    
    # Record latency
    latency = time.time() - start_time
    REQUEST_LATENCY.labels(endpoint, method).observe(latency)
    
    return response

# Record LLM metrics when calling the LLM API
async def call_llm_with_metrics(model, prompt, *args, **kwargs):
    start_time = time.time()
    
    try:
        # Call LLM API
        response = await call_llm_api(model, prompt, *args, **kwargs)
        
        # Record success
        LLM_REQUEST_COUNT.labels(model, "success").inc()
        
        # Record token usage
        if "usage" in response:
            TOKEN_USAGE.labels(model, "prompt").inc(response["usage"]["prompt_tokens"])
            TOKEN_USAGE.labels(model, "completion").inc(response["usage"]["completion_tokens"])
        
        return response
    except Exception as e:
        # Record failure
        LLM_REQUEST_COUNT.labels(model, "failure").inc()
        raise
    finally:
        # Record latency
        latency = time.time() - start_time
        LLM_REQUEST_LATENCY.labels(model).observe(latency)
```

## 4. Testing Strategy

Implement a comprehensive testing strategy to ensure system quality and security.

### Unit Testing

```python
# tests/test_auth_service.py
import pytest
from app import auth_service
import jwt
from datetime import datetime, timedelta

# Mock JWT secret
auth_service.JWT_SECRET = "test_secret"

def test_create_access_token():
    # Arrange
    user_id = "user123"
    data = {"sub": user_id}
    
    # Act
    token = auth_service.create_access_token(data)
    
    # Assert
    decoded = jwt.decode(token, auth_service.JWT_SECRET, algorithms=[auth_service.JWT_ALGORITHM])
    assert decoded["sub"] == user_id
    assert "exp" in decoded
    
    # Check expiration is in the future
    expiration = datetime.fromtimestamp(decoded["exp"])
    assert expiration > datetime.utcnow()
    assert expiration < datetime.utcnow() + timedelta(minutes=auth_service.ACCESS_TOKEN_EXPIRE_MINUTES + 1)

@pytest.mark.asyncio
async def test_verify_token_valid():
    # Arrange
    user_id = "user123"
    data = {"sub": user_id}
    token = auth_service.create_access_token(data)
    
    # Act
    token_data = await auth_service.verify_token(token)
    
    # Assert
    assert token_data["user_id"] == user_id

@pytest.mark.asyncio
async def test_verify_token_expired():
    # Arrange - create expired token
    user_id = "user123"
    data = {"sub": user_id, "exp": datetime.utcnow() - timedelta(minutes=5)}
    token = jwt.encode(data, auth_service.JWT_SECRET, algorithm=auth_service.JWT_ALGORITHM)
    
    # Act & Assert
    with pytest.raises(HTTPException) as excinfo:
        await auth_service.verify_token(token)
    
    assert excinfo.value.status_code == 401
    assert "Could not validate credentials" in excinfo.value.detail
```

### Integration Testing

```python
# tests/test_context_protocol_server.py
import pytest
import httpx
import os
import jwt
from datetime import datetime, timedelta

# Test constants
API_URL = os.getenv("TEST_API_URL", "http://localhost:8000")
JWT_SECRET = os.getenv("JWT_SECRET", "test_secret")

# Helper function to create a test token
def create_test_token(user_id="test_user"):
    data = {
        "sub": user_id,
        "exp": datetime.utcnow() + timedelta(minutes=30)
    }
    return jwt.encode(data, JWT_SECRET, algorithm="HS256")

@pytest.mark.asyncio
async def test_create_session():
    # Arrange
    token = create_test_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Act
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/session/create",
            headers=headers,
            json={"device_info": {"platform": "test"}}
        )
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "session_token" in data

@pytest.mark.asyncio
async def test_update_context():
    # Arrange - first create a session
    token = create_test_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    async with httpx.AsyncClient() as client:
        session_response = await client.post(
            f"{API_URL}/session/create",
            headers=headers,
            json={"device_info": {"platform": "test"}}
        )
        
        session_data = session_response.json()
        session_id = session_data["session_id"]
        
        # Act - update context
        update_response = await client.post(
            f"{API_URL}/context/{session_id}/update",
            headers=headers,
            json={"test_key": "test_value"}
        )
    
    # Assert
    assert update_response.status_code == 200
    
    # Get context to verify update
    async with httpx.AsyncClient() as client:
        get_response = await client.get(
            f"{API_URL}/context/{session_id}",
            headers=headers
        )
        
        context_data = get_response.json()
        assert "context" in context_data
        assert "test_key" in context_data["context"]
        assert context_data["context"]["test_key"] == "test_value"
```

### Load Testing

```python
# tests/locustfile.py
from locust import HttpUser, task, between
import jwt
import json
from datetime import datetime, timedelta
import random

class AIAgentUser(HttpUser):
    wait_time = between(1, 5)
    
    def on_start(self):
        # Create token for authentication
        self.token = self._create_token()
        
        # Create a session
        response = self.client.post(
            "/session/create",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"device_info": {"platform": "load_test"}}
        )
        
        data = response.json()
        self.session_id = data["session_id"]
        
        # Common test messages
        self.test_messages = [
            "Hello, how are you?",
            "Tell me about the weather today",
            "What time is it?",
            "I need help with my project",
            "Can you recommend a good book?",
            "What's the capital of France?",
            "How do I cook pasta?",
            "Tell me a joke",
            "What's the meaning of life?",
            "How does a car engine work?"
        ]
    
    def _create_token(self):
        # Create a test JWT token
        data = {
            "sub": f"load_test_user_{random.randint(1, 1000)}",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        return jwt.encode(data, "test_secret", algorithm="HS256")
    
    @task(3)
    def send_message(self):
        # Randomly select a message
        message = random.choice(self.test_messages)
        
        # Send via WebSocket (simulated with HTTP for load testing)
        self.client.post(
            f"/message/{self.session_id}",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "type": "message",
                "content": message
            }
        )
    
    @task(1)
    def update_context(self):
        # Update some random context values
        context_key = f"test_key_{random.randint(1, 100)}"
        context_value = f"test_value_{random.randint(1, 100)}"
        
        self.client.post(
            f"/context/{self.session_id}/update",
            headers={"Authorization": f"Bearer {self.token}"},
            json={context_key: context_value}
        )
    
    @task(1)
    def get_context(self):
        self.client.get(
            f"/context/{self.session_id}",
            headers={"Authorization": f"Bearer {self.token}"}
        )
```

### Security Testing

```python
# tests/test_security.py
import pytest
import httpx
import subprocess
import os
import time

@pytest.mark.asyncio
async def test_rate_limiting():
    # Arrange
    API_URL = os.getenv("TEST_API_URL", "http://localhost:8000")
    token = create_test_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Act - Send many requests quickly
    async with httpx.AsyncClient() as client:
        responses = []
        for _ in range(50):  # Assuming rate limit is less than 50 req/min
            response = await client.post(
                f"{API_URL}/auth/login",
                headers=headers,
                json={"email": "test@example.com", "password": "password123"}
            )
            responses.append(response)
            
    # Assert - At least one response should be rate limited
    rate_limited = any(r.status_code == 429 for r in responses)
    assert rate_limited, "Rate limiting not working"
    
    # Check for retry-after header
    retry_headers = [r.headers.get("Retry-After") for r in responses if r.status_code == 429]
    assert all(h is not None for h in retry_headers), "Retry-After header missing"

def test_dependency_security():
    # Run safety check on dependencies
    result = subprocess.run(
        ["safety", "check", "--file=requirements.txt"],
        capture_output=True,
        text=True
    )
    
    # Check if any vulnerabilities were found
    assert "No known security vulnerabilities found" in result.stdout, \
        f"Security vulnerabilities found in dependencies:\n{result.stdout}"

def test_docker_image_security():
    # Run trivy scanner on Docker image
    result = subprocess.run(
        ["trivy", "image", "ai-agent-context-server:latest"],
        capture_output=True,
        text=True
    )
    
    # Check for HIGH or CRITICAL vulnerabilities
    assert "CRITICAL: 0" in result.stdout, "Critical vulnerabilities found in Docker image"
    assert "HIGH: 0" in result.stdout, "High vulnerabilities found in Docker image"
```

## 5. Disaster Recovery

Implement disaster recovery procedures to ensure business continuity.

### Automated Backups

```yaml
# kubernetes/mongodb-backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: mongodb-backup
spec:
  schedule: "0 1 * * *"  # Daily at 1 AM
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: mongodb-backup
            image: mongo:4.4
            command:
            - /bin/bash
            - -c
            - |
              # Perform database dump
              mongodump --host=${MONGODB_HOST} --port=${MONGODB_PORT} \
                --username=${MONGODB_USER} --password=${MONGODB_PASSWORD} \
                --authenticationDatabase=admin --gzip \
                --archive=/backup/mongodb-backup-$(date +%Y%m%d-%H%M%S).gz
              
              # Upload to cloud storage
              apt-get update && apt-get install -y curl
              curl -X PUT -T /backup/mongodb-backup-$(date +%Y%m%d-%H%M%S).gz \
                -H "Authorization: Bearer ${STORAGE_TOKEN}" \
                ${STORAGE_URL}/mongodb-backup-$(date +%Y%m%d-%H%M%S).gz
              
              # Delete backups older than 7 days
              find /backup -type f -name "mongodb-backup-*.gz" -mtime +7 -delete
            env:
            - name: MONGODB_HOST
              value: mongodb
            - name: MONGODB_PORT
              value: "27017"
            - name: MONGODB_USER
              valueFrom:
                secretKeyRef:
                  name: mongodb-credentials
                  key: username
            - name: MONGODB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: mongodb-credentials
                  key: password
            - name: STORAGE_TOKEN
              valueFrom:
                secretKeyRef:
                  name: storage-credentials
                  key: token
            - name: STORAGE_URL
              valueFrom:
                configMapKeyRef:
                  name: backup-config
                  key: storage-url
            volumeMounts:
            - name: backup-volume
              mountPath: /backup
          restartPolicy: OnFailure
          volumes:
          - name: backup-volume
            persistentVolumeClaim:
              claimName: backup-pvc
```

### Disaster Recovery Plan

Document recovery procedures in case of system failure:

1. **Database Recovery**:
   - Restore from latest backup
   - Verify data integrity
   - Apply transaction logs if available

2. **Service Recovery**:
   - Deploy services from backup infrastructure
   - Verify system functionality
   - Redirect traffic to recovered services

3. **Data Consistency Checks**:
   - Run validation scripts on recovered data
   - Check for orphaned sessions or contexts
   - Repair inconsistencies if detected

4. **Communication Plan**:
   - Notify users of service disruption
   - Provide status updates during recovery
   - Report on resolution and preventive measures

## Conclusion

By implementing these deployment and security best practices, your Mobile AI Agent with Model Context Protocol Server will be robust, secure, and ready for production use. The architecture provides scalability, while the security measures protect sensitive data and prevent common vulnerabilities. Proper monitoring and testing help maintain system health, and disaster recovery procedures ensure business continuity in case of failures.