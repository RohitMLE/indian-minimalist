# @indian.minimalist — Space Transformer

AI-powered interior design tool that transforms room photos into styled spaces with shoppable Indian products.

## Setup

### 1. Add API keys
Edit `.env` in the project root:
```
ANTHROPIC_API_KEY=your_anthropic_key
MODELSLAB_API_KEY=your_modelslab_key
```

### 2. Backend
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```

### 4. Open
```
http://localhost:5173
```

## How it works
1. Upload a room photo
2. Select a design style (Japandi, Indian Minimalist, Bohemian, etc.)
3. Set your budget
4. Click "Transform this space →"
5. AI analyses your room → generates a redesigned image → recommends shoppable products from Amazon India + Pepperfry
