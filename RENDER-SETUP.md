# 🚀 Render Deployment Setup Guide

## Critical Environment Variables

**YOU MUST SET THESE IN RENDER DASHBOARD:**

### 1. Go to Render Dashboard
- Navigate to your service
- Go to **Environment** tab
- Click **Add Environment Variable**

### 2. Required Variables

```bash
# Database Configuration (REQUIRED)
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-service-key-here

# AI Configuration (REQUIRED)  
OPENAI_API_KEY=sk-your-openai-api-key-here

# Application Settings
ENVIRONMENT=production
DEBUG=false
PORT=10000
```

### 3. How to Get These Values

#### **Supabase Values:**
1. Go to [supabase.com/dashboard](https://supabase.com/dashboard)
2. Select your project
3. Go to **Settings** → **API**
4. Copy:
   - **Project URL** → Use for `SUPABASE_URL`
   - **Service Role Key** (secret) → Use for `SUPABASE_KEY`

#### **OpenAI API Key:**
1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create new key
3. Copy the key → Use for `OPENAI_API_KEY`

## Build Configuration

### Build Command:
```bash
cd backend && pip install --upgrade pip setuptools wheel && pip install --no-cache-dir -r requirements.txt
```

### Start Command:
```bash
cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Health Check Path:
```
/
```

## Deploy Steps

1. ✅ **Create Render Service**
   - Runtime: Python
   - Connect GitHub repository

2. ✅ **Set Build/Start Commands**
   - Use commands above

3. ✅ **Add Environment Variables**
   - All variables listed above
   - Double-check spelling!

4. ✅ **Deploy**
   - Click "Manual Deploy"
   - Wait for success

## Troubleshooting

### "Missing Supabase configuration" Error
- **Check**: Environment variables are set correctly
- **Verify**: `SUPABASE_URL` and `SUPABASE_KEY` are both set
- **Fix**: Go to Environment tab and add missing variables

### "OpenAI API Error"
- **Check**: `OPENAI_API_KEY` is set correctly
- **Verify**: API key is valid and has credits
- **Fix**: Generate new API key if needed

### Build Failures
- **Check**: Build command is exactly as shown above
- **Verify**: Python version is 3.11.10
- **Fix**: Ensure `backend/requirements.txt` exists

## Testing After Deployment

### 1. Health Check
```bash
curl https://your-app.onrender.com/
curl https://your-app.onrender.com/health
```

### 2. Expected Response
```json
{
  "status": "healthy",
  "database": "connected",
  "features": {
    "pdf_reports": true,
    "ai_chatbot": true,
    "image_upload": true
  }
}
```

## Success Indicators

✅ **Deployment Successful**
✅ **Health check returns 200**
✅ **Database shows "connected"**
✅ **No environment variable errors**

Your VarixScan backend is now live! 🎉
