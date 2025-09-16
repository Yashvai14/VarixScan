# 🌐 Vercel Frontend Deployment Guide

## 🚀 Quick Deploy Steps

### **Method 1: Vercel Dashboard (Recommended)**

1. **Go to [vercel.com/dashboard](https://vercel.com/dashboard)**
2. **Click "Add New" → "Project"**
3. **Import Git Repository:**
   - Connect your GitHub account
   - Select `VarixScan` repository 
   - Click "Import"

4. **Configure Project:**
   - **Framework Preset**: Next.js (auto-detected)
   - **Root Directory**: `.` (leave default)
   - **Build Command**: `npm run build` (auto-detected)
   - **Output Directory**: `.next` (auto-detected)

5. **Add Environment Variables:**
   Click "Environment Variables" and add:
   ```bash
   NEXT_PUBLIC_API_URL=https://varixscan.onrender.com
   NEXT_PUBLIC_SUPABASE_URL=your_supabase_project_url
   NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
   NODE_ENV=production
   ```

6. **Deploy!** 
   - Click "Deploy"
   - Wait 2-3 minutes
   - Get your live URL!

### **Method 2: Vercel CLI**

```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy (run from project root)
vercel --prod

# Follow prompts:
# Set up and deploy? Yes
# Which scope? (select your account)  
# Link to existing project? No
# What's your project's name? varixscan
# In which directory is your code located? ./
# Want to override settings? Yes
# Which framework? Next.js
# Continue? Yes
```

## 🔧 Required Environment Variables

**⚠️ CRITICAL:** You must set these in Vercel Dashboard:

```bash
# Backend API (REQUIRED)
NEXT_PUBLIC_API_URL=https://varixscan.onrender.com

# Supabase Database (REQUIRED)  
NEXT_PUBLIC_SUPABASE_URL=https://your-project-id.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key-here

# Application Settings
NODE_ENV=production
NEXT_PUBLIC_APP_NAME=VarixScan
NEXT_PUBLIC_APP_VERSION=2.0.0
```

### **Get Supabase Values:**
1. Go to [supabase.com/dashboard](https://supabase.com/dashboard)
2. Select your project → **Settings** → **API** 
3. Copy:
   - **Project URL** → `NEXT_PUBLIC_SUPABASE_URL`
   - **Anon Public Key** → `NEXT_PUBLIC_SUPABASE_ANON_KEY`

## ✅ Deployment Checklist

- [ ] Backend is live at https://varixscan.onrender.com ✅
- [ ] Vercel project created  
- [ ] Environment variables added
- [ ] Build successful
- [ ] Frontend deployed
- [ ] API calls working (test in browser network tab)

## 🧪 Test After Deployment

### 1. **Check Homepage**
Your Vercel URL should show the VarixScan landing page

### 2. **Test Backend Connection**
Open browser dev tools → Network tab → Visit dashboard
Should see API calls to `varixscan.onrender.com`

### 3. **Test Core Features**
- ✅ Landing page loads
- ✅ Navigation works
- ✅ Dashboard accessible
- ✅ API calls successful (no CORS errors)

## 🎯 Success Indicators

**✅ Deployment Successful:**
- Vercel shows "Deployment Ready"
- Your app loads at the Vercel URL
- No console errors
- API calls reach your backend

**❌ Common Issues:**
- **CORS errors**: Check backend CORS configuration
- **API not found**: Verify `NEXT_PUBLIC_API_URL` is set correctly
- **Build failures**: Check package.json scripts

## 🔗 Final URLs

- **Frontend**: https://your-app.vercel.app (will be provided after deploy)
- **Backend**: https://varixscan.onrender.com ✅
- **API Docs**: https://varixscan.onrender.com/docs

## 🎉 After Success

Your complete VarixScan platform will be live:
- **Frontend** on Vercel (global CDN, fast)
- **Backend** on Render (AI processing, database)
- **Full integration** between frontend and backend

Ready to deploy! 🚀
