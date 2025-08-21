@echo off
echo ========================================
echo Setting up Environment Variables
echo ========================================

echo Creating frontend environment file...
if not exist .env.local (
    copy env.example .env.local
    echo ✅ Created .env.local
) else (
    echo ⚠️  .env.local already exists
)

echo.
echo Creating backend environment file...
cd backend
if not exist .env (
    copy env.example .env
    echo ✅ Created backend/.env
) else (
    echo ⚠️  backend/.env already exists
)
cd ..

echo.
echo ========================================
echo Environment setup completed!
echo ========================================
echo.
echo 📝 Next steps:
echo 1. Edit .env.local to configure frontend
echo 2. Edit backend/.env to configure backend
echo 3. Start the services:
echo    - Backend: cd backend && python start_server.py
echo    - Frontend: npm run dev
echo.
pause 