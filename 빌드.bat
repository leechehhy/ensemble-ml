@echo off
cd /d %~dp0
echo ============================================
echo  앙상블 ML 앱 빌드 시작
echo ============================================
echo.

echo [1/2] PyInstaller 설치 확인 중...
pip install pyinstaller --quiet
echo.

echo [2/2] 앱 빌드 중 (패키지 크기에 따라 3~10분 소요)...
pyinstaller --noconfirm --onedir --noconsole ^
  --add-data "templates;templates" ^
  --add-data "static;static" ^
  --hidden-import sklearn.utils._cython_blas ^
  --hidden-import sklearn.neighbors._partition_nodes ^
  --hidden-import sklearn.tree._utils ^
  --hidden-import xgboost ^
  --hidden-import catboost ^
  --hidden-import catboost.core ^
  --name "앙상블ML" ^
  app.py

echo.
if exist "dist\앙상블ML\앙상블ML.exe" (
    echo ============================================
    echo  빌드 성공!
    echo  배포 폴더: dist\앙상블ML\
    echo  이 폴더 전체를 zip으로 압축해서 배포하세요.
    echo  사용자는 앙상블ML.exe 를 더블클릭하면 됩니다.
    echo ============================================
) else (
    echo  빌드 실패. 위 오류 메시지를 확인하세요.
)
echo.
pause
