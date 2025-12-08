@echo off
REM Script de démarrage de l'application Streamlit - Windows
REM Usage: double-cliquer ou executer depuis le terminal

echo.
echo ====================================================
echo  Démarrage de l'application - Centrale Hydrogène
echo ====================================================
echo.

call C:\Users\owens\Documents\projet\modeling-energy-systems\.venv\Scripts\activate.bat

REM Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python n'est pas installé ou non accessible.
    echo Veuillez installer Python 3.9+ depuis https://www.python.org
    echo.
    pause
    exit /b 1
)

REM Vérifier si streamlit est installé
python -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Streamlit n'est pas installé.
    echo Installation en cours...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERREUR: L'installation des dépendances a échoué.
        echo Veuillez vérifier votre connexion Internet.
        pause
        exit /b 1
    )
)

echo Dépendances vérifiées ✓
echo.
echo Lancement de l'application...
echo.
echo L'application s'ouvrira dans votre navigateur à: http://localhost:8501
echo.
echo Appuyez sur Ctrl+C pour arrêter l'application
echo.

REM Lancer l'application
python -m streamlit run app.py

REM Maintenir la fenêtre ouverte en cas d'erreur
if errorlevel 1 (
    echo.
    echo ERREUR: L'application n'a pas pu démarrer.
    pause
    exit /b 1
)
