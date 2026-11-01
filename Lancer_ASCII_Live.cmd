@echo off
cd /d "%~dp0"
echo ASCII Flux live : chargement du modele (environ 20 s), le navigateur s'ouvrira tout seul.
echo Fermer cette fenetre arrete le serveur.
"%~dp0.venv-amd\Scripts\python.exe" "%~dp0scripts\live.py"
