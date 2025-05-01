@echo off
setlocal

:: Definições do projeto
set PROJECT=lincoln_memorial_statue
set BASE_DIR=C:\_gerar3D\%PROJECT%\%PROJECT%
set IMAGES_PATH=%BASE_DIR%\assets
set DB_PATH=database_%PROJECT%.db
set SPARSE_DIR=sparse_%PROJECT%

:: Limpeza de projetos anteriores
if exist %DB_PATH% del %DB_PATH%
if exist %SPARSE_DIR% rmdir /s /q %SPARSE_DIR%

:: Criar diretórios necessários
if not exist %SPARSE_DIR% mkdir %SPARSE_DIR%

echo Configurando ambiente...
set QT_PLUGIN_PATH=C:\Program Files\colmap\plugins

echo Etapa 1: Extraindo features (feature_extractor)...
colmap feature_extractor ^
  --database_path %DB_PATH% ^
  --image_path %IMAGES_PATH% ^
  --SiftExtraction.use_gpu 0
if errorlevel 1 exit /b

echo Etapa 2: Matching de imagens (exhaustive_matcher)...
colmap exhaustive_matcher ^
  --database_path %DB_PATH% ^
  --SiftMatching.use_gpu 0
if errorlevel 1 exit /b

echo Etapa 3: Reconstruindo nuvem esparsa (mapper)...
if not exist %SPARSE_DIR% mkdir %SPARSE_DIR%
colmap mapper ^
  --database_path %DB_PATH% ^
  --image_path %IMAGES_PATH% ^
  --output_path %SPARSE_DIR% ^
  --Mapper.init_min_tri_angle 4.0
if errorlevel 1 exit /b

echo Processo de reconstrução esparsa finalizado! Verifique a pasta %SPARSE_DIR%.

pause
