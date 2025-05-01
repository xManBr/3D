@echo off
setlocal

:: Definições do projeto
set PROJECT=3d
set BASE_DIR=C:\_gerar3D\%PROJECT%
set IMAGES_PATH=%BASE_DIR%\assets
set DB_PATH=database_%PROJECT%.db
set SPARSE_DIR=sparse_%PROJECT%
set DENSE_DIR=dense_%PROJECT%

:: Limpeza
if exist %DB_PATH% del %DB_PATH%
if exist %SPARSE_DIR% rmdir /s /q %SPARSE_DIR%
if exist %DENSE_DIR% rmdir /s /q %DENSE_DIR%

echo Configurando ambiente...
set QT_PLUGIN_PATH=C:\Program Files\colmap\plugins

echo Etapa 1: Extraindo features...
colmap feature_extractor --database_path %DB_PATH% --image_path %IMAGES_PATH% --SiftExtraction.use_gpu 0
if errorlevel 1 exit /b

echo Etapa 2: Matching de imagens...
colmap exhaustive_matcher --database_path %DB_PATH% --SiftMatching.use_gpu 0
if errorlevel 1 exit /b

echo Etapa 3: Reconstruindo nuvem esparsa...
if not exist %SPARSE_DIR% mkdir %SPARSE_DIR%
colmap mapper --database_path %DB_PATH% --image_path %IMAGES_PATH% --output_path %SPARSE_DIR% --Mapper.init_min_tri_angle 4.0
if errorlevel 1 exit /b

echo Etapa 4: Corrigindo distorções...
if not exist %DENSE_DIR% mkdir %DENSE_DIR%
colmap image_undistorter --image_path %IMAGES_PATH% --input_path %SPARSE_DIR%\0 --output_path %DENSE_DIR% --output_type COLMAP
if errorlevel 1 exit /b

echo Etapa 5: Gerando mapas de profundidade...
colmap patch_match_stereo --workspace_path %DENSE_DIR% --workspace_format COLMAP --PatchMatchStereo.geom_consistency true
if errorlevel 1 exit /b

echo Etapa 6: Fundindo mapas em nuvem de pontos densa...
colmap stereo_fusion --workspace_path %DENSE_DIR% --workspace_format COLMAP --input_type geometric --output_path %DENSE_DIR%\fused.ply
if errorlevel 1 exit /b

echo Processo finalizado! A nuvem de pontos densa foi gerada em %DENSE_DIR%\fused.ply

pause


