@echo off
set shell_path=%~dp0
echo Current path is %shell_path%
cd  %shell_path%

cd ..
@echo off
set directory_name="build"

if exist %directory_name% (
    echo Directory exists. Deleting...
    rmdir /s /q %directory_name%
    echo Directory deleted.
)
echo Creating new directory...
mkdir %directory_name%
echo New directory created.

cd build
rem cmake -G "Visual Studio 16 2019" -A Win32 ..
cmake -G "Visual Studio 16 2019" -A x64 ..
cmake --build . --config Release

mkdir deploy
copy .\Release\ChatRWKV.exe deploy\
cd deploy
windeployqt --qmldir ..\..\qml  --release ChatRWKV.exe

cd %shell_path%
cd ..
mkdir x64
xcopy /E /I /Y "build\deploy" "x64"

start x64

