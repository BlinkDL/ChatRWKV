@echo off
set shell_path=%~dp0
echo Current path is %shell_path%
cd  %shell_path%

rem http://www.angusj.com/resourcehacker/#download
cd ..
"C:\Users\shenhaoyu\Code\tool\rcedit-x64.exe" "x64\ChatRWKV.exe" --set-icon "C:\Users\shenhaoyu\Code\chatrwkv\demo\client\resource\rwkv_profile.ico"
