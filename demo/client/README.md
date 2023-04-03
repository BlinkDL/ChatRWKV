# Quick Start

This document guides you on how to set up and run the rwkv client on the Windows platform.

## Using the Installation Package

You can find the installation package `ChatRWKV Setup.exe` in the `Installation package` directory. Simply click on the installer and follow the instructions to complete the installation.

## Compiling from Source

### Environment Setup

If you prefer to compile the project yourself, please set up the environment first:

1. Install CMake. You can either use winget or download the installation package directly. Remember to configure the environment variable:

```
C:\Program Files\CMake\bin

```


2. Download and install Visual Studio 2019 Community:
Visit the Visual Studio official website [https://visualstudio.microsoft.com/zh-hans/vs/older-downloads/ ](https://visualstudio.microsoft.com/zh-hans/vs/older-downloads/)and download the Visual Studio 2019 Community edition. Run the installer and follow the prompts. Ensure that you select the "Desktop development with C++" component during the installation process to install the required C++ compiler and libraries. Configure the environment variable:

```
C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64

```


3. Download and install Qt 5.15.2:
Visit the Qt official website [https://www.qt.io/download](https://www.qt.io/download) and download the Qt installer for Windows. Run the installer and follow the prompts. In the component selection step, ensure that you select Qt 5.15.2 and the "MSVC 2019 64-bit" component compatible with Visual Studio 2019. Configure the environment variable:

```
C:\Qt\5.15.2\msvc2019_64\bin

```


### Compilation

Run the `build_windows.bat` script located in the `tools` directory. The compiled output will be located in the `x64` directory. To run the application, simply open `ChatRWKV.exe`.

### Packaging

If you want to package the executable file, you can use `Inno Setup` for packaging.

1. To change the application icon, you can use the `win_replace_icon.bat` script in the `tools` directory. You will need to download and install `rcedit-x64.exe` from [http://www.angusj.com/resourcehacker/#download ](http://www.angusj.com/resourcehacker/#download )first. If you do not need to replace the icon, you can ignore this step.

2. Install the required software by downloading it from here: [https://jrsoftware.org/isinfo.php](https://jrsoftware.org/isinfo.php)

3. Run the `win_package.bat` script located in the `tools` directory. Before running, please modify the relevant directories in the `win_setup.iss` script to match your own.

## Development

After setting up the environment, you can use either Qt Creator or Visual Studio 2019 to open the `CMakeLists.txt` file as a project for local development.
