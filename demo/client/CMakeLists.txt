cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME ChatRWKV)
project(${PROJECT_NAME})

if (APPLE)
    # Qt 5.12.0 do not support arm64 currently.
    set(CMAKE_OSX_ARCHITECTURES x86_64)
    set(OS_BUNDLE MACOSX_BUNDLE)
elseif (WIN32)
    set(OS_BUNDLE WIN32)
endif(APPLE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15")

find_package(Qt5 COMPONENTS
        Core
        Gui
        Widgets
        Quick
        QuickWidgets
        Qml
        Svg
        QmlModels
        QuickControls2
        QuickTemplates2
        REQUIRED
        Network)

qt5_add_resources(ChatRWKV_RESOURCES
        ${CMAKE_CURRENT_LIST_DIR}/qml/qml.qrc
        ${CMAKE_CURRENT_LIST_DIR}/resource/images.qrc
        )

set(EXE_NAME ChatRWKV)
add_executable(${EXE_NAME} ${OS_BUNDLE} main.cpp ${ChatRWKV_RESOURCES})
add_subdirectory(source)

target_link_libraries(${EXE_NAME} PRIVATE
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        Qt5::Network
        Qt5::Quick
        Qt5::QuickWidgets
        Qt5::Qml
        Qt5::QmlModels
        Qt5::QuickControls2
        Qt5::QuickTemplates2
        Qt5::Svg
        )



