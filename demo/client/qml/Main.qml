import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.12
import QtQuick 2.0
import QtQuick.Window 2.2

Window {
    width: 1152
    height: 720
    visible: true
    title: qsTr("ChatRWKV")
    id: mainWindow

    Rectangle {
        id:leftTab
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        color: "#5E5FF6"

        Column {
            anchors.top: parent.top
            anchors.topMargin: 10
            anchors.horizontalCenter: parent.horizontalCenter
            spacing: 30

            NewChatButton {
                id: chatButton
                onSigClickButton: {
                    console.log("Clicked chatButton")
                    interactiveWindow.switchMode(0)
                }
            }
            NewChatButton {
                id: writtingButton
                text: "+   New writting"
                onSigClickButton: {
                    console.log("Clicked writtingButton")
                    interactiveWindow.switchMode(1)
                }
            }
        }

        Component.onCompleted: {
            leftTab.width = chatButton.width + 20;
        }

    }

    InteractiveWindow {
        id:interactiveWindow
        anchors.top: parent.top
        anchors.topMargin: 10
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 10
        anchors.left: leftTab.right
        anchors.leftMargin: 10
        anchors.right: parent.right
        anchors.rightMargin: 10
    }
}





