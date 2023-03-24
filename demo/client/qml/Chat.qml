import QtQuick.Layouts 1.12
import QtQuick.Controls 1.4 as Ctrl1
import QtQuick 2.2
import QtQuick.Controls 2.4 as Ctrl2
import QtQuick.Controls 2.0
import QtQuick.Controls.Styles 1.4
import ChatModel 1.0

Rectangle {
    id: rootItem
    width: parent.width
    height: parent.height
    property bool requesting: false


    function clear() {
        chatModel.clearModel()
    }

    ChatListModel{
        id: chatModel
        onModelReset: {
            updateTimer.start();
        }
        onRowsInserted: {
            updateTimer.start();
        }
    }

    Connections {
        target: interactive
        function onSigReceivedChatMessage(text) {
            chatModel.appendText("RWKV","Self", text)
            requesting = false
        }
    }

    Timer{
        id: updateTimer
        interval: 0
        repeat: false
        onTriggered: {
            chatView.currentIndex = chatView.count - 1
        }
    }

    Ctrl1.SplitView{
        id: splitView
        anchors.fill: parent
        anchors.margins: 10
        orientation: Qt.Vertical
        handleDelegate : Rectangle {
            height: 10
        }

        Rectangle{
            Layout.fillHeight: true
            Layout.fillWidth: true
            radius: 4
            border.color: "#E5E7EB"
            color: "white"
            ChatList{
                id: chatView
                anchors.fill: parent
                anchors.margins: 10
                model: chatModel
                talkModel: chatModel
            }
        }

        Rectangle{
            id: multipleLine
            height: 160
            Layout.fillWidth: true
            radius: 4
            border.color: "#E5E7EB"

            MultipleLineInput {
                id: textArea
                anchors.fill: parent
                border.width: 1
                text: "Here's a short cyberpunk sci-fi adventure story. The story's main character is an artificial human created by a company called OpenBot.

The Story:"
            }

        }

    }
    Ctrl2.Button {
        id: clearButton
        height: 40
        width: 80
        anchors.bottom: splitView.bottom
        anchors.bottomMargin: 5
        anchors.left: splitView.left
        anchors.leftMargin: 50

        text: "Clear"
        background: Rectangle{
            color: "white"
            border.width: 1
            border.color: Qt.rgba(89 / 255, 80.0 / 255, 249.0 / 255, 1)
        }
        contentItem: Text {
            text: clearButton.text
            font: clearButton.font
            color: Qt.rgba(89.0 / 255, 80.0 / 255, 249.0 / 255, 1)
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }

        onClicked: {
            chatModel.clearModel()
        }

    }

    Ctrl2.Button {
        id: submitButton
        height: 40
        width: 80
        anchors.bottom: splitView.bottom
        anchors.bottomMargin: 5
        anchors.topMargin: 10
        anchors.right: splitView.right
        anchors.rightMargin: 50
        text:{
            if(requesting) {
                return "Requesting"
            } else {
                return "Submit"
            }
        }
        background: Rectangle{
            color: Qt.rgba(89.0 / 255, 80.0 / 255, 249.0 / 255, 1)
        }
        enabled: !requesting
        contentItem: Text {
            text: submitButton.text
            font: submitButton.font
            color: "white"
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }
        onClicked: {
            if(textArea.text.length < 1)
                return
            chatModel.appendText("self","self",textArea.text)
            if(!requesting) {
                requesting = true
                interactive.sendMessage(textArea.text, true)
            }
        }
    }
}
