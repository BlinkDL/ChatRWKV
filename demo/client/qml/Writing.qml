import QtQuick 2.2
import QtQuick.Controls 2.14
import QtQuick.Controls.Styles 1.4

Rectangle {
    id: rootItem
    width: parent.width
    height: parent.height

    property string outputText: ""
    property var messages: [ ]
    property int index: 0
    property bool requesting: false

    function clear() {
        textArea.text = ""
        messages.splice(0, messages.length);
        if(timer) {
            timer.start()
        }
    }

    Connections {
        target: interactive
        function onSigReceivedChatMessage(text) {
            textArea.text = ""
            let lines = text.split("\n")
            for(var item of lines){
                messages.push(item)
            }
            timer.start()
            requesting = false
        }
    }

    Rectangle {
        width: parent.width / 2 -10
        height: parent.height - 10
        anchors.top:parent.top
        anchors.topMargin : 8
        border.color: "#E5E7EB"
        border.width: 1

        CustomText {
            id: prompt
            text: "Prompt"
            height:20
            color: "#6B7280"
            anchors.top:parent.top
            anchors.topMargin : 8
            anchors.left: parent.left
            anchors.leftMargin: 10
            anchors.right: parent.right
            anchors.rightMargin: 10
        }
        MultipleLineInput {
            id: multipleLine
            width: parent.width
            height: 208
            anchors.top: prompt.bottom
            anchors.topMargin: 8
            anchors.left: parent.left
            anchors.leftMargin: 10
            anchors.right: parent.right
            anchors.rightMargin: 10
            border.width: 1
            text: "Here's a short cyberpunk sci-fi adventure story. The story's main character is an artificial human created by a company called OpenBot.

The Story:"
        }

        Rectangle {
            id:tokenCount
            anchors.top: multipleLine.bottom
            anchors.topMargin: 10
            width: parent.width
            height: 65
            border.color: "#E5E7EB"
            border.width: 1

            CustomText {
                id: tokenCountLabel
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top: parent.top
                anchors.topMargin: 10
                text: "token_count"
                height:20
            }

            NumberInput {
                id:tokenCountInput
                anchors.right: parent.right
                anchors.rightMargin: 10
                anchors.verticalCenter: tokenCountLabel.verticalCenter
                valueStep: 10
                maxValue: 200
                minValue: 10
                doubleFixed: 0
                value: 150
                onSigValueChanged: {
                    tokenCountValueSlider.blockSignal(true)
                    tokenCountValueSlider.setValue(value)
                    tokenCountValueSlider.blockSignal(false)
                    interactive.updateTokenCount(value)
                }
            }

            CustomSlider {
                id: tokenCountValueSlider
                anchors.top: tokenCountLabel.bottom
                anchors.topMargin: 10
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.right: parent.right
                anchors.rightMargin: 10

                handleBarHeight: 8
                handleHeight:16
                handleWidth: 16
                handleColor: Qt.rgba(89.0 / 255, 80.0 / 255, 249.0 / 255, 1)
                backgroundColor:Qt.rgba(89.0 / 255, 80.0 / 255, 249.0 / 255, 1)
                from: 10
                to: 200
                stepSize: 10
                value: 150
                orientation: Qt.Horizontal

                onSigValueChanged: {
                    tokenCountInput.blockSignal(true)
                    tokenCountInput.setValue(value)
                    tokenCountInput.blockSignal(false)
                    interactive.updateTokenCount(value)
                }
            }
        }

        Rectangle {
            id:temperature
            anchors.top: tokenCount.bottom
            width: parent.width
            height: 65

            CustomText {
                id: temperatureLabel
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top: parent.top
                anchors.topMargin: 10
                text: "temperature"
                height:20
            }

            NumberInput {
                id:temperatureInput
                anchors.right: parent.right
                anchors.rightMargin: 10
                anchors.verticalCenter: temperatureLabel.verticalCenter
                valueStep: 10
                maxValue: 200
                minValue: 20
                doubleFixed: 0
                value: 100
                onSigValueChanged: {
                    temperatureValueSlider.blockSignal(true)
                    temperatureValueSlider.setValue(value)
                    temperatureValueSlider.blockSignal(false)
                    interactive.updateTemperature(value)
                }
            }

            CustomSlider {
                id: temperatureValueSlider
                anchors.top: temperatureLabel.bottom
                anchors.topMargin: 10
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.right: parent.right
                anchors.rightMargin: 10

                handleBarHeight: 8
                handleHeight:16
                handleWidth: 16
                handleColor: Qt.rgba(89.0 / 255, 80.0 / 255, 249.0 / 255, 1)
                backgroundColor: Qt.rgba(89.0 / 255, 80.0 / 255, 249.0 / 255, 1)
                from: 20
                to: 200
                stepSize: 10
                value: 100
                orientation: Qt.Horizontal

                onSigValueChanged: {
                    temperatureInput.blockSignal(true)
                    temperatureInput.setValue(value)
                    temperatureInput.blockSignal(false)
                    interactive.updateTemperature(value)
                }
            }
        }

        Rectangle {
            id:topP
            anchors.top: temperature.bottom
            width: parent.width
            height: 65
            border.color: "#E5E7EB"
            border.width: 1

            CustomText {
                id: topPLabel
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top: parent.top
                anchors.topMargin: 10
                text: "top_p"
                height:20
            }

            NumberInput {
                id:topPInput
                anchors.right: parent.right
                anchors.rightMargin: 10
                anchors.verticalCenter: topPLabel.verticalCenter
                valueStep: 10
                maxValue: 100
                minValue: 0
                doubleFixed: 0
                value: 80
                onSigValueChanged: {
                    topPValueSlider.blockSignal(true)
                    topPValueSlider.setValue(value)
                    topPValueSlider.blockSignal(false)
                    interactive.updateTopP(value)
                }
            }

            CustomSlider {
                id: topPValueSlider
                anchors.top: topPLabel.bottom
                anchors.topMargin: 10
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.right: parent.right
                anchors.rightMargin: 10

                handleBarHeight: 8
                handleHeight:16
                handleWidth: 16
                handleColor: Qt.rgba(89.0 / 255, 80.0 / 255, 249.0 / 255, 1)
                backgroundColor: Qt.rgba(89.0 / 255, 80.0 / 255, 249.0 / 255, 1)
                from: 0
                to: 100
                stepSize: 10
                value: 80
                orientation: Qt.Horizontal
                onSigValueChanged: {
                    topPInput.blockSignal(true)
                    topPInput.setValue(value)
                    topPInput.blockSignal(false)
                    interactive.updateTopP(value)
                }
            }
        }


        Rectangle {
            id:presencePenalty
            anchors.top: topP.bottom
            width: parent.width
            height: 65

            CustomText {
                id: presencePenaltyLabel
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top: parent.top
                anchors.topMargin: 10
                text: "presencePenalty"
                height:20
            }

            NumberInput {
                id:presencePenaltyInput
                anchors.right: parent.right
                anchors.rightMargin: 10
                anchors.verticalCenter: presencePenaltyLabel.verticalCenter
                valueStep: 10
                maxValue: 100
                minValue: 0
                doubleFixed: 0
                value: 10
                onSigValueChanged: {
                    presencePenaltyValueSlider.blockSignal(true)
                    presencePenaltyValueSlider.setValue(value)
                    presencePenaltyValueSlider.blockSignal(false)
                    interactive.updatePresencePenalty(value)
                }
            }

            CustomSlider {
                id: presencePenaltyValueSlider
                anchors.top: presencePenaltyLabel.bottom
                anchors.topMargin: 10
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.right: parent.right
                anchors.rightMargin: 10

                handleBarHeight: 8
                handleHeight:16
                handleWidth: 16
                handleColor: Qt.rgba(89.0 / 255, 80.0 / 255, 249.0 / 255, 1)
                backgroundColor: Qt.rgba(89.0 / 255, 80.0 / 255, 249.0 / 255, 1)
                from: 0
                to: 100
                stepSize:  10
                value: 10
                orientation: Qt.Horizontal

                onSigValueChanged: {
                    presencePenaltyInput.blockSignal(true)
                    presencePenaltyInput.setValue(value)
                    presencePenaltyInput.blockSignal(false)
                    interactive.updatePresencePenalty(value)
                }
            }
        }


        Rectangle {
            id:countPenalty
            anchors.top: presencePenalty.bottom
            width: parent.width
            height: 65
            border.color: "#E5E7EB"
            border.width: 1

            CustomText {
                id: countPenaltyLabel
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top: parent.top
                anchors.topMargin: 10
                text: "countPenalty"
                height:20
            }

            NumberInput {
                id:countPenaltyInput
                anchors.right: parent.right
                anchors.rightMargin: 10
                anchors.verticalCenter: countPenaltyLabel.verticalCenter
                valueStep: 10
                maxValue: 100
                minValue: 0
                doubleFixed: 0
                value: 10
                onSigValueChanged: {
                    countPenaltyValueSlider.blockSignal(true)
                    countPenaltyValueSlider.setValue(value)
                    countPenaltyValueSlider.blockSignal(false)
                    interactive.updateCountPenalty(value)
                }
            }

            CustomSlider {
                id: countPenaltyValueSlider
                anchors.top: countPenaltyLabel.bottom
                anchors.topMargin: 10
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.right: parent.right
                anchors.rightMargin: 10
                handleBarHeight: 8
                handleHeight:16
                handleWidth: 16
                handleColor: Qt.rgba(89.0 / 255, 80.0 / 255, 249.0 / 255, 1)
                backgroundColor: Qt.rgba(89.0 / 255, 80.0 / 255, 249.0 / 255, 1)
                from: 0
                to: 100
                stepSize: 10
                value: 10
                orientation: Qt.Horizontal
                onSigValueChanged: {
                    countPenaltyInput.blockSignal(true)
                    countPenaltyInput.setValue(value)
                    countPenaltyInput.blockSignal(false)
                    interactive.updateCountPenalty(value)
                }
            }
        }

        Row {
            id: buttonRow
            anchors.top: countPenalty.bottom
            anchors.topMargin: 10
            spacing: 10
            anchors.left: parent.left
            anchors.leftMargin: 10
            anchors.right: parent.right
            anchors.rightMargin: 10

            Button {
                id: clearButton
                width: (parent.width - 10) /2
                height: 40
                text: "Clear"
                background: Rectangle{
                    color: "white"
                    border.width: 1
                    border.color: Qt.rgba(89.0 / 255, 80.0 / 255, 249.0 / 255, 1)
                }
                contentItem: Text {
                    text: clearButton.text
                    font: clearButton.font
                    color: Qt.rgba(89.0 / 255, 80.0 / 255, 249.0 / 255, 1)
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }

                onClicked: {
                    clear()
                }

            }
            Button {
                id: submitButton
                width: (parent.width - 10) /2
                height: 40
                text: {
                    if(requesting) {
                        return "Requesting"
                    } else {
                        return "Submit"
                    }
                }
                enabled: !requesting
                background: Rectangle{
                    color: Qt.rgba(89.0 / 255, 80.0 / 255, 249.0 / 255, 1)
                }
                contentItem: Text {
                    text: submitButton.text
                    font: submitButton.font
                    color: "white"
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                onClicked: {
                    if(!requesting) {
                        requesting = true
                        interactive.sendMessage(multipleLine.text, false)
                    }
                }
            }


        }

        Button {
            id: stopButton
            anchors.top: buttonRow.bottom
            anchors.topMargin: 10
            anchors.left: parent.left
            anchors.leftMargin: 10
            anchors.right: parent.right
            anchors.rightMargin: 10
            height: 40
            text: "Stop"

            background: Rectangle{
                color: Qt.rgba(253.0 / 255, 197.0 / 255, 197.0 / 255, 1)
            }
            contentItem: Text {
                text: stopButton.text
                font: stopButton.font
                color: Qt.rgba(225.0 / 255, 64.0 / 255, 69.0 / 255, 1)
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }
            onClicked: {
                timer.stop()
                messages.splice(0, messages.length);
            }
        }

    }


    Rectangle {
        width: parent.width / 2 -10
        height: parent.height - 10
        anchors.top:parent.top
        anchors.topMargin : 8
        anchors.right: parent.right
        anchors.rightMargin: 10
        border.color: "#E5E7EB"
        border.width: 1

        CustomText {
            id: outpotLabel
            text: "Output"
            height:20
            color: "#6B7280"
            anchors.top:parent.top
            anchors.topMargin : 8
            anchors.left: parent.left
            anchors.leftMargin: 10
            anchors.right: parent.right
            anchors.rightMargin: 10
        }

        TextArea {
            id: textArea
            anchors.top:outpotLabel.bottom
            anchors.topMargin : 8
            anchors.left: parent.left
            anchors.leftMargin: 10
            anchors.right: parent.right
            anchors.rightMargin: 10
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 10
            readOnly: true
            selectByMouse: true
            wrapMode: Text.WrapAnywhere
            background: Rectangle {
                border.color: "#E5E7EB"
                border.width: 1
            }
        }

        Timer {
            id: timer
            interval: 1000
            repeat: true
            onTriggered: {
                var words = messages[index].split(' ')
                for (var i = 0; i < words.length; i++) {
                    textArea.cursorPosition = textArea.text.length
                    textArea.text += words[i] + " "
                }
                textArea.cursorPosition = textArea.text.length
                textArea.text += "\n"
                index = (index + 1) % messages.length
                if (index === 0) {
                    timer.stop()
                    messages.splice(0, messages.length);
                }
            }
        }

    }

}
