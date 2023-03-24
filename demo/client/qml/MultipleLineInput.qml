import QtQuick 2.14
import QtQuick.Controls 2.12
import "Style.js" as JsStyle

Rectangle {
    id: root
    property var style: JsStyle.Com_Mutiple_Input_Style
    property bool hovered: false
    property bool disable: false
    property int fontSize: 12
    property int maxCharSize: 0
    property alias text: inputTextArea.text
    property alias placeholderText: inputTextArea.placeholderText
    property alias horizontalAlignment: inputTextArea.horizontalAlignment
    property alias verticalAlignment: inputTextArea.verticalAlignment
    color: "transparent"
    width: parent.width
    height: parent.height
    border.color: getBorderColor()
    border.width: 1
    signal sigValueChanged(var value)

    Rectangle {
        id: borderRect
        color: getBackgroundColor()
        border.color: getBorderColor()
        border.width: 1
        radius: 2
        anchors.top: root.top
        anchors.bottom: root.bottom
        anchors.bottomMargin: root.maxCharSize > 0 ? 16 : 0
        anchors.left: root.left
        anchors.right: root.right
        clip: true

        ScrollView {
            id: scroll
            anchors.fill: borderRect

            TextArea {
                id: inputTextArea
                wrapMode: TextEdit.Wrap
                textFormat:TextEdit.AutoText
                selectByMouse:true
                selectByKeyboard: true
                font.family: "PingFang SC"
                font.pixelSize: fontSize
                selectionColor: style["selectionColor"]
                selectedTextColor: style["selectedTextColor"]
                clip: true
                color: disable? style["textDisableColor"] : style["textColor"]
                horizontalAlignment: Text.AlignLeft
                verticalAlignment: Text.AlignTop
                cursorVisible: inputTextArea.focus


                onEditingFinished: {
                    sigValueChanged(inputTextArea.text)
                }
                cursorDelegate:CursorDelegate {
                    delegate: inputTextArea
                }
            }
        }
    }

    CustomText {
        id: textMaxSizeText
        anchors.right: borderRect.right
        anchors.top: borderRect.bottom
        anchors.topMargin: 4
        width: maxCharSize > 0? contentWidth : 0
        height: maxCharSize > 0? 12 : 0
        textFormat:Text.RichText
        horizontalAlignment: Text.AlignRight
        verticalAlignment: Text.AlignVCenter
        text: maxCharSize > 0? getSizeStr() : ""
        textSize: 11
    }

    function getBackgroundColor() {
        if(hovered || inputTextArea.focus || disable) {
            return style["backgroundColor"]
        } else {
            return "transparent"
        }
    }

    function getSizeStr() {
        var str = '<font color="#8C8C8C">' + inputTextArea.length + '</font>'
        if(inputTextArea.length > maxCharSize) {
            str = '<font color="#FF385D">' + inputTextArea.length + '</font>'
        }
        return  str + "/" + '<font color="#8C8C8C">' + maxCharSize + '</font>'
    }

    function getBorderColor() {
        if(inputTextArea.focus) {
            if(maxCharSize > 0 && inputTextArea.length > maxCharSize) {
                return  style["charSizeOverMaxColor"]
            }else {
                return style["focusBorderColor"]
            }
        }
        return style["hoverBorderColor"]
    }

    MouseArea {
        anchors.fill: root
        hoverEnabled:true
        propagateComposedEvents:true
        onPressed: {
            mouse.accepted = false
        }
        onEntered: {
            root.hovered = true
        }
        onExited: {
            root.hovered = false
        }
    }
}



