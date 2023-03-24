import QtQuick 2.12

ChatItemBase {
    id: control
    Row{
        width: control.contentWidth
        layoutDirection: control.isUser?Qt.RightToLeft:Qt.LeftToRight
        Rectangle{
            id: wrap_item
            radius: 4
            width: text_item.width
            height: text_item.height
            color: control.messageBgColor

            Rectangle{
                width: 10
                height: 10
                y: control.messageHeight/2-10
                anchors.horizontalCenter: control.isUser?parent.right:parent.left
                rotation: 45
                color: control.messageBgColor
            }

            ChatLable {
                id: text_item
                text: model.text_text
                width: Math.min(control.contentWidth,textWidth)
            }
        }
    }

    Row{
        width: control.contentWidth
        layoutDirection: control.isUser?Qt.RightToLeft:Qt.LeftToRight
        ChatLable {
            text: model.datetime
            padding: 0
        }
    }
}
