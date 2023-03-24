import QtQuick 2.12
import QtQuick.Layouts 1.12

Item {
    id: control
    width: parent.ListView.view.width
    height: Math.max(profileHeight,
                     content_item.height)

    property int profileHeight: 48
    property int messageHeight: 48

    property int leftWidth: 110
    property int rightWidth: 110
    property int contentWidth: Math.max(10,control.width-control.leftWidth-control.rightWidth)
    property bool isUser: (model.user === model.sender)
    property color messageBgColor: control.isUser? Qt.rgba(205 / 255.0, 202 / 255.0, 253 / 255, 1) : "#E5E7EB"
    default property alias contentItems: content_item.children

    Item{
        id: left_item
        height: control.height
        width: control.leftWidth
        Image {
            visible: !control.isUser
            width: control.profileHeight
            height: control.profileHeight
            anchors.horizontalCenter: parent.horizontalCenter
            source:  "qrc:/rwkv_profile.png"
        }
    }

    Column{
        id: content_item
        x: control.leftWidth
        width: control.contentWidth
        spacing: 6
    }
    Item {
        id: right_item
        height: control.height
        width: control.rightWidth
        anchors.right: parent.right
        Image {
            visible: control.isUser
            width: control.profileHeight
            height: control.profileHeight
            anchors.horizontalCenter: parent.horizontalCenter
            source: "qrc:/profile_user_gray.png"
        }
    }
}
