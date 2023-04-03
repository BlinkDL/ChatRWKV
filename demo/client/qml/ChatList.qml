import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import ChatModel 1.0

ListView {
    id: control

    property ChatListModel talkModel

    clip: true
    headerPositioning: ListView.OverlayHeader
    footerPositioning: ListView.OverlayFooter
    boundsBehavior: Flickable.StopAtBounds

    highlightFollowsCurrentItem: true
    highlightMoveDuration: 0
    highlightResizeDuration: 0

    spacing: 10
    delegate: Loader{
        sourceComponent: textCmp
        Component{
            id: textCmp
            ChatItemText { }
        }
    }

    header: Item{
        height: 10
    }
    footer: Item{
        height: 10
    }

    ScrollBar.vertical: ScrollBar {
        id: scroll_vertical
        contentItem: Item{
            visible: (scroll_vertical.size<1.0)
            implicitWidth: 10
            Rectangle{
                anchors.centerIn: parent
                width: parent.width
                height: parent.height>20?parent.height:20
                color: (scroll_vertical.hovered||scroll_vertical.pressed)
                       ? Qt.darker("#A4ACC6")
                       : "#A4ACC6"
            }
        }
    }
}
