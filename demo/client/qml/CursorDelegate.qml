import QtQuick 2.0


Rectangle {
    id: cursor
    property var delegate
    visible: delegate.cursorVisible
    color:  "#4938FF"
    width: delegate.cursorRectangle.width
    height: delegate.cursorRectangle.height
    SequentialAnimation {
        loops: Animation.Infinite
        running: delegate.cursorVisible
        PropertyAction {
            target: cursor
            property: 'visible'
            value: true
        }
        PauseAnimation {
            duration: 600
        }
        PropertyAction {
            target: cursor
            property: 'visible'
            value: false
        }
        PauseAnimation {
            duration: 600
        }

        onStopped: {
            cursor.visible = false
        }
    }
}

