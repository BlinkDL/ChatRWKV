import QtQuick 2.14

Image{
    id : comImage
    property bool hoverEnabled: false
    property bool hovered: false
    property bool bPressed: false

    signal sigClicked()
    signal sigReleased()
    signal sigPressed()

    antialiasing: true
    smooth: true
    mipmap: true

    MouseArea{
        anchors.fill: parent
        hoverEnabled: comImage.hoverEnabled

        onClicked: {
            comImage.sigClicked()
        }
        onReleased:{
            bPressed = false
            comImage.sigReleased()
        }
        onPressed: {
            bPressed = true
            comImage.sigPressed()
        }
        onEntered: {
            comImage.hovered = true
        }
        onExited: {
            comImage.hovered = false
        }
    }
}
