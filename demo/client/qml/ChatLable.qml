import QtQuick 2.12
import QtQuick.Controls 2.12

TextEdit {
    id: control
    property int textWidth: private_text.implicitWidth
    padding: 14
    font{
        pixelSize: 14
    }
    color: "#666666"

    verticalAlignment: TextEdit.AlignVCenter
    horizontalAlignment: TextEdit.AlignLeft
    readOnly: true

    selectByMouse: true
    selectionColor: "black"
    selectedTextColor: "white"
    wrapMode: TextEdit.WrapAnywhere
    Text{
        id: private_text
        visible: false
        font: control.font
        padding: control.padding
        text: control.text
    }
}
