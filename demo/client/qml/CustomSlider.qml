import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.impl 2.12
import QtQuick.Templates 2.12 as T

T.Slider {
    id: control
    property bool acceptWheel: true
    property color handleColor: "red"
    property color backgroundColor: "cadetblue"
    property int handleWidth: 10
    property int handleHeight: 10
    property int handleBarHeight: 2
    height: handleBarHeight > handleHeight ? handleBarHeight : handleHeight
    signal sigValueChanged(double oriValue, double value, bool changing)
    property int blockFlag: 0
    property double oriValue: 0
    onValueChanged: {
        if(blockFlag === 0) {
            sigValueChanged(oriValue, value, true)
        }
    }

    onPressedChanged: {
        if(value === oriValue){
            return
        }

        if(pressed === false && blockFlag === 0) {
            sigValueChanged(oriValue, value, false)
        }else if(pressed){
            oriValue = value
        }
    }

    handle: Rectangle{
        width: control.handleWidth
        height: control.handleHeight
        radius: width / 2
        color: control.handleColor
        x: control.leftPadding + control.visualPosition * (control.availableWidth - width)
        y: control.topPadding + (control.availableHeight - height) / 2
    }

    background: Rectangle {
        x: control.leftPadding
        y: control.topPadding + (control.availableHeight - height) / 2
        width: control.availableWidth
        height: control.availableHeight
        color: "transparent"

        Rectangle {
            width: control.availableWidth
            height: control.handleBarHeight
            radius: control.availableWidth / 2
            anchors.centerIn: parent
            color: control.backgroundColor
            scale: 1
        }
    }

    function setValue(value){
        control.value = value
    }

    function blockSignal(block){
        if(block){
            blockFlag++
        }else{
            blockFlag--
        }
    }
}
