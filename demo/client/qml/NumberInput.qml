import QtQuick 2.14
import "Style.js" as JsStyle

Rectangle {
    id: root
    property var style: JsStyle.Com_Number_Input_Style
    property bool hovered: false
    property bool disable: false
    property int fontSize: 12

    property double value: 0
    property double minValue: 0
    property double maxValue: 100
    property double valueStep: 1
    property int doubleFixed: 0
    property double oriValue: 0
    property alias inputValidator: textInput.validator
    property alias horizontalAlignment: textInput.horizontalAlignment
    property alias verticalAlignment: textInput.verticalAlignment
    property alias textFocus: textInput.focus
    color: 'transparent'
    border.width: getBorderWidth()
    border.color: getBorderColor()
    radius: 2
    width: 120
    height: 24
    property int blockFlag: 0
    signal sigValueChanged(double value, double preValue, bool pressed)

    function setValue(value){
        if(value === root.value){
            return
        }
        root.value = value
        if(blockFlag === 0){
            sigValueChanged(value, oriValue, false)
        }
        oriValue = root.value
    }

    onValueChanged: {
        textInput.text = value.toFixed(doubleFixed)
    }

    MouseArea {
        anchors.fill: root
        hoverEnabled:true
        propagateComposedEvents:true
        onPressed: {
            mouse.accepted = true
        }
        onEntered: {
            root.hovered = true
        }
        onExited: {
            root.hovered = false
        }
    }

    TextInput {
        id:textInput
        property double lastValue: 0
        anchors.left: root.left
        anchors.leftMargin: 8
        anchors.top: root.top
        anchors.bottom: root.bottom
        width: parent.width - 16
        font.family: "PingFang SC"
        font.pixelSize: fontSize
        selectionColor: style["selectionColor"]
        selectedTextColor: style["selectedTextColor"]
        clip: true
        activeFocusOnPress: !disable
        selectByMouse:!disable
        color: disable? style["textDisableColor"] : style["textColor"]
        horizontalAlignment: Text.AlignLeft
        verticalAlignment: Text.AlignVCenter
        text: root.value

        onFocusChanged: {
            if(focus){
                oriValue = Number(textInput.text)
            } else if(text === "" || isNaN(Number(textInput.text))) {
                root.value = root.minValue
                sigValueChanged(Number(textInput.text), oriValue, false)
                textInput.text = root.minValue.toFixed(doubleFixed)
                return
            }
            if (oriValue === Number(textInput.text)) {
                return
            }
            let localValue = parseFloat(text)
            if(localValue > root.maxValue) {
                root.value = root.maxValue
            } else if(localValue < root.minValue) {
                root.value = root.minValue
            } else {
                root.value = localValue
            }
            textInput.text = root.value.toFixed(doubleFixed)
            sigValueChanged(Number(textInput.text), oriValue, false)
        }
        Keys.onPressed: {
            if(event.key === Qt.Key_Enter || event.key === Qt.Key_Return) {
                root.forceActiveFocus()
            }
        }

        cursorDelegate:CursorDelegate {
            delegate: textInput
        }
    }

    Rectangle {
        anchors.top: root.top
        anchors.topMargin: 1
        anchors.bottom: root.bottom
        anchors.bottomMargin: 1
        anchors.right: splitter.left
        width: getBorderWidth() > 0? 1 : 0
        color: style["hoverBorderColor"]
    }

    Rectangle {
        id:increase
        property bool hovered: false
        property bool pressed: false
        anchors.right: root.right
        anchors.top: root.top
        width: getBorderWidth() > 0? 16 : 0
        color: 'transparent'
        height: getIncreaseHeight()

        Image {
            visible: increase.width > 0
            anchors.centerIn: increase
            width: 6
            height: 6
            source: getIncreaseIcon()
        }

        MouseArea {
            anchors.fill: increase
            hoverEnabled:true
            propagateComposedEvents: true
            onPressed: {
                mouse.accepted = true
                increase.pressed = true
                let text = textInput.text
                if(text === "" || isNaN(Number(text))) {
                    root.value = root.minValue
                    textInput.text = root.minValue.toFixed(doubleFixed)
                    if(value < maxValue) {
                        value += valueStep
                    }
                    return
                }
                let localValue = parseFloat(text)
                if(localValue > root.maxValue) {
                    root.value = root.maxValue
                } else if(localValue < root.minValue) {
                    root.value = root.minValue
                } else {
                    root.value = localValue
                }
                textInput.text = root.value.toFixed(doubleFixed)
                let originalValue = root.value
                if(value < maxValue) {
                    value += valueStep
                }
                sigValueChanged(value, originalValue, false)

            }
            onReleased: {
                increase.pressed = false
            }
            onEntered: {
                increase.hovered = true
            }
            onExited: {
                increase.hovered = false
            }
        }

    }

    Rectangle {
        id: splitter
        anchors.top: increase.bottom
        anchors.right: root.right
        anchors.rightMargin: 1
        height: 1
        width: getBorderWidth() > 0? 16 : 0
        color: style["hoverBorderColor"]
    }

    Rectangle {
        id:decrease
        property bool hovered: false
        property bool pressed: false
        anchors.right: root.right
        anchors.bottom: root.bottom
        width: getBorderWidth() > 0? 16 : 0
        color: 'transparent'
        height:getDecreaseHeight()
        Image {
            visible: decrease.width > 0
            anchors.centerIn: decrease
            width: 6
            height: 6
            source: getDecreaseIcon()
        }

        MouseArea {
            anchors.fill: decrease
            hoverEnabled:true
            propagateComposedEvents: true
            onPressed: {
                mouse.accepted = true
                decrease.pressed = true
                let text = textInput.text
                if(text === "" || isNaN(Number(text))) {
                    root.value = root.maxValue
                    textInput.text = root.value.toFixed(doubleFixed)
                    if(value > minValue) {
                        value -= valueStep
                    }
                    return
                }
                let localValue = parseFloat(text)
                if(localValue > root.maxValue) {
                    root.value = root.maxValue
                } else if(localValue < root.minValue) {
                    root.value = root.minValue
                } else {
                    root.value = localValue
                }
                textInput.text = root.value.toFixed(doubleFixed)
                let originalValue = root.value
                if(value > minValue) {
                    value -= valueStep
                }
                sigValueChanged(value, originalValue, false)
            }
            onReleased: {
                decrease.pressed = false
            }
            onEntered: {
                decrease.hovered = true
            }
            onExited: {
                decrease.hovered = false
            }
        }

    }

    function getIncreaseHeight() {
        if(increase.hovered) {
            return 14
        } else if(decrease.hovered){
            return 9
        } else {
            return 11.5
        }
    }

    function getDecreaseHeight() {
        if(decrease.hovered) {
            return 14
        } else if(increase.hovered){
            return 9
        } else {
            return 11.5
        }
    }

    function getIncreaseIcon() {
        if(Math.abs(value - maxValue) < 0.00000001) {
            return  "qrc:/number_input_up_disable.png"
        } else if(increase.pressed) {
            return  "qrc:/number_input_up_normal.png"
        } else if(increase.hovered) {
            return  "qrc:/number_input_up_light.png"
        }
        return "qrc:/number_input_up_normal.png"
    }
    function getDecreaseIcon() {
        if(Math.abs(value - minValue)  < 0.00000001) {
            return  "qrc:/number_input_down_disable.png"
        } else  if(decrease.pressed) {
            return  "qrc:/number_input_down_normal.png"
        } else if(decrease.hovered) {
            return  "qrc:/number_input_down_light.png"
        }
        return  "qrc:/number_input_down_normal.png"
    }

    function getBorderWidth() {
        return (((root.hovered || textInput.focus || increase.hovered || decrease.hovered) && !root.disable)?  1 : 0)
    }

    function getBorderColor() {
        if(textInput.focus) {
            return style["focusBorderColor"]
        }
        return style["hoverBorderColor"]
    }

    function blockSignal(block){
        if(block){
            blockFlag++
        }else{
            blockFlag--
        }
    }
}



