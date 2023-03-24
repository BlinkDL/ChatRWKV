import QtQuick 2.0
import QtQuick.Controls 2.0
import "Style.js" as JsStyle

Rectangle {
    id: comButton
    width: getWidth()
    height: style.height === undefined ? defaultStyle.height : style.height
    color: getBackgroundColor(bPressed, bHovered, bEnable)
    radius: style.borderRadius === undefined ? 4 : style.borderRadius
    border.width: style.borderWidth === undefined ? 0 : style.borderWidth
    border.color: getBorderColor(bPressed, bHovered, bEnable)
    property var style: JsStyle.ComButton_Style_Black_M
    property var defaultStyle: JsStyle.ComButton_Style_Blue_M
    property bool bHovered: false
    property bool bPressed: false
    property bool bEnable: true
    property string text: ""
    property string iconPath: ""
    property int iconWidth: 0
    property int iconHeight: 0
    property int iconAndTextSpacing: 2
    property string textColor: getTextColor(bPressed, bHovered, bEnable)
    property real textSize: 14
    property alias propagateComposedEvents: rootMouseArea.propagateComposedEvents
    signal sigClicked

    CustomImage {
        id: iconCom
        source: iconPath === "" ? "" : "qrc" + iconPath
        width: iconWidth
        height: iconHeight
        anchors.verticalCenter: parent.verticalCenter
        visible: iconPath !== ""
        anchors.left: parent.left
        anchors.leftMargin: style.paddingLeft === undefined ? 0 : style.paddingLeft
    }

    CustomText {
        id: btnText
        width: contentWidth
        anchors.verticalCenter: parent.verticalCenter
        anchors.left: iconCom.visible ? iconCom.right : parent.left
        anchors.leftMargin: {
            if(iconCom.visible) {
                return iconAndTextSpacing
            }
            if(comButton.style.paddingLeft === undefined) {
                return 0
            }
            return comButton.style.paddingLeft
        }

        text: comButton.text
        textSize: comButton.textSize
        elide: Text.ElideNone
        color: textColor
    }

    MouseArea {
        id: rootMouseArea
        anchors.fill: parent
        hoverEnabled: true
        propagateComposedEvents: true

        onEntered: {
            comButton.bHovered = true
        }
        onExited: {
            comButton.bHovered = false
        }

        onClicked: {
            if (!comButton.bEnable)
                return
            comButton.sigClicked()
        }
        onPressed: {

            comButton.bPressed = true
        }
        onReleased: {
            comButton.bPressed = false
        }
    }

    function getBackgroundColor(pressed, hovered, enable){
        if(enable === false){
            return style.disableColor === undefined ? defaultStyle.disableColor : style.disableColor
        }else if(pressed === true){
            return style.pressColor === undefined ? defaultStyle.pressColor : style.pressColor
        }else if(hovered === true){
            return style.hoverColor === undefined ? defaultStyle.hoverColor : style.hoverColor
        }

        return style.color === undefined ? defaultStyle.color : style.color
    }

    function getTextColor(pressed, hovered, enable){
        if(enable === false){
            return style.textDisableColor === undefined ? defaultStyle.textDisableColor : style.textDisableColor
        }else if(pressed === true){
            return style.textPressColor === undefined ? defaultStyle.textPressColor : style.textPressColor
        }else if(hovered === true){
            return style.textHoverColor === undefined ? defaultStyle.textHoverColor : style.textHoverColor
        }

        return style.textColor === undefined ? defaultStyle.textColor : style.textColor
    }

    function getBorderColor(pressed, hovered, enable){
        if(enable === false){
            return style.borderDisableColor === undefined ? "transparent" : style.borderDisableColor
        }else if(pressed === true){
            return style.borderPressColor === undefined ?"transparent" : style.borderPressColor
        }else if(hovered === true){
            return style.borderHoverColor === undefined ? "transparent" : style.borderHoverColor
        }

        return style.borderColor === undefined ? "transparent" : style.borderColor
    }

    function getWidth() {
        var newWidth = btnText.contentWidth
        if(style.paddingLeft !== undefined) {
            newWidth += style.paddingLeft
        }
        if(style.paddingRight !== undefined) {
            newWidth += style.paddingRight
        }
        if(iconPath !== "") {
            newWidth += iconWidth + iconAndTextSpacing
        }
        return newWidth
    }

}
