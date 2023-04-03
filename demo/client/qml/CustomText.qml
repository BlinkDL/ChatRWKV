import QtQuick 2.14
import "Style.js" as JsStyle

Text {
    property var style: ({})
    property var defaultStyle: JsStyle.ComText_Style
    property int textSize: JsStyle.getStyleValue(style, defaultStyle,
                                                 "textSize")
    font.pixelSize: textSize
    elide: Text.ElideRight
    color: "#6B7280"
    font.family: JsStyle.getStyleValue(style, defaultStyle, "font_family")
    font.weight: JsStyle.getStyleValue(style, defaultStyle, "font_weight")

    function setText(newText) {
        text = newText
    }

    function getText() {
        return text
    }
}
