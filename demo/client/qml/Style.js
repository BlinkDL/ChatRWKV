.pragma library

function getSystemNormalFont() {
    switch (Qt.platform.os) {
    case "windows":
        return "Microsoft Yahei"
    case "osx":
        return "PingFang SC"
    default:
        return ""
    }
}

function platformIsWindows() {
    var curOs = Qt.platform.os
    return curOs === "windows"
}

function getFilePrefix() {
    var curOs = Qt.platform.os
    if (curOs === "windows") {
        return "file:///"
    } else {
        return "file://"
    }
}

function getStyleValue(style, defaultStyle, strKey) {
    if (style === undefined || style[strKey] === undefined)
        return defaultStyle[strKey]
    else
        return style[strKey]
}

var ComText_Style = {
    "color": "#E6E6E6",
    "textSize": 12,
    "font_family": getSystemNormalFont(),
    "font_weight": "Normal"
}

var Com_Number_Input_Style = {
    "textColor": "#6B7280",
    "selectionColor": "#E2E2E2",
    "selectedTextColor": "#060606",
    "textDisableColor": "#4F4F4F",
    "charLengthColor": "#8C8C8C",
    "charSizeOverMaxColor" : "#FF385D",
    "backgroundColor": "#1C1C1F",
    "titleTextColor": "#8a8a8a",
    "hoverBorderColor": "#E5E7EB",
    "focusBorderColor": Qt.rgba(89.0 / 255, 80.0 / 255, 249.0 / 255, 1.0),
}

var Com_Mutiple_Input_Style = {
    "hoverBorderColor": "#E5E7EB",
    "focusBorderColor": Qt.rgba(89.0 / 255, 80.0 / 255, 249.0 / 255, 1.0),
    "textColor": "black",
    "selectionColor": "#E2E2E2",
    "selectedTextColor": "#060606",
    "textDisableColor": "#4F4F4F",
    "prefixTextColor": "#595959",
    "suffixTextColor": "#8C8C8C",
    "charSizeOverMaxColor" : "#FF385D",
    "backgroundColor": "white"
}


