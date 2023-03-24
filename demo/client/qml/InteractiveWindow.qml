import QtQuick 2.14
import QtQuick.Controls 2.2
import "Style.js" as JsStyle

Rectangle{
    id: interactiveWin
    width: 240
    height: 250
    clip: true
    color: "white"
    property int currentIndex: 0

    Component.onCompleted: {
        paramAreaLoader.sourceComponent = chatComponent
    }

    function switchMode (index) {
        var item = paramAreaLoader.item
        if (item) {
            item.clear()
        }
        if(index === 0) {
            paramAreaLoader.sourceComponent = chatComponent
        } else {
            paramAreaLoader.sourceComponent = writtingComponent
        }
        currentIndex = index
    }

    Loader {
        id: paramAreaLoader
        anchors.fill: parent
        Component {
            id: chatComponent
            Chat {
                id: chat
            }
        }

        Component {
            id: writtingComponent
            Writing{
                id:writting
            }
        }

    }
}
