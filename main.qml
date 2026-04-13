import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import QtQuick.Dialogs
import Qt5Compat.GraphicalEffects

ApplicationWindow {
    id: mainWindow
    title: "BUK-EmotionAI Studio (2 Emotions)"
    width: 1366
    height: 768
    minimumWidth: 1200
    minimumHeight: 600
    visible: true
    color: "#0F172A"

    // Modern Color Palette - Dark Professional Theme
    property color primaryColor: "#1E293B"
    property color secondaryColor: "#334155"
    property color accentColor: "#3B82F6"
    property color successColor: "#10B981"
    property color warningColor: "#F59E0B"
    property color errorColor: "#EF4444"
    property color surfaceColor: "#1E293B"
    property color cardColor: "#1E293B"
    property color textPrimary: "#F1F5F9"
    property color textSecondary: "#94A3B8"
    property color borderColor: "#334155"
    property color gradientStart: "#3B82F6"
    property color gradientEnd: "#8B5CF6"

    // Fonts
    property string fontFamily: "Segoe UI"
    property int fontSizeSmall: 12
    property int fontSizeMedium: 14
    property int fontSizeLarge: 16
    property int fontSizeXLarge: 20
    property int fontSizeXXLarge: 28

    // Properties
    property int faceCount: 0
    property var probabilities: ({})
    property real currentConfidence: 0
    property string currentEmotion: "happy"

    // Helper functions - MODIFIED for 2 emotions
    function getEmotionColor(emotion) {
        var colors = {
            "happy": "#F59E0B",  // Orange/Yellow
            "sad": "#3B82F6",    // Blue
        }
        return colors[emotion] || "#64748B"
    }

    function getEmoji(emotion) {
        var emojis = {
            "happy": "😊",
            "sad": "😢",
        }
        return emojis[emotion] || "❓"
    }

    // Models
    ListModel { id: detectionModel }
    ListModel { id: cameraModel }
    ListModel { id: statsModel }

    // Header with Gradient
    Rectangle {
        id: header
        width: parent.width
        height: 70
        color: "transparent"

        LinearGradient {
            anchors.fill: parent
            start: Qt.point(0, 0)
            end: Qt.point(parent.width, 0)
            gradient: Gradient {
                GradientStop { position: 0.0; color: "#1E293B" }
                GradientStop { position: 1.0; color: "#0F172A" }
            }
        }

        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 24
            anchors.rightMargin: 24
            spacing: 20

            // Logo and Title
            Row {
                spacing: 12
                Layout.alignment: Qt.AlignLeft

                Rectangle {
                    width: 40
                    height: 40
                    radius: 10
                    color: accentColor
                    gradient: Gradient {
                        GradientStop { position: 0.0; color: gradientStart }
                        GradientStop { position: 1.0; color: gradientEnd }
                    }

                    Text {
                        anchors.centerIn: parent
                        text: "😊"
                        font.pixelSize: 20
                        color: "white"
                    }
                }

                Column {
                    spacing: 2
                    anchors.verticalCenter: parent.verticalCenter

                    Text {
                        text: "BUK-EmotionAI Studio"
                        font.family: fontFamily
                        font.pixelSize: fontSizeXLarge
                        font.bold: true
                        color: textPrimary
                    }

                    Text {
                        text: "Real-time 2-Emotion Detection (Happy/Sad)"
                        font.family: fontFamily
                        font.pixelSize: fontSizeSmall
                        color: textSecondary
                    }
                }
            }

            Item { Layout.fillWidth: true }

            // Status Indicators
            Row {
                spacing: 16
                Layout.alignment: Qt.AlignRight

                // FPS Indicator
                Rectangle {
                    width: 120
                    height: 50
                    radius: 12
                    color: primaryColor
                    border.color: borderColor
                    border.width: 1

                    Row {
                        anchors.centerIn: parent
                        spacing: 12

                        Rectangle {
                            width: 36
                            height: 36
                            radius: 18
                            color: accentColor
                            anchors.verticalCenter: parent.verticalCenter

                            Text {
                                anchors.centerIn: parent
                                text: "⚡"
                                font.pixelSize: 16
                                color: "white"
                            }
                        }

                        Column {
                            spacing: 2
                            anchors.verticalCenter: parent.verticalCenter

                            Text {
                                text: "FPS"
                                font.family: fontFamily
                                font.pixelSize: fontSizeSmall
                                color: textSecondary
                            }

                            Text {
                                id: fpsLabel
                                text: "0"
                                font.family: fontFamily
                                font.pixelSize: fontSizeLarge
                                font.bold: true
                                color: textPrimary
                            }
                        }
                    }
                }

                // Face Count Indicator
                Rectangle {
                    width: 120
                    height: 50
                    radius: 12
                    color: primaryColor
                    border.color: borderColor
                    border.width: 1
                    visible: backend && backend.is_camera_active

                    Row {
                        anchors.centerIn: parent
                        spacing: 12

                        Rectangle {
                            width: 36
                            height: 36
                            radius: 18
                            color: successColor
                            anchors.verticalCenter: parent.verticalCenter

                            Text {
                                anchors.centerIn: parent
                                text: "👤"
                                font.pixelSize: 16
                                color: "white"
                            }
                        }

                        Column {
                            spacing: 2
                            anchors.verticalCenter: parent.verticalCenter

                            Text {
                                text: "Faces"
                                font.family: fontFamily
                                font.pixelSize: fontSizeSmall
                                color: textSecondary
                            }

                            Text {
                                text: faceCount
                                font.family: fontFamily
                                font.pixelSize: fontSizeLarge
                                font.bold: true
                                color: textPrimary
                            }
                        }
                    }
                }

                // Recording Indicator
                Rectangle {
                    width: 120
                    height: 50
                    radius: 12
                    color: primaryColor
                    border.color: borderColor
                    border.width: 1

                    Row {
                        anchors.centerIn: parent
                        spacing: 12

                        Rectangle {
                            width: 36
                            height: 36
                            radius: 18
                            color: backend && backend.is_recording ? errorColor : textSecondary
                            anchors.verticalCenter: parent.verticalCenter

                            Text {
                                anchors.centerIn: parent
                                text: backend && backend.is_recording ? "🔴" : "⚫"
                                font.pixelSize: 16
                                color: "white"
                            }
                        }

                        Column {
                            spacing: 2
                            anchors.verticalCenter: parent.verticalCenter

                            Text {
                                text: "Recording"
                                font.family: fontFamily
                                font.pixelSize: fontSizeSmall
                                color: textSecondary
                            }

                            Text {
                                text: backend && backend.is_recording ? "ON" : "OFF"
                                font.family: fontFamily
                                font.pixelSize: fontSizeLarge
                                font.bold: true
                                color: backend && backend.is_recording ? errorColor : textPrimary
                            }
                        }
                    }
                }
            }
        }

        // Separator line
        Rectangle {
            anchors.bottom: parent.bottom
            width: parent.width
            height: 1
            color: borderColor
            opacity: 0.5
        }
    }

    // Main Content Area - Using RowLayout for proper panel management
    Rectangle {
        id: contentArea
        anchors.top: header.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        color: "transparent"

        RowLayout {
            anchors.fill: parent
            spacing: 0

            // Left Panel - Camera
            Rectangle {
                id: leftPanel
                Layout.preferredWidth: 900
                Layout.fillHeight: true
                color: "transparent"

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 20
                    spacing: 20

                    // Camera Feed Card
                    Rectangle {
                        id: cameraCard
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        radius: 16
                        color: cardColor
                        border.color: borderColor
                        border.width: 1

                        // Card shadow
                        layer.enabled: true
                        layer.effect: DropShadow {
                            transparentBorder: true
                            radius: 8
                            samples: 17
                            color: "#20000000"
                        }

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 20
                            spacing: 16

                            // Title
                            Text {
                                text: "Live Camera Feed"
                                font.family: fontFamily
                                font.pixelSize: fontSizeXLarge
                                font.bold: true
                                color: textPrimary
                                Layout.alignment: Qt.AlignLeft
                            }

                            // Camera Display
                            Rectangle {
                                id: cameraContainer
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                radius: 12
                                color: primaryColor
                                border.color: borderColor
                                border.width: 1

                                Image {
                                    id: cameraImage
                                    anchors.fill: parent
                                    anchors.margins: 2
                                    fillMode: Image.PreserveAspectFit
                                    source: "image://camera/frame"
                                    cache: false

                                    Timer {
                                        interval: 33  // ~30 FPS
                                        running: true
                                        repeat: true
                                        onTriggered: {
                                            cameraImage.source = ""
                                            cameraImage.source = "image://camera/frame?" + Date.now()
                                        }
                                    }
                                }

                                // Camera Off Overlay
                                Rectangle {
                                    anchors.fill: parent
                                    color: "#800F172A"
                                    visible: !backend || !backend.is_camera_active
                                    radius: parent.radius

                                    Column {
                                        anchors.centerIn: parent
                                        spacing: 20

                                        Text {
                                            text: "📷"
                                            font.pixelSize: 64
                                            anchors.horizontalCenter: parent.horizontalCenter
                                            color: textSecondary
                                        }

                                        Text {
                                            text: "Camera is offline"
                                            font.family: fontFamily
                                            font.pixelSize: fontSizeLarge
                                            color: textSecondary
                                            anchors.horizontalCenter: parent.horizontalCenter
                                        }

                                        Text {
                                            text: "Click 'Start Camera' to begin"
                                            font.family: fontFamily
                                            font.pixelSize: fontSizeMedium
                                            color: textSecondary
                                            anchors.horizontalCenter: parent.horizontalCenter
                                        }
                                    }
                                }

                                // Face Count Indicator
                                Rectangle {
                                    x: 16
                                    y: 16
                                    width: faceCountLabel.width + 24
                                    height: 32
                                    radius: 16
                                    color: Qt.rgba(0.12, 0.12, 0.12, 0.9)
                                    visible: backend && backend.is_camera_active && faceCount > 0

                                    Row {
                                        anchors.centerIn: parent
                                        spacing: 8

                                        Text {
                                            text: "👤"
                                            font.pixelSize: 16
                                            color: "white"
                                        }

                                        Text {
                                            id: faceCountLabel
                                            text: faceCount + (faceCount === 1 ? " face" : " faces")
                                            font.family: fontFamily
                                            font.pixelSize: fontSizeSmall
                                            color: "white"
                                            font.bold: true
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Control Buttons Card
                    Rectangle {
                        id: controlsCard
                        Layout.fillWidth: true
                        height: 120
                        radius: 16
                        color: cardColor
                        border.color: borderColor
                        border.width: 1

                        layer.enabled: true
                        layer.effect: DropShadow {
                            transparentBorder: true
                            radius: 8
                            samples: 17
                            color: "#20000000"
                        }

                        RowLayout {
                            anchors.fill: parent
                            anchors.margins: 20
                            spacing: 16

                            // Camera Button
                            Rectangle {
                                id: cameraButton
                                Layout.fillWidth: true
                                height: 50
                                radius: 10
                                color: (backend && backend.is_camera_active) ? errorColor : accentColor

                                Row {
                                    anchors.centerIn: parent
                                    spacing: 10

                                    Text {
                                        text: (backend && backend.is_camera_active) ? "⏸️" : "▶️"
                                        font.pixelSize: 18
                                        color: "white"
                                        anchors.verticalCenter: parent.verticalCenter
                                    }

                                    Text {
                                        text: (backend && backend.is_camera_active) ? "Stop Camera" : "Start Camera"
                                        font.family: fontFamily
                                        font.pixelSize: fontSizeMedium
                                        color: "white"
                                        font.bold: true
                                        anchors.verticalCenter: parent.verticalCenter
                                    }
                                }

                                MouseArea {
                                    anchors.fill: parent
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: {
                                        if (backend) {
                                            backend.toggle_camera()
                                        }
                                    }
                                }
                            }

                            // Record Button
                            Rectangle {
                                id: recordButton
                                Layout.fillWidth: true
                                height: 50
                                radius: 10
                                color: (backend && backend.is_recording) ? errorColor : warningColor
                                enabled: backend && backend.is_camera_active
                                opacity: enabled ? 1 : 0.6

                                Row {
                                    anchors.centerIn: parent
                                    spacing: 10

                                    Text {
                                        text: (backend && backend.is_recording) ? "⏹️" : "🔴"
                                        font.pixelSize: 18
                                        color: "white"
                                        anchors.verticalCenter: parent.verticalCenter
                                    }

                                    Text {
                                        text: (backend && backend.is_recording) ? "Stop Recording" : "Start Recording"
                                        font.family: fontFamily
                                        font.pixelSize: fontSizeMedium
                                        color: "white"
                                        font.bold: true
                                        anchors.verticalCenter: parent.verticalCenter
                                    }
                                }

                                MouseArea {
                                    anchors.fill: parent
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: {
                                        if (backend) {
                                            backend.toggle_recording()
                                        }
                                    }
                                }
                            }

                            // Capture Button
                            Rectangle {
                                id: captureButton
                                Layout.fillWidth: true
                                height: 50
                                radius: 10
                                color: secondaryColor
                                enabled: backend && backend.is_camera_active
                                opacity: enabled ? 1 : 0.6

                                Row {
                                    anchors.centerIn: parent
                                    spacing: 10

                                    Text {
                                        text: "📸"
                                        font.pixelSize: 18
                                        color: "white"
                                        anchors.verticalCenter: parent.verticalCenter
                                    }

                                    Text {
                                        text: "Capture"
                                        font.family: fontFamily
                                        font.pixelSize: fontSizeMedium
                                        color: "white"
                                        font.bold: true
                                        anchors.verticalCenter: parent.verticalCenter
                                    }
                                }

                                MouseArea {
                                    anchors.fill: parent
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: {
                                        if (backend) {
                                            var filename = backend.capture_frame()
                                            if (filename !== "")
                                                notification.show("Capture saved: " + filename, "success")
                                        }
                                    }
                                }
                            }

                            // Camera Selection
                            Rectangle {
                                Layout.fillWidth: true
                                height: 50
                                radius: 10
                                color: secondaryColor

                                ComboBox {
                                    id: cameraComboBox
                                    anchors.fill: parent
                                    anchors.margins: 2
                                    model: cameraModel
                                    enabled: !backend || !backend.is_camera_active

                                    background: Rectangle {
                                        radius: 8
                                        color: "transparent"
                                    }

                                    contentItem: Text {
                                        text: cameraComboBox.displayText
                                        font.family: fontFamily
                                        font.pixelSize: fontSizeMedium
                                        color: textPrimary
                                        verticalAlignment: Text.AlignVCenter
                                        leftPadding: 12
                                    }

                                    onCurrentIndexChanged: {
                                        if (currentIndex >= 0 && backend)
                                            backend.set_camera(currentIndex)
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Right Panel - Analytics Dashboard
            Rectangle {
                id: rightPanel
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: "transparent"

                ScrollView {
                    id: scrollView
                    anchors.fill: parent
                    anchors.margins: 20
                    clip: true

                    ColumnLayout {
                        width: scrollView.width - 40  // Account for margins
                        spacing: 20

                        // Current Emotion Card
                        Rectangle {
                            Layout.fillWidth: true
                            height: 200
                            radius: 16
                            color: cardColor
                            border.color: borderColor
                            border.width: 1

                            layer.enabled: true
                            layer.effect: DropShadow {
                                transparentBorder: true
                                radius: 8
                                samples: 17
                                color: "#20000000"
                            }

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 20
                                spacing: 16

                                Text {
                                    text: "Current Emotion"
                                    font.family: fontFamily
                                    font.pixelSize: fontSizeXLarge
                                    font.bold: true
                                    color: textPrimary
                                    Layout.alignment: Qt.AlignLeft
                                }

                                RowLayout {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    spacing: 20

                                    // Emotion Icon
                                    Rectangle {
                                        width: 100
                                        height: 100
                                        radius: 50
                                        gradient: Gradient {
                                            GradientStop { position: 0.0; color: gradientStart }
                                            GradientStop { position: 1.0; color: gradientEnd }
                                        }

                                        Text {
                                            anchors.centerIn: parent
                                            text: getEmoji(currentEmotion)
                                            font.pixelSize: 40
                                            color: "white"
                                        }
                                    }

                                    ColumnLayout {
                                        Layout.fillWidth: true
                                        Layout.fillHeight: true
                                        spacing: 8

                                        Text {
                                            id: emotionLabel
                                            text: currentEmotion.charAt(0).toUpperCase() + currentEmotion.slice(1)
                                            font.family: fontFamily
                                            font.pixelSize: fontSizeXXLarge
                                            font.bold: true
                                            color: textPrimary
                                            Layout.alignment: Qt.AlignLeft
                                        }

                                        // Confidence Bar
                                        ColumnLayout {
                                            Layout.fillWidth: true
                                            spacing: 6

                                            RowLayout {
                                                Text {
                                                    text: "Confidence"
                                                    font.family: fontFamily
                                                    font.pixelSize: fontSizeMedium
                                                    color: textSecondary
                                                    Layout.fillWidth: true
                                                }

                                                Text {
                                                    id: confidenceLabel
                                                    text: (currentConfidence * 100).toFixed(1) + "%"
                                                    font.family: fontFamily
                                                    font.pixelSize: fontSizeMedium
                                                    color: accentColor
                                                    font.bold: true
                                                }
                                            }

                                            Rectangle {
                                                Layout.fillWidth: true
                                                height: 8
                                                radius: 4
                                                color: secondaryColor

                                                Rectangle {
                                                    width: parent.width * currentConfidence
                                                    height: parent.height
                                                    radius: 4
                                                    gradient: Gradient {
                                                        GradientStop { position: 0.0; color: gradientStart }
                                                        GradientStop { position: 1.0; color: gradientEnd }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Emotion Probabilities Card - MODIFIED for 2 emotions
                        Rectangle {
                            Layout.fillWidth: true
                            height: 200  // Reduced height for 2 emotions
                            radius: 16
                            color: cardColor
                            border.color: borderColor
                            border.width: 1

                            layer.enabled: true
                            layer.effect: DropShadow {
                                transparentBorder: true
                                radius: 8
                                samples: 17
                                color: "#20000000"
                            }

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 20
                                spacing: 16

                                Text {
                                    text: "Emotion Probabilities"
                                    font.family: fontFamily
                                    font.pixelSize: fontSizeXLarge
                                    font.bold: true
                                    color: textPrimary
                                    Layout.alignment: Qt.AlignLeft
                                }

                                GridLayout {
                                    columns: 1  // Single column for 2 emotions
                                    rowSpacing: 12
                                    columnSpacing: 16
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true

                                    Repeater {
                                        // MODIFIED: Only 2 emotions
                                        model: ["happy", "sad"]

                                        Rectangle {
                                            Layout.fillWidth: true
                                            Layout.preferredHeight: 40
                                            radius: 8
                                            color: secondaryColor

                                            Rectangle {
                                                width: parent.width * (probabilities[modelData] || 0)
                                                height: parent.height
                                                radius: 8
                                                color: getEmotionColor(modelData)
                                            }

                                            Row {
                                                anchors.fill: parent
                                                anchors.leftMargin: 12
                                                anchors.rightMargin: 12
                                                spacing: 10

                                                Text {
                                                    text: getEmoji(modelData)
                                                    font.pixelSize: 16
                                                    color: "white"
                                                    anchors.verticalCenter: parent.verticalCenter
                                                }

                                                Text {
                                                    text: modelData.charAt(0).toUpperCase() + modelData.slice(1)
                                                    font.family: fontFamily
                                                    font.pixelSize: fontSizeMedium
                                                    color: textPrimary
                                                    font.bold: true
                                                    anchors.verticalCenter: parent.verticalCenter
                                                    Layout.fillWidth: true
                                                }

                                                Item { width: 8 }

                                                Text {
                                                    text: ((probabilities[modelData] || 0) * 100).toFixed(1) + "%"
                                                    font.family: fontFamily
                                                    font.pixelSize: fontSizeMedium
                                                    color: textPrimary
                                                    anchors.verticalCenter: parent.verticalCenter
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Stats and History Row
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 20

                            // Session Statistics Card
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 300
                                radius: 16
                                color: cardColor
                                border.color: borderColor
                                border.width: 1

                                layer.enabled: true
                                layer.effect: DropShadow {
                                    transparentBorder: true
                                    radius: 8
                                    samples: 17
                                    color: "#20000000"
                                }

                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 20
                                    spacing: 16

                                    Text {
                                        text: "Session Statistics"
                                        font.family: fontFamily
                                        font.pixelSize: fontSizeXLarge
                                        font.bold: true
                                        color: textPrimary
                                        Layout.alignment: Qt.AlignLeft
                                    }

                                    ListView {
                                        id: statsList
                                        Layout.fillWidth: true
                                        Layout.fillHeight: true
                                        model: statsModel
                                        clip: true
                                        spacing: 8

                                        delegate: Item {
                                            width: parent.width
                                            height: 36

                                            RowLayout {
                                                anchors.fill: parent
                                                anchors.margins: 4
                                                spacing: 12

                                                Rectangle {
                                                    width: 32
                                                    height: 32
                                                    radius: 16
                                                    color: getEmotionColor(name.split(": ")[0])
                                                    Layout.alignment: Qt.AlignVCenter

                                                    Text {
                                                        anchors.centerIn: parent
                                                        text: getEmoji(name.split(": ")[0])
                                                        font.pixelSize: 14
                                                        color: "white"
                                                    }
                                                }

                                                Text {
                                                    text: name.split(": ")[0]
                                                    font.family: fontFamily
                                                    font.pixelSize: fontSizeMedium
                                                    color: textPrimary
                                                    Layout.preferredWidth: 80
                                                    Layout.alignment: Qt.AlignVCenter
                                                }

                                                Item { Layout.fillWidth: true }

                                                Rectangle {
                                                    Layout.fillWidth: true
                                                    Layout.preferredHeight: 8
                                                    radius: 4
                                                    color: primaryColor
                                                    Layout.alignment: Qt.AlignVCenter

                                                    Rectangle {
                                                        width: parent.width * (percentage / 100)
                                                        height: parent.height
                                                        radius: 4
                                                        color: getEmotionColor(name.split(": ")[0])
                                                    }
                                                }

                                                Text {
                                                    text: percentage.toFixed(1) + "%"
                                                    font.family: fontFamily
                                                    font.pixelSize: fontSizeMedium
                                                    color: textPrimary
                                                    font.bold: true
                                                    Layout.alignment: Qt.AlignVCenter
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Detection History Card
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 300
                                radius: 16
                                color: cardColor
                                border.color: borderColor
                                border.width: 1

                                layer.enabled: true
                                layer.effect: DropShadow {
                                    transparentBorder: true
                                    radius: 8
                                    samples: 17
                                    color: "#20000000"
                                }

                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 20
                                    spacing: 16

                                    Text {
                                        text: "Detection History"
                                        font.family: fontFamily
                                        font.pixelSize: fontSizeXLarge
                                        font.bold: true
                                        color: textPrimary
                                        Layout.alignment: Qt.AlignLeft
                                    }

                                    ListView {
                                        id: detectionList
                                        Layout.fillWidth: true
                                        Layout.fillHeight: true
                                        model: detectionModel
                                        clip: true
                                        spacing: 8

                                        delegate: Rectangle {
                                            width: parent.width
                                            height: 60
                                            radius: 12
                                            color: secondaryColor

                                            RowLayout {
                                                anchors.fill: parent
                                                anchors.margins: 12
                                                spacing: 16

                                                Rectangle {
                                                    width: 40
                                                    height: 40
                                                    radius: 20
                                                    color: getEmotionColor(emotion)
                                                    Layout.alignment: Qt.AlignVCenter

                                                    Text {
                                                        anchors.centerIn: parent
                                                        text: getEmoji(emotion)
                                                        font.pixelSize: 18
                                                        color: "white"
                                                    }
                                                }

                                                ColumnLayout {
                                                    spacing: 2
                                                    Layout.fillWidth: true
                                                    Layout.alignment: Qt.AlignVCenter

                                                    Text {
                                                        text: emotion.charAt(0).toUpperCase() + emotion.slice(1)
                                                        font.family: fontFamily
                                                        font.pixelSize: fontSizeMedium
                                                        font.bold: true
                                                        color: textPrimary
                                                    }

                                                    Text {
                                                        text: timestamp.substring(11, 19)
                                                        font.family: fontFamily
                                                        font.pixelSize: fontSizeSmall
                                                        color: textSecondary
                                                    }
                                                }

                                                Rectangle {
                                                    width: 80
                                                    height: 28
                                                    radius: 14
                                                    color: accentColor
                                                    Layout.alignment: Qt.AlignVCenter

                                                    Text {
                                                        anchors.centerIn: parent
                                                        text: (confidence * 100).toFixed(0) + "%"
                                                        font.family: fontFamily
                                                        font.pixelSize: fontSizeMedium
                                                        color: "white"
                                                        font.bold: true
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Settings and Controls Card
                        Rectangle {
                            Layout.fillWidth: true
                            height: 240
                            radius: 16
                            color: cardColor
                            border.color: borderColor
                            border.width: 1

                            layer.enabled: true
                            layer.effect: DropShadow {
                                transparentBorder: true
                                radius: 8
                                samples: 17
                                color: "#20000000"
                            }

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 20
                                spacing: 16

                                // First Row: Method Selection and Model Status
                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 20

                                    // Method Selection
                                    ColumnLayout {
                                        Layout.fillWidth: true
                                        spacing: 8

                                        Text {
                                            text: "Detection Method"
                                            font.family: fontFamily
                                            font.pixelSize: fontSizeMedium
                                            color: textSecondary
                                        }

                                        Rectangle {
                                            Layout.fillWidth: true
                                            height: 44
                                            radius: 8
                                            color: secondaryColor
                                            border.color: borderColor
                                            border.width: 1

                                            ComboBox {
                                                id: methodComboBox
                                                anchors.fill: parent
                                                anchors.margins: 2
                                                model: backend && backend.available_methods ? backend.available_methods : []
                                                enabled: backend && backend.available_methods && backend.available_methods.length > 0
                                                currentIndex: 0

                                                background: Rectangle {
                                                    radius: 6
                                                    color: "transparent"
                                                }

                                                contentItem: Text {
                                                    text: methodComboBox.currentText
                                                    font.family: fontFamily
                                                    font.pixelSize: fontSizeMedium
                                                    color: textPrimary
                                                    verticalAlignment: Text.AlignVCenter
                                                    leftPadding: 12
                                                }

                                                popup: Popup {
                                                    width: methodComboBox.width
                                                    implicitHeight: contentItem.implicitHeight
                                                    padding: 1

                                                    contentItem: ListView {
                                                        clip: true
                                                        implicitHeight: contentHeight
                                                        model: methodComboBox.popup.visible ? methodComboBox.delegateModel : null
                                                        currentIndex: methodComboBox.highlightedIndex

                                                        ScrollIndicator.vertical: ScrollIndicator { }
                                                    }

                                                    background: Rectangle {
                                                        radius: 6
                                                        color: cardColor
                                                        border.color: borderColor
                                                    }
                                                }

                                                delegate: ItemDelegate {
                                                    width: methodComboBox.width
                                                    height: 36
                                                    text: modelData
                                                    font.family: fontFamily
                                                    font.pixelSize: fontSizeMedium
                                                    highlighted: methodComboBox.highlightedIndex === index

                                                    background: Rectangle {
                                                        color: highlighted ? accentColor : "transparent"
                                                        radius: 4
                                                    }
                                                }

                                                onCurrentTextChanged: {
                                                    if (backend && currentText !== "") {
                                                        backend.set_detection_method(currentText)
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Model Status
                                    ColumnLayout {
                                        Layout.fillWidth: true
                                        spacing: 8

                                        Text {
                                            text: "Model Status"
                                            font.family: fontFamily
                                            font.pixelSize: fontSizeMedium
                                            color: textSecondary
                                        }

                                        RowLayout {
                                            spacing: 10
                                            Layout.alignment: Qt.AlignLeft

                                            Rectangle {
                                                width: 12
                                                height: 12
                                                radius: 6
                                                color: (backend && backend.model_loaded) ? successColor : errorColor
                                            }

                                            Text {
                                                text: (backend && backend.model_loaded) ? "✅ Model Loaded" : "❌ Model Not Loaded"
                                                font.family: fontFamily
                                                font.pixelSize: fontSizeMedium
                                                color: (backend && backend.model_loaded) ? successColor : errorColor
                                            }
                                        }
                                    }

                                    // Confidence Threshold
                                    ColumnLayout {
                                        Layout.fillWidth: true
                                        spacing: 8

                                        RowLayout {
                                            Text {
                                                text: "Confidence Threshold"
                                                font.family: fontFamily
                                                font.pixelSize: fontSizeMedium
                                                color: textSecondary
                                                Layout.fillWidth: true
                                            }

                                            Text {
                                                id: thresholdLabel
                                                text: (backend ? (backend.confidence_threshold * 100).toFixed(0) : 50) + "%"
                                                font.family: fontFamily
                                                font.pixelSize: fontSizeMedium
                                                color: accentColor
                                                font.bold: true
                                            }
                                        }

                                        Slider {
                                            id: confidenceSlider
                                            Layout.fillWidth: true
                                            from: 0.1
                                            to: 0.9
                                            value: 0.5
                                            stepSize: 0.05

                                            background: Rectangle {
                                                implicitHeight: 4
                                                radius: 2
                                                color: secondaryColor

                                                Rectangle {
                                                    width: confidenceSlider.visualPosition * parent.width
                                                    height: parent.height
                                                    radius: 2
                                                    color: accentColor
                                                }
                                            }

                                            handle: Rectangle {
                                                x: confidenceSlider.visualPosition * (parent.width - width)
                                                y: parent.height / 2 - height / 2
                                                width: 20
                                                height: 20
                                                radius: 10
                                                color: "white"
                                                border.color: accentColor
                                                border.width: 2
                                            }

                                            onValueChanged: {
                                                if (backend) {
                                                    backend.confidence_threshold = value
                                                    thresholdLabel.text = (value * 100).toFixed(0) + "%"
                                                }
                                            }
                                        }
                                    }
                                }

                                // Second Row: Control Buttons
                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 16

                                    Rectangle {
                                        Layout.fillWidth: true
                                        height: 44
                                        radius: 10
                                        color: errorColor

                                        Row {
                                            anchors.centerIn: parent
                                            spacing: 8

                                            Text {
                                                text: "🗑️"
                                                font.pixelSize: 18
                                                color: "white"
                                                anchors.verticalCenter: parent.verticalCenter
                                            }

                                            Text {
                                                text: "Clear Session"
                                                font.family: fontFamily
                                                font.pixelSize: fontSizeMedium
                                                color: "white"
                                                font.bold: true
                                                anchors.verticalCenter: parent.verticalCenter
                                            }
                                        }

                                        MouseArea {
                                            anchors.fill: parent
                                            cursorShape: Qt.PointingHandCursor
                                            onClicked: {
                                                if (backend) backend.clear_session()
                                                detectionModel.clear()
                                            }
                                        }
                                    }

                                    Rectangle {
                                        Layout.fillWidth: true
                                        height: 44
                                        radius: 10
                                        gradient: Gradient {
                                            GradientStop { position: 0.0; color: warningColor }
                                            GradientStop { position: 1.0; color: Qt.darker(warningColor, 1.2) }
                                        }

                                        Row {
                                            anchors.centerIn: parent
                                            spacing: 8

                                            Text {
                                                text: "🔄"
                                                font.pixelSize: 18
                                                color: "white"
                                                anchors.verticalCenter: parent.verticalCenter
                                            }

                                            Text {
                                                text: "Refresh"
                                                font.family: fontFamily
                                                font.pixelSize: fontSizeMedium
                                                color: "white"
                                                font.bold: true
                                                anchors.verticalCenter: parent.verticalCenter
                                            }
                                        }

                                        MouseArea {
                                            anchors.fill: parent
                                            cursorShape: Qt.PointingHandCursor
                                            onClicked: if (backend) backend.initialize()
                                        }
                                    }

                                    Rectangle {
                                        Layout.fillWidth: true
                                        height: 44
                                        radius: 10
                                        color: successColor

                                        Row {
                                            anchors.centerIn: parent
                                            spacing: 8

                                            Text {
                                                text: "💾"
                                                font.pixelSize: 18
                                                color: "white"
                                                anchors.verticalCenter: parent.verticalCenter
                                            }

                                            Text {
                                                text: "Export CSV"
                                                font.family: fontFamily
                                                font.pixelSize: fontSizeMedium
                                                color: "white"
                                                font.bold: true
                                                anchors.verticalCenter: parent.verticalCenter
                                            }
                                        }

                                        MouseArea {
                                            anchors.fill: parent
                                            cursorShape: Qt.PointingHandCursor
                                            onClicked: exportDialog.open()
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Export Dialog
    Dialog {
        id: exportDialog
        title: "Export Session Data"
        standardButtons: Dialog.Save | Dialog.Cancel
        modal: true
        width: 400
        height: 200
        anchors.centerIn: parent

        background: Rectangle {
            radius: 12
            color: cardColor
            border.color: borderColor
            border.width: 1
        }

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 20
            spacing: 16

            Text {
                text: "Export to CSV"
                font.family: fontFamily
                font.pixelSize: fontSizeLarge
                font.bold: true
                color: textPrimary
                Layout.alignment: Qt.AlignLeft
            }

            TextField {
                id: filenameInput
                Layout.fillWidth: true
                height: 40
                text: "emotion_data.csv"
                placeholderText: "Enter filename"
                font.family: fontFamily
                font.pixelSize: fontSizeMedium
                color: textPrimary

                background: Rectangle {
                    radius: 8
                    color: secondaryColor
                    border.color: borderColor
                    border.width: 1
                }
            }

            Text {
                text: "File will be saved in the current directory"
                font.family: fontFamily
                font.pixelSize: fontSizeSmall
                color: textSecondary
                Layout.alignment: Qt.AlignLeft
            }
        }

        onAccepted: {
            var filename = filenameInput.text
            if (!filename.toLowerCase().endsWith('.csv')) {
                filename += '.csv'
            }

            var success = backend && backend.export_session_data(filename)
            if (success)
                notification.show("Data exported successfully!", "success")
            else
                notification.show("Export failed!", "error")
        }
    }

    // Notification System
    Rectangle {
        id: notification
        width: 320
        height: 56
        radius: 12
        anchors.top: parent.top
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.topMargin: 20
        visible: false
        z: 1000
        color: accentColor

        Behavior on opacity {
            NumberAnimation { duration: 300 }
        }

        function show(message, type) {
            notificationText.text = message
            notification.color = type === "success" ? successColor :
                                type === "error" ? errorColor : accentColor
            notification.opacity = 1
            notification.visible = true
            notificationTimer.start()
        }

        Timer {
            id: notificationTimer
            interval: 3000
            onTriggered: {
                notification.opacity = 0
                notification.visible = false
            }
        }

        Row {
            anchors.centerIn: parent
            spacing: 12

            Rectangle {
                width: 24
                height: 24
                radius: 12
                color: "white"
                anchors.verticalCenter: parent.verticalCenter

                Text {
                    anchors.centerIn: parent
                    text: "✓"
                    font.pixelSize: 16
                    font.bold: true
                    color: notification.color
                }
            }

            Text {
                id: notificationText
                anchors.verticalCenter: parent.verticalCenter
                color: "white"
                font.family: fontFamily
                font.pixelSize: fontSizeMedium
                font.bold: true
            }
        }
    }

    // Connections to backend
    Connections {
        target: backend
        enabled: backend !== null

        function onEmotion_detected(emotion, confidence, probabilities) {
            currentEmotion = emotion
            currentConfidence = confidence
            emotionLabel.text = emotion.charAt(0).toUpperCase() + emotion.slice(1)
            confidenceLabel.text = (confidence * 100).toFixed(1) + "%"

            // Update probabilities
            var newProbs = {}
            for (var key in probabilities) {
                newProbs[key] = probabilities[key]
            }
            mainWindow.probabilities = newProbs

            // Add to detection history
            if (backend && backend.is_recording) {
                detectionModel.append({
                    emotion: emotion,
                    confidence: confidence,
                    timestamp: new Date().toISOString()
                })
            }
        }

        function onFps_updated(fps) {
            fpsLabel.text = fps
        }

        function onFace_count_updated(count) {
            faceCount = count
        }

        function onAvailable_cameras_changed(cameras) {
            cameraModel.clear()
            for (var i = 0; i < cameras.length; i++) {
                cameraModel.append({text: cameras[i]})
            }
            if (cameras.length > 0) {
                cameraComboBox.currentIndex = 0
            }
        }

        function onSession_stats_updated(stats) {
            statsModel.clear()
            var total = stats.total_detections
            var distribution = stats.emotion_distribution

            for (var emotion in distribution) {
                var count = distribution[emotion]
                var percentage = total > 0 ? (count / total) * 100 : 0
                statsModel.append({
                    name: emotion + ": " + count,
                    percentage: percentage,
                    count: count
                })
            }
        }

        function onError_occurred(message) {
            notification.show("Error: " + message, "error")
        }

        function onDetection_method_changed(method) {
            // Update method combo box if needed
            for (var i = 0; i < methodComboBox.model.length; i++) {
                if (methodComboBox.model[i] === method) {
                    methodComboBox.currentIndex = i
                    break
                }
            }
        }

        function onModel_status_changed(message, success) {
            if (success) {
                notification.show(message, "success")
            } else {
                notification.show(message, "warning")
            }
        }
    }

    Component.onCompleted: {
        // Initialize the backend if it exists
        if (backend) {
            backend.initialize()
        }
    }
}
