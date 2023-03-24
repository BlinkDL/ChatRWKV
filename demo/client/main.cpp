//
// Created by diwa on 2023/3/21.
//
#include <QApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QIcon>
#include "source/InteractiveMain.h"

int main(int argc, char** argv)
{
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QGuiApplication app(argc, argv);

#ifdef Q_OS_WIN
    app.setWindowIcon(QIcon(":/rwkv_profile.ico"));
#endif

    InteractiveMain interactive;
    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty("interactive", &interactive);

    const QUrl url(QStringLiteral("qrc:/Main.qml"));
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
                     &app, [url](QObject *obj, const QUrl &objUrl) {
        if (!obj && url == objUrl)
            QCoreApplication::exit(-1);
    }, Qt::QueuedConnection);
    engine.load(url);

    return app.exec();
}
