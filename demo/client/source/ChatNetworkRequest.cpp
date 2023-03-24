//
// Created by diwa on 2023/3/24.
//
// ChatNetworkRequest.cpp
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QUrl>
#include <QEventLoop>
#include "ChatNetworkRequest.h"

ChatNetworkRequest::ChatNetworkRequest(QObject *parent) :
        QObject(parent)
{

}

void ChatNetworkRequest::sendGetRequest(const QUrl &url, const QUrlQuery &params)
{
    QUrl requestUrl(url);
    requestUrl.setQuery(params);
    QNetworkRequest request(requestUrl);
    request.setUrl(requestUrl);

    QNetworkReply *reply = m_networkAccessManager.get(request);
    QObject::connect(reply, &QNetworkReply::finished, this,  [=]() {
        if (reply->error() == QNetworkReply::NoError) {
            emit responseReceived(reply->readAll(), true);
        } else {
            emit responseReceived(QByteArray(), false);
        }
        reply->deleteLater();
    });
}
