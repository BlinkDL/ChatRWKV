//
// Created by diwa on 2023/3/21.
//

#pragma once
#include <QAbstractListModel>
#include "ChatListDefine.h"

class ChatListModel : public QAbstractListModel
{
    Q_OBJECT
public:
    explicit ChatListModel(QObject *parent = nullptr);

    //data
    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    Q_INVOKABLE void clearModel();

    Q_INVOKABLE void appendText(const QString &user,
                                const QString &sender,
                                const QString &text);

private:
    bool isVaidRow(int row) const;

private:
    QList<QSharedPointer<ChatDataBasic>> chatList;
};

