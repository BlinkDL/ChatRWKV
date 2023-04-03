//
// Created by diwa on 2023/3/21.
//

#include <QTimer>
#include <QDateTime>
#include <QRandomGenerator>
#include "ChatListModel.h"

ChatListModel::ChatListModel(QObject *parent)
    : QAbstractListModel(parent)
{
}

int ChatListModel::rowCount(const QModelIndex &parent) const
{
    if (parent.isValid())
        return 0;

    return chatList.count();
}

QVariant ChatListModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();

    const int row = index.row();
    auto item = chatList.at(row);
    switch(role)
    {
    default:break;
    case Qt::UserRole+0:
        return item->id;
    case Qt::UserRole+1:
        return item->user;
    case Qt::UserRole+2:
        return item->sender;
    case Qt::UserRole+3:
        return QDateTime::fromMSecsSinceEpoch(item->datetime).toString("hh:mm:ss yyyy-MM-dd");
    case Qt::UserRole+4:
        return item->type;
    case Qt::UserRole+5:
        return item->status;
    case Qt::UserRole + 100:
    {
        ChatDataText *talk_data = static_cast<ChatDataText*>(item.get());
        return talk_data->text;
    }
    }

    return QVariant();
}

QHash<int, QByteArray> ChatListModel::roleNames() const
{
    return QHash<int,QByteArray>{
        { Qt::UserRole+0, "id" }
        ,{ Qt::UserRole+1, "user" }
        ,{ Qt::UserRole+2, "sender" }
        ,{ Qt::UserRole+3, "datetime" }
        ,{ Qt::UserRole+4, "type" }
        ,{ Qt::UserRole+5, "status" }
        ,{ Qt::UserRole+100, "text_text" }

    };
}

void ChatListModel::clearModel()
{
    beginResetModel();
    chatList.clear();
    endResetModel();
}

void ChatListModel::appendText(const QString &user,
                               const QString &sender,
                               const QString &text)
{
    ChatDataText *chatData=new ChatDataText;
    chatData->id = 0;
    chatData->user = user;
    chatData->sender = sender;
    chatData->datetime = QDateTime::currentDateTime().toMSecsSinceEpoch();
    chatData->type = ChatData::Text;
    chatData->text = text;

    beginInsertRows(QModelIndex(),chatList.count(),chatList.count());
    chatList.push_back(QSharedPointer<ChatDataBasic>(chatData));
    endInsertRows();
}


bool ChatListModel::isVaidRow(int row) const
{
    return (row >=0 &&row < chatList.count());
}
