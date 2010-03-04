/***************************************************************************
*   Copyright (C) 2009 by Nithin Shivashankar,   *
*   nithin@gauss   *
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
*   This program is distributed in the hope that it will be useful,       *
*   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
*   GNU General Public License for more details.                          *
*                                                                         *
*   You should have received a copy of the GNU General Public License     *
*   along with this program; if not, write to the                         *
*   Free Software Foundation, Inc.,                                       *
*   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
***************************************************************************/
#ifndef GRID_DATAMANAGER_UI
#define GRID_DATAMANAGER_UI

#include <QFrame>
#include <QAbstractItemModel>
#include <QModelIndex>
#include <QVariant>
#include <QItemSelectionModel>
#include <QTreeView>

#include <tree.h>

struct GridDataPiece;

class GridTreeModel : public QAbstractItemModel
{
    Q_OBJECT

  public:
  GridTreeModel ( std::vector<GridDataPiece *> *, QObject *parent = 0 );
    ~GridTreeModel();

    QVariant data ( const QModelIndex &index, int role ) const;
    Qt::ItemFlags flags ( const QModelIndex &index ) const;
    QVariant headerData ( int section, Qt::Orientation orientation,
                          int role = Qt::DisplayRole ) const;
    QModelIndex index ( int row, int column,
                        const QModelIndex &parent = QModelIndex() ) const;
    QModelIndex parent ( const QModelIndex &index ) const;
    int rowCount ( const QModelIndex &parent = QModelIndex() ) const;
    int columnCount ( const QModelIndex &parent = QModelIndex() ) const;

    struct _treeitem
    {

      std::vector<_treeitem *> children;

      GridDataPiece * node;
      _treeitem * parent;

      _treeitem ( GridDataPiece * _node , _treeitem * par ) :
          node ( _node ),parent ( par )
      {
      }

      _treeitem()
      {
        node = NULL;
        parent = NULL;
      }

      int row()
      {
        return std::find ( parent->children.begin(),parent->children.end(),this ) - parent->children.begin();
      }
    };


  private:
    void setupModelData ( std::vector<GridDataPiece *> *);

    _treeitem *m_tree;
};

#endif
