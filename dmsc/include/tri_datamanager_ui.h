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
#ifndef TRI_DATAMANAGER_UI
#define TRI_DATAMANAGER_UI

#include <QFrame>
#include <tri_datamanager.h>
#include <ui_tri_datamanager_frame.h>

#include <QAbstractItemModel>
#include <QModelIndex>
#include <QVariant>


class TriTreeModel : public QAbstractItemModel
{
    Q_OBJECT

  public:
    TriTreeModel ( tree<TriDataManager::_data_piece *> *, QObject *parent = 0 );
    ~TriTreeModel();

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

      TriDataManager::_data_piece *node;

      _treeitem * parent;

      _treeitem ( TriDataManager::_data_piece * _node , _treeitem * par ) :node ( _node ),parent ( par )
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
    void setupModelData ( tree<TriDataManager::_data_piece *> * );

    _treeitem *m_tree;
};

class TriRecursiveTreeItemSelectionModel:public QItemSelectionModel
{
    Q_OBJECT

  public:
    TriRecursiveTreeItemSelectionModel ( QAbstractItemModel * model,QTreeView * pTreeView ) :QItemSelectionModel ( model ),m_pTreeView ( pTreeView ) {};

  public slots:

    virtual void select ( const QModelIndex &index, QItemSelectionModel::SelectionFlags command )
    {

      QItemSelectionModel::select ( index,command );

      if ( ! ( command & QItemSelectionModel::Select ) )
      {
        return;
      }

      if ( index.isValid() && !m_pTreeView->isExpanded ( index ) )
      {
        QModelIndexList indexes_to_select;

        collect_all_children ( index,indexes_to_select );

        for ( QModelIndexList::iterator ind_it = indexes_to_select.begin(); ind_it != indexes_to_select.end();++ind_it )
        {
          QItemSelectionModel::select ( *ind_it,QItemSelectionModel::Select );
        }
      }
    }

    virtual void select ( const QItemSelection &selection, QItemSelectionModel::SelectionFlags command )
    {
      QItemSelectionModel::select ( selection,command );
    }
  private:

    void collect_all_children ( const QModelIndex &index,QModelIndexList &retlist )
    {
      QModelIndexList ind_stack;

      ind_stack.push_back ( index );

      while ( ind_stack.size() !=0 )
      {
        QModelIndex top_ind = ind_stack.front();

        ind_stack.pop_front();

        retlist.push_back ( top_ind );

        uint child_ind = 0;

        QModelIndex child = top_ind.child ( child_ind++, top_ind.column() );

        while ( child.isValid() )
        {
          ind_stack.push_back ( child );

          child = top_ind.child ( child_ind++, top_ind.column() );
        }
      }

    }

    QTreeView *m_pTreeView;
};


class TriDataManager_ui :public QFrame,public TriDataManager,public Ui::TriDataManager_QtFrame
{
    Q_OBJECT

  public:
    TriDataManager_ui ( std::string cmdline );

    virtual ~TriDataManager_ui() {};

    virtual QFrame * getQFrame() { return this;}

    enum eTreeViewActions
    {
      TVA_SURF,
      TVA_CPS,
      TVA_GRAPH,
      TVA_QUADS,
      TVA_GRAD,
      TVA_CANCELABLE_PAIRS,
      TVA_EXTERIOR
    };

    void perform_tva_action ( const eTreeViewActions &,const bool & );

    bool get_tva_state ( const eTreeViewActions & );

  private slots:

    void on_show_critical_point_labels_checkBox_stateChanged ( int state );
    void on_show_critical_points_checkBox_stateChanged ( int state );
    void on_show_msgraph_checkBox_stateChanged ( int state );
    void on_datapiece_treeView_customContextMenuRequested ( const QPoint &p );


    void show_surf_toggled ( bool state ) {perform_tva_action ( TVA_SURF,state );}
    void show_cps_toggled ( bool state ) {perform_tva_action ( TVA_CPS,state );}
    void show_graph_toggled ( bool state ) {perform_tva_action ( TVA_GRAPH,state );}
    void show_quads_toggled ( bool state ) {perform_tva_action ( TVA_QUADS,state );}
    void show_grad_toggled ( bool state ) {perform_tva_action ( TVA_GRAD,state );}
    void show_exterior_toggled ( bool state ) {perform_tva_action ( TVA_EXTERIOR,state );}
    void show_cancelable_pairs_toggled ( bool state ) {perform_tva_action ( TVA_CANCELABLE_PAIRS,state );}

};

#endif
