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

#include <stack>
#include <QMenu>
#include <QSignalMapper>
#include <string>

#include <grid_datamanager.h>
#include <grid_datamanager_ui.h>

// only for the def of recursive treemodel
#include <quad_datamanager_ui.h>

#include <ui_grid_datamanager_frame.h>

using namespace std;

bool GridDataManager::get_tva_state ( const eTreeViewActions &action )
{

  uint num_checked_items = 0;
  uint num_unchecked_items = 0;

  QModelIndexList indexes = m_ui->datapiece_treeView->selectionModel()->selectedIndexes();

  for ( QModelIndexList::iterator ind_it = indexes.begin();ind_it != indexes.end(); ++ind_it )
  {
    GridTreeModel::_treeitem *item = static_cast<GridTreeModel::_treeitem*> ( ( *ind_it ).internalPointer() );

    switch ( action )
    {
    case TVA_SURF             :
      if ( item->node->m_bShowSurface ) ++num_checked_items;
      else ++num_unchecked_items;
      break;

    case TVA_CPS              :
      if ( item->node->m_bShowCps ) ++num_checked_items;
      else ++num_unchecked_items;
      break;

    case TVA_GRAPH            :
      if ( item->node->m_bShowMsGraph ) ++num_checked_items;
      else ++num_unchecked_items;
      break;

    case TVA_GRAD             :
      if ( item->node->m_bShowGrad ) ++num_checked_items;
      else ++num_unchecked_items;
      break;

    }
  }

  if ( num_checked_items > num_unchecked_items )
    return true;
  else
    return false;
}



void GridDataManager::perform_tva_action ( const eTreeViewActions &action, const bool & state )
{

  QModelIndexList indexes = m_ui->datapiece_treeView->selectionModel()->selectedIndexes();

  for ( QModelIndexList::iterator ind_it = indexes.begin();ind_it != indexes.end(); ++ind_it )
  {
    GridTreeModel::_treeitem *item = static_cast<GridTreeModel::_treeitem*> ( ( *ind_it ).internalPointer() );

    switch ( action )
    {
    case TVA_SURF             :
      item->node->m_bShowSurface            = state;
      break;
    case TVA_CPS              :
      item->node->m_bShowCps                = state;
      break;
    case TVA_GRAPH            :
      item->node->m_bShowMsGraph            = state;
      break;
    case TVA_GRAD             :
      item->node->m_bShowGrad               = state;
      break;
    default:
      break;
    }
  }

}

#define ADD_MENU_ACTION(menu_ptr,action_string,start_state,recv_func_name) \
{\
 QAction * _ama_action  = (menu_ptr)->addAction ( tr(action_string) );\
                          _ama_action->setCheckable ( true );\
                          _ama_action->setChecked ( (start_state));\
                          connect ( _ama_action,SIGNAL ( toggled ( bool ) ),this,SLOT ( recv_func_name(bool) ) );\
                        }\



void GridDataManager::on_datapiece_treeView_customContextMenuRequested ( const QPoint &pos )
{

  QMenu menu;

  ADD_MENU_ACTION ( &menu, "show surface", get_tva_state ( TVA_SURF ), show_surf_toggled );
  ADD_MENU_ACTION ( &menu, "show critical points", get_tva_state ( TVA_CPS ), show_cps_toggled );
  ADD_MENU_ACTION ( &menu, "show graph", get_tva_state ( TVA_GRAPH ), show_graph_toggled );
  ADD_MENU_ACTION ( &menu, "show grad", get_tva_state ( TVA_GRAD ), show_grad_toggled );


  menu.exec ( m_ui->datapiece_treeView->mapToGlobal ( pos ) );

}



void GridDataManager::on_show_critical_point_labels_checkBox_stateChanged ( int state )
{
  switch ( state )
  {

  case Qt::Checked:
    m_bShowCriticalPointLabels = true;
    break;

  case Qt::Unchecked:
    m_bShowCriticalPointLabels  = false;
    break;

  default:
    break;
  }
}



void GridDataManager::create_ui()
{
  m_ui = new Ui::GridDataManager_QtFrame();

  m_ui->setupUi ( this );

  GridTreeModel *model = new GridTreeModel ( &m_pieces);

  RecursiveTreeItemSelectionModel * sel_model =
      new RecursiveTreeItemSelectionModel ( model, m_ui->datapiece_treeView );

  m_ui->datapiece_treeView->setModel ( model );

  m_ui->datapiece_treeView->setSelectionModel ( sel_model );

}


void GridDataManager::destroy_ui()
{
  delete m_ui;
}

GridTreeModel::GridTreeModel ( std::vector<GridDataPiece *> * dpList, QObject *parent )
  : QAbstractItemModel ( parent )
{
  setupModelData ( dpList );
}

GridTreeModel::~GridTreeModel()
{
  delete m_tree;
  m_tree = NULL;
}


int GridTreeModel::columnCount ( const QModelIndex &/*parent*/ ) const
{
  return 1;
}

QVariant GridTreeModel::data ( const QModelIndex &index, int role ) const
{
  if ( !index.isValid() )
    return QVariant();

  if ( role != Qt::DisplayRole )
    return QVariant();

  _treeitem *item = static_cast<_treeitem*> ( index.internalPointer() );

  return QString(item->node->label().c_str());
}

Qt::ItemFlags GridTreeModel::flags ( const QModelIndex &index ) const
{
  if ( !index.isValid() )
    return 0;

  return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

QVariant GridTreeModel::headerData ( int /*section*/, Qt::Orientation orientation,
                                 int role ) const
{
  if ( orientation == Qt::Horizontal && role == Qt::DisplayRole )
    return "Data Pieces";

  return QVariant();
}

QModelIndex GridTreeModel::index ( int row, int column, const QModelIndex &parent ) const
{
  if ( !hasIndex ( row, column, parent ) )
    return QModelIndex();

  _treeitem *parentItem;

  if ( !parent.isValid() )
    parentItem = m_tree;
  else
    parentItem = static_cast<_treeitem*> ( parent.internalPointer() );

  if ( row < ( int ) parentItem->children.size() )
    return createIndex ( row, column, parentItem->children[row] );
  else
    return QModelIndex();

}

QModelIndex GridTreeModel::parent ( const QModelIndex &index ) const
{
  if ( !index.isValid() )
    return QModelIndex();

  _treeitem *childItem  = static_cast<_treeitem*> ( index.internalPointer() );

  _treeitem *parentItem = childItem->parent;

  if ( parentItem == m_tree )
    return QModelIndex();

  return createIndex ( parentItem->row(), 0, parentItem );

}

int GridTreeModel::rowCount ( const QModelIndex &parent ) const
{
  _treeitem *parentItem;

  if ( parent.column() > 0 )
    return 0;

  if ( !parent.isValid() )
    parentItem = m_tree;
  else
    parentItem = static_cast<_treeitem*> ( parent.internalPointer() );

  return parentItem->children.size();

}

void GridTreeModel::setupModelData ( std::vector<GridDataPiece *> *dpList)
{
  m_tree = new _treeitem();


  for ( std::vector<GridDataPiece*>::iterator dp_it =  dpList->begin();
          dp_it != dpList->end(); ++dp_it )
  {
    GridDataPiece *dp = *dp_it;

    _treeitem *parentItem = m_tree;


    _treeitem * dpItem = new _treeitem ( dp, parentItem );

    parentItem->children.push_back ( dpItem );

  }
}
