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

#include <tri_datamanager_ui.h>
#include <stack>
#include <QMenu>
#include <QSignalMapper>

TriDataManager_ui::TriDataManager_ui ( std::string cmdline )
    : TriDataManager ( cmdline )
{
  this->setupUi ( this );

  TriTreeModel *model = new TriTreeModel ( &m_dpTree );

  TriRecursiveTreeItemSelectionModel * sel_model = new TriRecursiveTreeItemSelectionModel ( model, datapiece_treeView );

  datapiece_treeView->setModel ( model );

  datapiece_treeView->setSelectionModel ( sel_model );
}

bool TriDataManager_ui::get_tva_state ( const eTreeViewActions &action )
{

  uint num_checked_items = 0;
  uint num_unchecked_items = 0;

  QModelIndexList indexes = datapiece_treeView->selectionModel()->selectedIndexes();

  for ( QModelIndexList::iterator ind_it = indexes.begin();ind_it != indexes.end(); ++ind_it )
  {
    TriTreeModel::_treeitem *item = static_cast<TriTreeModel::_treeitem*> ( ( *ind_it ).internalPointer() );

    switch ( action )
    {

      case TVA_SURF             :

        if ( item->node->m_bShowSurface )
          ++num_checked_items;
        else
          ++num_unchecked_items;

        break;

      case TVA_EXTERIOR             :
        if ( item->node->m_bShowExterior )
          ++num_checked_items;
        else
          ++num_unchecked_items;

        break;


      case TVA_CPS              :
        if ( item->node->m_bShowCps )
          ++num_checked_items;
        else
          ++num_unchecked_items;

        break;

      case TVA_GRAPH            :
        if ( item->node->m_bShowMsGraph )
          ++num_checked_items;
        else
          ++num_unchecked_items;

        break;

      case TVA_QUADS            :
        if ( item->node->m_bShowMsQuads )
          ++num_checked_items;
        else
          ++num_unchecked_items;

        break;

      case TVA_GRAD             :
        if ( item->node->m_bShowGrad )
          ++num_checked_items;
        else
          ++num_unchecked_items;

        break;

      case TVA_CANCELABLE_PAIRS :
        if ( item->node->m_bShowCancelablePairs )
          ++num_checked_items;
        else
          ++num_unchecked_items;

        break;
    }
  }

  if ( num_checked_items > num_unchecked_items )
    return true;
  else
    return false;
}



void TriDataManager_ui::perform_tva_action ( const eTreeViewActions &action, const bool & state )
{

  QModelIndexList indexes = datapiece_treeView->selectionModel()->selectedIndexes();

  for ( QModelIndexList::iterator ind_it = indexes.begin();ind_it != indexes.end(); ++ind_it )
  {
    TriTreeModel::_treeitem *item = static_cast<TriTreeModel::_treeitem*> ( ( *ind_it ).internalPointer() );

    switch ( action )
    {

      case TVA_SURF             :
        item->node->m_bShowSurface            = state;
        break;

      case TVA_EXTERIOR         :
        item->node->m_bShowExterior           = state;
        break;

      case TVA_CPS              :
        item->node->m_bShowCps                = state;
        break;

      case TVA_GRAPH            :
        item->node->m_bShowMsGraph            = state;
        break;

      case TVA_QUADS            :
        item->node->m_bShowMsQuads            = state;
        break;

      case TVA_GRAD             :
        item->node->m_bShowGrad               = state;
        break;

      case TVA_CANCELABLE_PAIRS :
        item->node->m_bShowCancelablePairs    = state;
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
   


void TriDataManager_ui::on_datapiece_treeView_customContextMenuRequested ( const QPoint &pos )
{

  QMenu menu;

  ADD_MENU_ACTION ( &menu, "show surface", get_tva_state ( TVA_SURF ), show_surf_toggled );
  ADD_MENU_ACTION ( &menu, "show exterior", get_tva_state ( TVA_EXTERIOR ), show_exterior_toggled );
  ADD_MENU_ACTION ( &menu, "show critical points", get_tva_state ( TVA_CPS ), show_cps_toggled );
  ADD_MENU_ACTION ( &menu, "show graph", get_tva_state ( TVA_GRAPH ), show_graph_toggled );
  ADD_MENU_ACTION ( &menu, "show quads", get_tva_state ( TVA_QUADS ), show_quads_toggled );
  ADD_MENU_ACTION ( &menu, "show grad", get_tva_state ( TVA_GRAD ), show_grad_toggled );
  ADD_MENU_ACTION ( &menu, "show cancelable pairs", get_tva_state ( TVA_CANCELABLE_PAIRS ), show_cancelable_pairs_toggled );

  menu.exec ( datapiece_treeView->mapToGlobal ( pos ) );

}



void TriDataManager_ui::on_show_critical_point_labels_checkBox_stateChanged ( int state )
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

void TriDataManager_ui::on_show_critical_points_checkBox_stateChanged ( int state )
{
  switch ( state )
  {

    case Qt::Checked:
      m_bShowCriticalPoints = true;
      break;

    case Qt::Unchecked:
      m_bShowCriticalPoints  = false;
      break;

    default:
      break;
  }
}

void TriDataManager_ui::on_show_msgraph_checkBox_stateChanged ( int state )
{
  switch ( state )
  {

    case Qt::Checked:
      m_bShowMsGraph = true;
      break;

    case Qt::Unchecked:
      m_bShowMsGraph  = false;
      break;

    default:
      break;
  }
}


IModel* CreateTriDataManager_ui ( int argc, char *argv[] )
{

  std::stringstream ss;

  for ( int i = 0 ; i < argc;i++ )
    ss << argv[i] << " ";

  IModel * model = new TriDataManager_ui ( ss.str() );

  return model;
}

TriTreeModel::TriTreeModel ( tree<TriDataManager::_data_piece  *> * dpTree, QObject *parent )
    : QAbstractItemModel ( parent )
{
  setupModelData ( dpTree );
}

TriTreeModel::~TriTreeModel()
{
  delete m_tree;
  m_tree = NULL;
}


int TriTreeModel::columnCount ( const QModelIndex &/*parent*/ ) const
{
  return 1;
}

QVariant TriTreeModel::data ( const QModelIndex &index, int role ) const
{
  if ( !index.isValid() )
    return QVariant();

  if ( role != Qt::DisplayRole )
    return QVariant();

  _treeitem *item = static_cast<_treeitem*> ( index.internalPointer() );

  return item->node->level;
}

Qt::ItemFlags TriTreeModel::flags ( const QModelIndex &index ) const
{
  if ( !index.isValid() )
    return 0;

  return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

QVariant TriTreeModel::headerData ( int /*section*/, Qt::Orientation orientation,
                                    int role ) const
{
  if ( orientation == Qt::Horizontal && role == Qt::DisplayRole )
    return "Data Pieces";

  return QVariant();
}

QModelIndex TriTreeModel::index ( int row, int column, const QModelIndex &parent ) const
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

QModelIndex TriTreeModel::parent ( const QModelIndex &index ) const
{
  if ( !index.isValid() )
    return QModelIndex();

  _treeitem *childItem  = static_cast<_treeitem*> ( index.internalPointer() );

  _treeitem *parentItem = childItem->parent;

  if ( parentItem == m_tree )
    return QModelIndex();

  return createIndex ( parentItem->row(), 0, parentItem );

}

int TriTreeModel::rowCount ( const QModelIndex &parent ) const
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

void TriTreeModel::setupModelData ( tree<TriDataManager::_data_piece *> * dpTree )
{
  m_tree = new _treeitem();

  std::map<TriDataManager::_data_piece *, _treeitem *> dp_treeitem_map;

  for ( tree<TriDataManager::_data_piece *>::iterator dp_it =  dpTree->begin(); dp_it != dpTree->end(); ++dp_it )
  {
    TriDataManager::_data_piece *dp = *dp_it;

    tree<TriDataManager::_data_piece *>::iterator parentdp_it =  tree<TriDataManager::_data_piece *>::parent ( dp_it );

    _treeitem *parentItem = m_tree;

    if ( dpTree->is_valid ( parentdp_it ) )
    {
      parentItem = dp_treeitem_map[*parentdp_it];
    }

    _treeitem * dpItem = new _treeitem ( dp, parentItem );

    parentItem->children.push_back ( dpItem );

    dp_treeitem_map[dp] = dpItem;

  }
}

