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

#include <quad_datamanager.h>
#include <quad_datamanager_ui.h>
#include <ui_quad_datamanager_frame.h>

using namespace std;

bool QuadDataManager::get_tva_state ( const eTreeViewActions &action )
{

  uint num_checked_items = 0;
  uint num_unchecked_items = 0;

  QModelIndexList indexes = m_ui->datapiece_treeView->selectionModel()->selectedIndexes();

  for ( QModelIndexList::iterator ind_it = indexes.begin();ind_it != indexes.end(); ++ind_it )
  {
    TreeModel::_treeitem *item = static_cast<TreeModel::_treeitem*> ( ( *ind_it ).internalPointer() );

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

    case TVA_QUADS            :
      if ( item->node->m_bShowMsQuads ) ++num_checked_items;
      else ++num_unchecked_items;
      break;

    case TVA_GRAD             :
      if ( item->node->m_bShowGrad ) ++num_checked_items;
      else ++num_unchecked_items;
      break;

    case TVA_CANCELABLE_PAIRS :
      if ( item->node->m_bShowCancelablePairs )++num_checked_items;
      else ++num_unchecked_items;
      break;
    }
  }

  if ( num_checked_items > num_unchecked_items )
    return true;
  else
    return false;
}



void QuadDataManager::perform_tva_action ( const eTreeViewActions &action, const bool & state )
{

  QModelIndexList indexes = m_ui->datapiece_treeView->selectionModel()->selectedIndexes();

  for ( QModelIndexList::iterator ind_it = indexes.begin();ind_it != indexes.end(); ++ind_it )
  {
    TreeModel::_treeitem *item = static_cast<TreeModel::_treeitem*> ( ( *ind_it ).internalPointer() );

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
    case TVA_QUADS            :
      item->node->m_bShowMsQuads            = state;
      break;
    case TVA_GRAD             :
      item->node->m_bShowGrad               = state;
      break;
    case TVA_CANCELABLE_PAIRS :
      item->node->m_bShowCancelablePairs    = state;
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



void QuadDataManager::on_datapiece_treeView_customContextMenuRequested ( const QPoint &pos )
{

  QMenu menu;

  ADD_MENU_ACTION ( &menu, "show surface", get_tva_state ( TVA_SURF ), show_surf_toggled );
  ADD_MENU_ACTION ( &menu, "show critical points", get_tva_state ( TVA_CPS ), show_cps_toggled );
  ADD_MENU_ACTION ( &menu, "show graph", get_tva_state ( TVA_GRAPH ), show_graph_toggled );
  ADD_MENU_ACTION ( &menu, "show quads", get_tva_state ( TVA_QUADS ), show_quads_toggled );
  ADD_MENU_ACTION ( &menu, "show grad", get_tva_state ( TVA_GRAD ), show_grad_toggled );
  ADD_MENU_ACTION ( &menu, "show cancelable pairs", get_tva_state ( TVA_CANCELABLE_PAIRS ), show_cancelable_pairs_toggled );

  menu.exec ( m_ui->datapiece_treeView->mapToGlobal ( pos ) );

}



void QuadDataManager::on_show_critical_point_labels_checkBox_stateChanged ( int state )
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

void QuadDataManager::on_show_critical_points_checkBox_stateChanged ( int state )
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

void QuadDataManager::on_show_msgraph_checkBox_stateChanged ( int state )
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


string toString_eGradDirection(eGradDirection e)
{
  switch(e)
  {
  case GRADIENT_DIR_DOWNWARD: return "Descending Manifold";
  case GRADIENT_DIR_UPWARD: return "Ascending Manifold";
  default: _ERROR("Unknownd grad dir string requested");return "unknown";
  }
}

void QuadDataManager::create_ui()
{
  m_ui = new Ui::QuadDataManager_QtFrame();

  m_ui->setupUi ( this );

  TreeModel *model = new TreeModel ( &m_dpTree );

  RecursiveTreeItemSelectionModel * sel_model = new RecursiveTreeItemSelectionModel ( model, m_ui->datapiece_treeView );

  m_ui->datapiece_treeView->setModel ( model );

  m_ui->datapiece_treeView->setSelectionModel ( sel_model );

  for(int i = 0 ; i < (int)GRADIENT_DIR_COUNT;++i)
  {
    m_ui->critical_point_disc_type_comboBox->addItem
        (toString_eGradDirection((eGradDirection)i).c_str());
  }

  DataPiece *dp_root = *m_dpTree.begin();

  m_ui->critical_point_number_spinBox->setMaximum(dp_root->mscomplex->m_cp_count-1);

}

void QuadDataManager::on_critical_point_number_spinBox_valueChanged(int i)
{
  m_disc_critpt_no = i;
  update_disc();
}

void QuadDataManager::on_critical_point_disc_type_comboBox_currentIndexChanged(int i)
{
  m_disc_grad_type = (eGradDirection)i;
  update_disc();
}


void QuadDataManager::destroy_ui()
{
  delete m_ui;
}

TreeModel::TreeModel ( tree<DataPiece *> * dpTree, QObject *parent )
  : QAbstractItemModel ( parent )
{
  setupModelData ( dpTree );
}

TreeModel::~TreeModel()
{
  delete m_tree;
  m_tree = NULL;
}


int TreeModel::columnCount ( const QModelIndex &/*parent*/ ) const
{
  return 1;
}

QVariant TreeModel::data ( const QModelIndex &index, int role ) const
{
  if ( !index.isValid() )
    return QVariant();

  if ( role != Qt::DisplayRole )
    return QVariant();

  _treeitem *item = static_cast<_treeitem*> ( index.internalPointer() );

  return item->node->level;
}

Qt::ItemFlags TreeModel::flags ( const QModelIndex &index ) const
{
  if ( !index.isValid() )
    return 0;

  return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

QVariant TreeModel::headerData ( int /*section*/, Qt::Orientation orientation,
                                 int role ) const
{
  if ( orientation == Qt::Horizontal && role == Qt::DisplayRole )
    return "Data Pieces";

  return QVariant();
}

QModelIndex TreeModel::index ( int row, int column, const QModelIndex &parent ) const
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

QModelIndex TreeModel::parent ( const QModelIndex &index ) const
{
  if ( !index.isValid() )
    return QModelIndex();

  _treeitem *childItem  = static_cast<_treeitem*> ( index.internalPointer() );

  _treeitem *parentItem = childItem->parent;

  if ( parentItem == m_tree )
    return QModelIndex();

  return createIndex ( parentItem->row(), 0, parentItem );

}

int TreeModel::rowCount ( const QModelIndex &parent ) const
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

void TreeModel::setupModelData ( tree<DataPiece *> * dpTree )
{
  m_tree = new _treeitem();

  std::map<DataPiece *, _treeitem *> dp_treeitem_map;

  for ( tree<DataPiece*>::iterator dp_it =  dpTree->begin(); dp_it != dpTree->end(); ++dp_it )
  {
    DataPiece *dp = *dp_it;

    tree<DataPiece*>::iterator parentdp_it =  tree<DataPiece*>::parent ( dp_it );

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

