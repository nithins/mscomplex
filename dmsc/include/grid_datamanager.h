/***************************************************************************
 *   Copyright (C) 2009 by nithin,,,   *
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

#ifndef GRID_DATAMANAGER_H_INCLUDED_
#define GRID_DATAMANAGER_H_INCLUDED_

#include <fstream>
#include <vector>

#include <QFrame>

#include <tree.h>

#include <model.h>
#include <input.h>

#include <rectangle_complex.h>
#include <discreteMorseDS.h>
#include <grid_dataset.h>

class IModelController;

namespace Ui
{
  class GridDataManager_QtFrame;
}

struct GridDataPiece
{
  typedef GridDataset::cell_fn_t cell_fn_t;
  typedef GridDataset::mscomplex_t mscomplex_t;
  typedef GridDataset::cell_coord_t cell_coord_t;
  typedef GridDataset::rect_t rect_t;
  typedef GridDataset::rect_size_t rect_size_t;
  typedef GridDataset::cellid_t cellid_t;
  typedef GridDataset::critpt_conn_t conn_t;

  GridDataset *dataset;
  mscomplex_t *msgraph;

  uint level;

  bool m_bShowSurface;
  bool m_bShowCps;
  bool m_bShowMsGraph;
  bool m_bShowGrad;

  glutils::renderable_t  *ren_surf;
  glutils::renderable_t  *ren_grad;
  glutils::renderable_t  *ren_cp_labels[3];
  glutils::renderable_t  *ren_cp[3];
  glutils::renderable_t  *ren_cp_conns[2];

  void create_cp_rens();
  void create_grad_rens();
  void create_surf_ren();

  GridDataPiece ( ) ;

  std::string label();
};

class GridDataManager:
    virtual public QFrame,
    virtual public IModel,
    virtual public IHandleInput

{

  typedef GridDataset::rect_t rect_t;
  typedef GridDataset::cell_coord_t cell_coord_t;
  typedef GridDataset::cellid_t cellid_t;
  typedef GridDataset::rect_point_t rect_point_t;
  typedef GridDataset::rect_size_t rect_size_t;
  typedef std::vector<GridDataPiece *> pieces_list_t;

public:

  pieces_list_t                m_pieces;

  bool                         m_bShowCriticalPointLabels;
  std::string                  m_filename;
  u_int                        m_size_x;
  u_int                        m_size_y;
  u_int                        m_num_levels;
  bool                         m_single_threaded_mode;
  bool                         m_use_ocl;

  IModelController            *m_controller;

public:

  GridDataManager
      ( std::string filename,
        u_int        size_x,
        u_int        size_y,
        u_int        num_levels,
        bool         threaded_mode,
        bool         use_ocl);

  virtual ~GridDataManager ();

  void createPieces_quadtree(rect_t r,rect_t e,u_int level );

  void createDataPieces();

  void readFile ();

  void workPiece ( GridDataPiece  * );

  void workAllPieces_mt( );

  void workPiecesInRange_mt(uint ,uint );

  void workAllPieces_st( );

  void clearInteriorGrad(uint start,uint end );

  void mergePiecesUp_mt( );

  void mergePiecesUp_st( );

  void mergePiecesDown_mt( );

  void mergePiecesDown_st( );

  void renderDataPiece ( GridDataPiece  *dp ) const;

  void logAllConnections(const std::string &prefix);

  void logAllCancelPairs(const std::string &prefix);

  // IModel Interface

public:

  virtual int Render() const;

  virtual std::string Name() const {return std::string("Grid");}

  //IInputHandler Interface

public:

  bool MousePressedEvent
      ( const int &x, const int &y, const eMouseButton &mb,
        const eKeyFlags &,const eMouseFlags &);

  bool MouseReleasedEvent
      ( const int &x, const int &y, const eMouseButton &mb,
        const eKeyFlags &,const eMouseFlags &);

  bool MouseMovedEvent
      ( const int &x, const int &y, const int &, const int &,
        const eKeyFlags &,const eMouseFlags &mf);

  virtual bool WheelEvent
      ( const int &, const int &, const int &d,
        const eKeyFlags &kf,const eMouseFlags &);


  // Ui stuff

  virtual QFrame * getQFrame() { return this;}

private:

  Q_OBJECT

  enum eTreeViewActions
  {
    TVA_SURF,
    TVA_CPS,
    TVA_GRAPH,
    TVA_QUADS,
    TVA_GRAD,
    TVA_CANCELABLE_PAIRS
  };

  void perform_tva_action ( const eTreeViewActions &,const bool & );
  bool get_tva_state ( const eTreeViewActions & );

  void create_ui();
  void destroy_ui();

  Ui::GridDataManager_QtFrame *m_ui;

private slots:
  void on_show_critical_point_labels_checkBox_stateChanged ( int state );
  void on_datapiece_treeView_customContextMenuRequested ( const QPoint &p );

  void show_surf_toggled ( bool state ) {perform_tva_action ( TVA_SURF,state );}
  void show_cps_toggled ( bool state ) {perform_tva_action ( TVA_CPS,state );}
  void show_graph_toggled ( bool state ) {perform_tva_action ( TVA_GRAPH,state );}
  void show_quads_toggled ( bool state ) {perform_tva_action ( TVA_QUADS,state );}
  void show_grad_toggled ( bool state ) {perform_tva_action ( TVA_GRAD,state );}
  void show_cancelable_pairs_toggled ( bool state ) {perform_tva_action ( TVA_CANCELABLE_PAIRS,state );}

};

#endif
