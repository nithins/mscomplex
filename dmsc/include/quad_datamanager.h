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

#ifndef QUAD_DATAMANAGER_H_INCLUDED_
#define QUAD_DATAMANAGER_H_INCLUDED_

#include <fstream>
#include <vector>

#include <QFrame>

#include <tree.h>

#include <model.h>
#include <input.h>

#include <rectangle_complex.h>
#include <discreteMorseDS.h>
#include <quad_dataset.h>

class IModelController;

namespace Ui
{
  class QuadDataManager_QtFrame;
}

struct DataPiece
{
  typedef rectangle_complex<uint>::point_def       point_t;
  typedef rectangle_complex<uint>::size_def        size_t;
  typedef rectangle_complex<uint>::rectangle_def   rectangle_t;
  typedef QuadGenericCell2D<uint>                  generic_cell_t;
  typedef MSComplex<generic_cell_t>                generic_mscomplex_t;
  typedef std::pair<generic_cell_t,generic_cell_t> cancel_pair_t;

  QuadDataset             q;
  rectangle_complex<uint> c;
  generic_mscomplex_t *mscomplex;

  uint *vertlist;
  uint *indlist;
  uint vert_ct;
  uint quad_ct;
  uint ext_vert_ct;
  uint ext_quad_ct;

  uint level;
  std::string label;

  bool m_bShowSurface;
  bool m_bShowCps;
  bool m_bShowMsGraph;
  bool m_bShowMsQuads;
  bool m_bShowGrad;
  bool m_bShowCancelablePairs;

  std::vector<cancel_pair_t> cancellable_boundry_pairs;

  ArrayRenderer<unsigned int ,double> *ren_surf;
  ArrayRenderer<unsigned int ,double> *ren_cancelpairs;
  ArrayRenderer<unsigned int ,double> *ren_grad0;
  ArrayRenderer<unsigned int ,double> *ren_grad1;

  ArrayRenderer<unsigned int ,double> *ren_msgraph;
  ArrayRenderer<unsigned int ,double> *ren_msquads;

  glutils::renderable_t               *ren_cp_labels[3];
  glutils::renderable_t               *ren_cp[3];
  glutils::renderable_t               *ren_cp_conns[2];

  void create_cp_rens(double *data,uint size_x,uint size_y);

  DataPiece ( rectangle_t rec ) ;

};

class QuadDataManager:
    virtual public QFrame,
    virtual public IModel,
    virtual public IHandleInput

{
public:

  typedef rectangle_complex<uint>::point_def       point_t;
  typedef rectangle_complex<uint>::size_def        size_t;
  typedef rectangle_complex<uint>::rectangle_def   rectangle_t;
  typedef QuadGenericCell2D<uint>                  generic_cell_t;
  typedef MSComplex<generic_cell_t>                generic_mscomplex_t;
  typedef std::map<generic_cell_t,generic_cell_t>  cancel_pair_map_t;


  ArrayRenderer<generic_cell_t ,double>         *ren_crit;
  ArrayRenderer<generic_cell_t ,double>         *ren_msgraph;
  std::vector<DataPiece *>                       m_pieces;
  tree<DataPiece *>                              m_dpTree;
  double                                        *m_pData;


  std::string                                    m_filename;
  uint                                           m_size_x;
  uint                                           m_size_y;
  uint                                           m_buf_zone_size;
  uint                                           m_max_levels;
  bool                                           m_single_threaded_mode;

  bool                                           m_bShowCriticalPoints;
  bool                                           m_bShowCriticalPointLabels;
  bool                                           m_bShowMsGraph;

  ArrayRenderer<generic_cell_t ,double>         *m_ren_disc;
  uint                                           m_disc_critpt_no;
  eGradDirection                                 m_disc_grad_type;

  IModelController                              *m_controller;

public:

  QuadDataManager
      ( std::string filename,
        uint        size_x,
        uint        size_y,
        uint        buf_zone_size,
        uint        max_levels,
        bool        threaded_mode);

  virtual ~QuadDataManager ();

  void createRectangleComplexes ( const rectangle_t & rec,uint level,
                                  tree<DataPiece *>::iterator dpTree_pos );

  void setDataPieceLabels();

  void readFile ();

  void initDataPieceForWork ( DataPiece *dp );

  void initDataPieceForRender ( DataPiece *dp );

  void workPiece ( DataPiece * );

  void mergeChildMSGraphs ( tree<DataPiece *>::iterator dp_it );

  void mergeDownChildMSGraphs ( tree<DataPiece *>::iterator dp_it );

  void workAllPieces_mt( );

  void workAllPieces_st( );

  void workPiecesAtFixedLevel_mt ( int i );

  void workPiecesAtFixedLevel_st ( int i );

  void mergePiecesAtFixedLevel_mt ( int i );

  void mergePiecesAtFixedLevel_st ( int i );

  void mergeDownPiecesAtFixedLevel_st( int i );

  void renderDataPiece ( DataPiece *dp ) const;

  void logAllConnections(const std::string &prefix);

  void logAllCancelPairs(const std::string &prefix);

  // IModel Interface

public:

  virtual int Render() const;

  virtual std::string Name() const {return std::string("QuadGrid");}

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

  void update_disc();

  Ui::QuadDataManager_QtFrame *m_ui;

private slots:

  void on_show_critical_point_labels_checkBox_stateChanged ( int state );
  void on_show_critical_points_checkBox_stateChanged ( int state );
  void on_show_msgraph_checkBox_stateChanged ( int state );
  void on_datapiece_treeView_customContextMenuRequested ( const QPoint &p );

  void on_critical_point_number_spinBox_valueChanged(int i);
  void on_critical_point_disc_type_comboBox_currentIndexChanged(int i);

  void show_surf_toggled ( bool state ) {perform_tva_action ( TVA_SURF,state );}
  void show_cps_toggled ( bool state ) {perform_tva_action ( TVA_CPS,state );}
  void show_graph_toggled ( bool state ) {perform_tva_action ( TVA_GRAPH,state );}
  void show_quads_toggled ( bool state ) {perform_tva_action ( TVA_QUADS,state );}
  void show_grad_toggled ( bool state ) {perform_tva_action ( TVA_GRAD,state );}
  void show_cancelable_pairs_toggled ( bool state ) {perform_tva_action ( TVA_CANCELABLE_PAIRS,state );}

};

#endif
