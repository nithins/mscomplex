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

#include <iostream>
#include <fstream>

#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/regex.hpp>

#include <modelcontroller.h>

#include <quad_datamanager.h>
#include <discreteMorseAlgorithm.h>
#include <discreteMorseDSRenderFuncs.h>
#include <quad_dataset.h>

using namespace std;

DataPiece::DataPiece ( rectangle_t rec ) :
    c ( rec ),
    mscomplex(NULL),
    vertlist(NULL),
    indlist(NULL),
    vert_ct(0),
    quad_ct(0),
    ext_vert_ct(0),
    ext_quad_ct(0),
    level(0),
    m_bShowSurface ( false ),
    m_bShowCps ( false ),
    m_bShowMsGraph ( false ),
    m_bShowMsQuads ( false ),
    m_bShowGrad ( false ),
    m_bShowCancelablePairs ( false ),
    ren_surf(NULL),
    ren_cancelpairs(NULL),
    ren_grad0(NULL),
    ren_grad1(NULL),
    ren_msgraph(NULL),
    ren_msquads(NULL)
{
  memset(ren_cp,0,sizeof(ren_cp));
  memset(ren_cp,0,sizeof(ren_cp_labels));
}


void QuadDataManager::createRectangleComplexes ( const rectangle_t & rec,uint level,tree<DataPiece *>::iterator dpTree_parNode_it )
{

  _LOG_V ( "level = "<<level );
  _LOG_V ( "rec   = "<<rec );
  _LOG_V ( "-----------------" );


  DataPiece * dp = new DataPiece ( rec );
  dp->level = level;
  tree<DataPiece *>::iterator dpTreeNode_it;

  if ( level == m_max_levels -1 )
    dpTreeNode_it = m_dpTree.set_head ( dp );
  else
    dpTreeNode_it = m_dpTree.append_child ( dpTree_parNode_it,dp );


  m_pieces.push_back ( dp );

  if ( level <= 0 ) return ;

  rectangle_t subrec1 ( rec.bottom_left().add ( size_t ( 2*m_buf_zone_size,0 ) ).add ( size_t ( 0,2*m_buf_zone_size ) ),
                        rec.mid().sub ( size_t ( 1*m_buf_zone_size,0 ) ).sub ( size_t ( 0,1*m_buf_zone_size ) ) );

  dp->c.remove_region ( subrec1 );

  createRectangleComplexes ( subrec1,level-1,dpTreeNode_it );


  rectangle_t subrec2 ( rec.bottom_right().sub ( size_t ( 2*m_buf_zone_size,0 ) ).add ( size_t ( 0,2*m_buf_zone_size ) ),
                        rec.mid().add ( size_t ( 1*m_buf_zone_size,0 ) ).sub ( size_t ( 0,1*m_buf_zone_size ) ) );

  dp->c.remove_region ( subrec2 );

  createRectangleComplexes ( subrec2,level-1,dpTreeNode_it );


  rectangle_t subrec3 ( rec.top_right().sub ( size_t ( 2*m_buf_zone_size,0 ) ).sub ( size_t ( 0,2*m_buf_zone_size ) ),
                        rec.mid().add ( size_t ( 1*m_buf_zone_size,0 ) ).add ( size_t ( 0,1*m_buf_zone_size ) ) );

  dp->c.remove_region ( subrec3 );

  createRectangleComplexes ( subrec3,level-1,dpTreeNode_it );


  rectangle_t subrec4 ( rec.top_left().add ( size_t ( 2*m_buf_zone_size,0 ) ).sub ( size_t ( 0,2*m_buf_zone_size ) ),
                        rec.mid().sub ( size_t ( 1*m_buf_zone_size,0 ) ).add ( size_t ( 0,1*m_buf_zone_size ) ) );

  dp->c.remove_region ( subrec4 );

  createRectangleComplexes ( subrec4,level-1 ,dpTreeNode_it );

  return;
}

void QuadDataManager::setDataPieceLabels()
{

  static const char * child_label_prefixes[] =  {"TL","TR","BR","BL"};

  DataPiece * dpRoot = (*m_dpTree.begin());
  dpRoot->label = "Root";


  for(uint level = 0; level < m_dpTree.max_depth();++level)
  {
    for ( tree<DataPiece*>::fixed_depth_iterator dp_it = m_dpTree.begin_fixed ( m_dpTree.begin(),level );
    m_dpTree.is_valid ( dp_it );++dp_it )
    {
      DataPiece * parent_ptr = *dp_it;

      uint count = 0 ;

      for ( tree<DataPiece*>::sibling_iterator child_it = m_dpTree.begin ( dp_it );
      child_it != m_dpTree.end ( dp_it ); ++child_it )
      {
        DataPiece * child_ptr = *child_it;

        child_ptr->label = parent_ptr->label+
                           std::string(".")+
                           std::string(child_label_prefixes[count]);

        ++count;

        if(count > 4)
          throw std::length_error("count is >= 4 ");
      }
    }
  }


}

void QuadDataManager::readFile ( )
{
  m_pData = new double[m_size_x*m_size_y];

  fstream infile ( m_filename.c_str(),fstream::in|fstream::binary );

  for ( uint i = 0 ; i < m_size_x*m_size_y;i++ )
  {
    if ( infile.eof() )
    {
      _ERROR ( "Premature end of file" );
      _ERROR ( "Expected "<<m_size_y*m_size_x<<" values" );
      _ERROR ( "got      "<<i  <<" values" );

      throw std::length_error(string(" Premature end of file "));
    }
    infile.read ( reinterpret_cast<char *> ( m_pData +i ),sizeof ( double ) );
  }
}

void QuadDataManager::initDataPieceForWork ( DataPiece *dp )
{
  dp->c.createVertAndQuadIndexLists ( dp->vertlist,
                                      dp->indlist,
                                      dp->vert_ct,
                                      dp->quad_ct,
                                      dp->ext_vert_ct,
                                      dp->ext_quad_ct ,
                                      ( dp->level != m_max_levels-1 ) );

  dp->q.setNumVerts ( dp->vert_ct );

  dp->q.setNumQuads ( dp->quad_ct );

  dp->q.setNumExtVert ( dp->ext_vert_ct );

  dp->q.setNumExtQuad ( dp->ext_quad_ct );

  dp->q.startAddingQuads();

  for ( uint i = 0 ; i < dp->quad_ct;i++ )
  {
    dp->q.addQuad ( dp->indlist+i*4,i,0 );
  }

  dp->q.endAddingQuads();

  for ( uint i = 0 ; i < dp->vert_ct ;i++ )
  {
    uint x     = dp->vertlist[2*i+0];
    uint y     = dp->vertlist[2*i+1];
    uint ind   = y*m_size_x+x;
    double val = m_pData[ind];

    dp->q.setVert ( i,val,ind, ( double ) x/ ( double ) m_size_x,val*0.1, ( double ) y/ ( double ) m_size_y );
  }

  delete []dp->vertlist;
  delete []dp->indlist;
}

void QuadDataManager::initDataPieceForRender ( DataPiece *dp )
{
  dp->ren_surf        = dp->q.createSurfaceRenderer();
  dp->ren_grad0       = dp->q.createGradientRenderer ( 0 );
  dp->ren_grad1       = dp->q.createGradientRenderer ( 1 );
  dp->ren_msgraph     = dp->q.createMsGraphRenderer();
  dp->ren_msquads     = dp->q.createMsQuadsRenderer();
  dp->ren_cancelpairs = dp->q.createCancellablePairsRenderer();

}

void QuadDataManager::workPiece ( DataPiece *dp )
{
  initDataPieceForWork ( dp );

  dp->q.assignGradients();
  dp->q.computeDiscs ();

  dp->mscomplex = new DataPiece::generic_mscomplex_t;

  dp->q.getGenericMsComplex(dp->mscomplex);

  dp->q.getBoundryCancellablePairs(dp->cancellable_boundry_pairs);

  if(m_single_threaded_mode)
    initDataPieceForRender(dp);

  dp->q.destroy();
}

typedef QuadGenericCell2D<uint> generic_cell_t;

void QuadDataManager::mergeChildMSGraphs ( tree<DataPiece *>::iterator dp_it )
{
  DataPiece *dp = *dp_it;

  for ( tree<DataPiece*>::sibling_iterator child_it = m_dpTree.begin ( dp_it );
  child_it != m_dpTree.end ( dp_it ); ++child_it )
  {
    DataPiece *child = *child_it;

    union_complexes_up(dp->mscomplex,child->mscomplex,
                       child->cancellable_boundry_pairs);

  }
}

void QuadDataManager::mergeDownChildMSGraphs ( tree<DataPiece *>::iterator dp_it )
{
  DataPiece *dp = *dp_it;

  DataPiece *dp_childs[4];

  uint child_count = 0;

  cancel_pair_map_t cancel_pair_map;

  form_cancel_pairs_map(dp->cancellable_boundry_pairs,cancel_pair_map);


  for ( tree<DataPiece*>::sibling_iterator child_it = m_dpTree.begin ( dp_it );
  child_it != m_dpTree.end ( dp_it ); ++child_it )
  {
    DataPiece *child = *child_it;

    dp_childs[child_count++] = child;

    form_cancel_pairs_map(child->cancellable_boundry_pairs,cancel_pair_map);

    if(child_count > 4)
      throw std::out_of_range("there shouldnt be more than 4 children");
  }

  for(int i = 3 ; i >=0 ; --i)
  {

    DataPiece *child = dp_childs[i];

    _LOG("down mergeing "<<dp->label<<" and "<<child->label);

    union_complexes_down(dp->mscomplex,child->mscomplex,
                          child->cancellable_boundry_pairs,
                          cancel_pair_map);
  }

  dp->cancellable_boundry_pairs.clear();

}


struct get3DCoordsFromList_ftor
{

  const double *list;
  const uint size_x,size_y;

  typedef void result_type ;

  get3DCoordsFromList_ftor ( const double *_list,const uint &_size_x,const uint &_size_y )
      :list ( _list ),size_x ( _size_x ),size_y ( _size_y )
  {
  }

  void operator() ( generic_cell_t cell, double &x ,double &y,double &z )
  {
    x=0;y=0;z=0;

    for ( uint i = 0 ; i < cell.numv() ;i++ )
    {
      uint index = cell._v[i];

      x += ( double ) ( index%size_x ) / ( double ) size_x;
      y += list[index]*0.1;
      z += ( double ) ( index/size_y ) / ( double ) size_y;
    }

    x/=cell.numv();
    y/=cell.numv();
    z/=cell.numv();
  }

};

void QuadDataManager::workAllPieces_mt( )
{
  _LOG ( "Begin calculating asc/des manifolds for all pieces " );

  boost::thread ** threads = new boost::thread*[ m_pieces.size() ];

  uint threadno = 0;

  for ( uint i = 0 ; i < m_pieces.size();i++ )
  {
    _LOG ( "Kicking off thread "<<threadno );

    DataPiece * dp = m_pieces[i];
    threads[threadno] = new boost::thread ( boost::bind ( &QuadDataManager::workPiece,this,dp ) );
    threadno++;
  }

  threadno = 0;

  for ( uint i = 0 ; i < m_pieces.size();i++ )
  {
    threads[threadno]->join();
    _LOG ( "thread "<<threadno<<" joint" );
    threadno++;
  }

  delete []threads;

  _LOG ( "End calculating asc/des manifolds for all pieces " );
}

void QuadDataManager::workAllPieces_st( )
{
  _LOG ( "Begin calculating asc/des manifolds for all pieces " );

  for ( uint i = 0 ; i < m_pieces.size();i++ )
  {
    DataPiece * dp = m_pieces[i];

    workPiece(dp);
  }

  _LOG ( "End calculating asc/des manifolds for all pieces " );
}

void QuadDataManager::workPiecesAtFixedLevel_mt ( int level )
{
  _LOG ( "Begin calculating asc/des manifolds for level = "<< level );

  int num_threads = pow ( 4,level );

  boost::thread ** threads = new boost::thread*[ num_threads];

  uint threadno = 0;

  for ( tree<DataPiece*>::fixed_depth_iterator dp_it = m_dpTree.begin_fixed ( m_dpTree.begin(),level );
  m_dpTree.is_valid ( dp_it );++dp_it )
  {
    _LOG ( "Kicking off thread "<<threadno );

    DataPiece * dp = *dp_it;
    threads[threadno] = new boost::thread ( boost::bind ( &QuadDataManager::workPiece,this,dp ) );
    threadno++;
  }

  threadno = 0;

  for ( tree<DataPiece*>::fixed_depth_iterator dp_it = m_dpTree.begin_fixed ( m_dpTree.begin(),level );
  m_dpTree.is_valid ( dp_it );++dp_it )
  {
    threads[threadno]->join();
    _LOG ( "thread "<<threadno<<" joint" );
    threadno++;
  }

  delete []threads;

  _LOG ( "End calculating asc/des manifolds for level = "<< level );
}

void QuadDataManager::mergePiecesAtFixedLevel_mt ( int level )
{
  _LOG ( "Begin merge at level = "<< level );

  int num_threads = pow ( 4,level );

  boost::thread ** threads = new boost::thread*[ num_threads];

  uint threadno = 0;

  for ( tree<DataPiece*>::fixed_depth_iterator dp_it = m_dpTree.begin_fixed ( m_dpTree.begin(),level );
  m_dpTree.is_valid ( dp_it );++dp_it )
  {
    _LOG ( "Kicking off thread "<<threadno );

    threads[threadno] = new boost::thread ( boost::bind ( &QuadDataManager::mergeChildMSGraphs,this,dp_it ) );
    threadno++;
  }

  threadno =0;

  for ( tree<DataPiece*>::fixed_depth_iterator dp_it = m_dpTree.begin_fixed ( m_dpTree.begin(),level );
  m_dpTree.is_valid ( dp_it );++dp_it )
  {
    threads[threadno]->join();
    _LOG ( "thread "<<threadno<<" joint" );

    threadno++;
  }

  delete []threads;

  _LOG ( "End merge at level = "<< level );
}

void QuadDataManager::workPiecesAtFixedLevel_st ( int level )
{
  _LOG ( "Begin calculating asc/des manifolds for level = "<< level );

  uint count = 0;

  for ( tree<DataPiece*>::fixed_depth_iterator dp_it = m_dpTree.begin_fixed ( m_dpTree.begin(),level );
  m_dpTree.is_valid ( dp_it );++dp_it )
  {
    ++count;

    DataPiece * dp = *dp_it;
    workPiece ( dp );
  }

  _LOG ( "End calculating asc/des manifolds for level = "<< level );
}

void QuadDataManager::mergePiecesAtFixedLevel_st ( int level )
{
  _LOG ( "Begin merge at level = "<< level );

  for ( tree<DataPiece*>::fixed_depth_iterator dp_it = m_dpTree.begin_fixed ( m_dpTree.begin(),level );
  m_dpTree.is_valid ( dp_it );++dp_it )
  {
    mergeChildMSGraphs ( dp_it );
  }
  _LOG ( "End merge at level = "<< level );
}

void QuadDataManager::mergeDownPiecesAtFixedLevel_st ( int level )
{
  _LOG ( "Begin merge down at level = "<< level );

  for ( tree<DataPiece*>::fixed_depth_iterator dp_it = m_dpTree.begin_fixed ( m_dpTree.begin(),level );
  m_dpTree.is_valid ( dp_it );++dp_it )
  {
    mergeDownChildMSGraphs ( dp_it );
  }
  _LOG ( "End merge at down level = "<< level );
}




QuadDataManager::QuadDataManager
    ( std::string filename,
      uint        size_x,
      uint        size_y,
      uint        buf_zone_size,
      uint        max_levels,
      bool        single_threaded_mode):
    m_filename(filename),
    m_size_x(size_x),
    m_size_y(size_y),
    m_buf_zone_size(buf_zone_size),
    m_max_levels(max_levels),
    m_single_threaded_mode(single_threaded_mode)
{

  m_controller = IModelController::Create();

  createRectangleComplexes ( rectangle_t ( point_t ( 0,0 ),point_t ( m_size_x-1,m_size_y-1 ) ) ,max_levels-1,m_dpTree.begin() );

  setDataPieceLabels();

  readFile ();

  _LOG ( "==========================" );
  _LOG ( "Starting Processing Peices" );
  _LOG ( "--------------------------" );

  if ( m_single_threaded_mode == false )
  {
    workAllPieces_mt();

//    for ( int i = m_dpTree.max_depth()-1 ; i >=0 ;i-- )
//    {
//      mergePiecesAtFixedLevel_mt ( i );
//    }

    for ( int i = m_dpTree.max_depth()-1 ; i >=0 ;i-- )
    {
      mergePiecesAtFixedLevel_st ( i );
    }

    for ( int i = 0;i < m_dpTree.max_depth() ;i++ )
    {
      mergeDownPiecesAtFixedLevel_st( i );
    }
    exit(0);
  }
  else
  {

    workAllPieces_st();

    DataPiece *dp_root = *m_dpTree.begin();

    dp_root->cancellable_boundry_pairs.clear();

//    logAllConnections("log/pre-merge-");
//
//    logAllCancelPairs("log/cancel-pairs-");

    for ( int i = m_dpTree.max_depth()-1 ; i >=0 ;i-- )
    {
      mergePiecesAtFixedLevel_st ( i );

//      stringstream ss;
//      ss<<"log/post-merge-up-level-"<<i<<"-";

//      logAllConnections(ss.str());
    }

    for ( int i = 0;i < m_dpTree.max_depth() ;i++ )
    {
      mergeDownPiecesAtFixedLevel_st( i );

//      stringstream ss;
//      ss<<"log/post-merge-down-level-"<<i<<"-";
//
//      logAllConnections(ss.str());
    }
  }

  _LOG ( "--------------------------" );
  _LOG ( "Finished Processing peices" );
  _LOG ( "==========================" );

  for(uint i = 0 ; i <m_pieces.size();++i)
  {
    DataPiece *dp = m_pieces[i];

    dp->create_cp_rens(m_pData,m_size_x,m_size_y);
  }

  m_bShowCriticalPoints = true;
  m_bShowCriticalPointLabels = false;
  m_bShowMsGraph = true;

  DataPiece *dp_root = *m_dpTree.begin();

  createCritPtRen ( dp_root->mscomplex,ren_crit,
                    get3DCoordsFromList_ftor ( m_pData,m_size_x,m_size_y ),
                    boost::bind ( &generic_cell_t::dim,_1 )
                    );


  createCombinatorialStructureRenderer
      ( dp_root->mscomplex,
        ren_msgraph,
        get3DCoordsFromList_ftor ( m_pData,m_size_x,m_size_y ) );

  ren_msgraph->set_common_color ( 0.5,1.0,1.0 );

  m_disc_critpt_no = 0;
  m_disc_grad_type = GRADIENT_DIR_UPWARD;

  m_ren_disc       = NULL;
  update_disc();

  create_ui();
}

void QuadDataManager::logAllConnections(const std::string &prefix)
{
  for(uint i = 0 ; i <m_pieces.size();++i)
  {
    DataPiece *dp = m_pieces[i];

    fstream outfile((prefix+dp->label+string(".txt")).c_str(),ios::out);
    print_connections(*((std::ostream *) &outfile),*dp->mscomplex);
  }

}

void QuadDataManager::logAllCancelPairs(const std::string &prefix)
{
  for(uint i = 0 ; i <m_pieces.size();++i)
  {
    DataPiece *dp = m_pieces[i];
    fstream outfile((prefix+dp->label+string(".txt")).c_str(),ios::out);

    for(uint j = 0 ; j < dp->cancellable_boundry_pairs.size(); ++j)
    {
      outfile<<dp->cancellable_boundry_pairs[j].first<<" ";
      outfile<<dp->cancellable_boundry_pairs[j].second<<std::endl;
    }
  }

}

QuadDataManager::~QuadDataManager ()
{
  IModelController::Delete(m_controller);

  destroy_ui();
}

glutils::color_t g_cp_colors[] =
{
  glutils::color_t(1.0,0.0,0.0),
  glutils::color_t(0.0,1.0,0.0),
  glutils::color_t(0.0,0.0,1.0),
};


void QuadDataManager::renderDataPiece ( DataPiece *dp ) const
{
  glPushMatrix();
  glPushAttrib ( GL_ENABLE_BIT );

  glScalef ( 2.0,2.0,2.0 );
  glTranslatef ( -0.5,0.0,-0.5 );

  if ( dp->m_bShowSurface && dp->ren_surf)
  {
    glColor3f ( 0.75,0.75,0.75 );
    dp->ren_surf->render_triangles();
  }

  if ( dp->m_bShowGrad && dp->ren_grad0 && dp->ren_grad1)
  {
    glColor3f ( 0.7,0.7,0.2 );
    dp->ren_grad0->render_triangles();
    glColor3f ( 0.7,0.3,0.7 );
    dp->ren_grad1->render_triangles();
  }

  if ( dp->m_bShowCancelablePairs&& dp->ren_cancelpairs )
  {
    dp->ren_cancelpairs->render_triangles();
  }

  glDisable ( GL_LIGHTING );
  glPointSize ( 4.0 );

  if ( dp->m_bShowCps)
  {
    for(uint i = 0 ; i < 3;++i)
    {
      if(dp->ren_cp[i])
      {
        glColor3dv(g_cp_colors[i].data());

        dp->ren_cp[i]->render();

        if(dp->ren_cp_labels[i] &&
           m_bShowCriticalPointLabels)
          dp->ren_cp_labels[i]->render();
      }
    }
  }

  glTranslatef ( 0.0,0.0002,0.0 );

  if ( dp->m_bShowMsGraph && dp->ren_msgraph)
  {
    glColor3f ( 0.0,0.5,1.0 );
    dp->ren_cp_conns[0]->render();

    glColor3f ( 1.0,0.5,0.0 );
    dp->ren_cp_conns[1]->render();

  }

  if ( dp->m_bShowMsQuads && dp->ren_msquads)
  {
    dp->ren_msquads->render_quads();
  }

  glPopAttrib();
  glPopMatrix();
}

int QuadDataManager::Render() const
{
  m_controller->Render();

  for ( uint i = 0 ; i < m_pieces.size();i++ )
  {
    renderDataPiece ( m_pieces[i] );
  }
  glPushMatrix();
  glPushAttrib ( GL_ENABLE_BIT );


  glScalef ( 2.0,2.0,2.0 );
  glTranslatef ( -0.5,0.0,-0.5 );

  glDisable ( GL_LIGHTING );

  glPointSize ( 4.0 );

  if ( m_bShowCriticalPoints && ren_crit)
  {
    ren_crit->render_points();

    if ( m_bShowCriticalPointLabels )
      ren_crit->render_text();
  }

  if ( m_bShowMsGraph && ren_msgraph)
  {
    glColor3f ( 0.5,1.0,1.0 );
    ren_msgraph->render_points();
  }

  glColor3f ( 0.0,0.0,0.0 );

  if(m_ren_disc)
    m_ren_disc->render_points();

  glPopAttrib();
  glPopMatrix();


  return 0;
}

bool QuadDataManager::MousePressedEvent
    ( const int &x, const int &y, const eMouseButton &mb,
      const eKeyFlags &,const eMouseFlags &)
{
  switch(mb)
  {
  case MOUSEBUTTON_RIGHT:
    {
      m_controller->StartTB ( x, y );
      return true;
    }
  case MOUSEBUTTON_LEFT:
    {
      m_controller->StartTrans ( x, y );
      return true;
    }
  default:
    return false;
  }
}

bool QuadDataManager::MouseReleasedEvent
    ( const int &x, const int &y, const eMouseButton &mb,
      const eKeyFlags &,const eMouseFlags &)
{
  switch(mb)
  {
  case MOUSEBUTTON_RIGHT:
    {
      m_controller->StopTB ( x, y );
      return true;
    }

  case MOUSEBUTTON_LEFT:
    {
      m_controller->StopTrans( x, y );
      return true;
    }
  default:
    return false;
  }
}

bool QuadDataManager::MouseMovedEvent
    ( const int &x, const int &y, const int &, const int &,
      const eKeyFlags &,const eMouseFlags &mf)
{
  if(mf &MOUSEBUTTON_LEFT || mf &MOUSEBUTTON_RIGHT)
  {
    m_controller->Move ( x, y );
    return true;
  }
  return false;
}

bool QuadDataManager::WheelEvent
    ( const int &, const int &, const int &d,
      const eKeyFlags &kf,const eMouseFlags &)
{

  if(kf&KEYFLAG_CTRL)
  {

    double gf = 1.0;

    if(d>0)
      gf += 0.1;
    else
      gf -= 0.1;

    m_controller->set_uniform_scale(m_controller->get_uniform_scale()*gf);
    return true;
  }

  return false;
}

void  DataPiece::create_cp_rens(double *data,uint size_x,uint size_y)
{
  typedef generic_mscomplex_t::critical_point critical_point_t;
  typedef critical_point_t::connection_t      connection_t;

  std::vector<std::string>            crit_labels[3];
  std::vector<glutils::vertex_t>      crit_label_locations[3];

  std::vector<glutils::vertex_t>      crit_locations;
  std::vector<glutils::point_idx_t>   crit_pt_idxs[3];
  std::vector<glutils::line_idx_t>    crit_conn_idxs[2];
  std::map<uint,uint>                 crit_ms_idx_ren_idx_map;

  for(uint i = 0; i < this->mscomplex->m_cps.size(); ++i)
  {
    if(this->mscomplex->m_cps[i]->isCancelled)
      continue;

    uint dim = this->mscomplex->m_cps[i]->cellid.dim();

    crit_labels[dim].push_back(mscomplex->m_cps[i]->cellid.toString());

    double x,y,z;

    get3DCoordsFromList_ftor ( data,size_x,size_y )
        (mscomplex->m_cps[i]->cellid,x,y,z);

    crit_label_locations[dim].push_back(glutils::vertex_t(x,y,z) );

    crit_locations.push_back(glutils::vertex_t(x,y,z));

    crit_ms_idx_ren_idx_map[i] = crit_locations.size()-1;

    if(this->mscomplex->m_cps[i]->isBoundryCancelable)
      continue;

    crit_pt_idxs[dim].push_back(glutils::point_idx_t(crit_locations.size()-1));
  }

  glutils::bufobj_ptr_t crit_loc_bo = glutils::make_buf_obj(crit_locations);

  for(uint i = 0 ; i < 3; ++i)
  {
    ren_cp_labels[i] =
        glutils::create_buffered_text_ren
        (crit_labels[i],crit_label_locations[i]);

    ren_cp[i] =glutils::create_buffered_points_ren
               (crit_loc_bo,
                glutils::make_buf_obj(crit_pt_idxs[i]),
                glutils::make_buf_obj());
  }

  for(uint i = 0; i < this->mscomplex->m_cps.size(); ++i)
  {
    if(this->mscomplex->m_cps[i]->isCancelled)
      continue;

    uint dim = this->mscomplex->m_cps[i]->cellid.dim();

    if(dim == 2)
      continue;

    uint cp_ren_idx = crit_ms_idx_ren_idx_map[i];

    for(connection_t::iterator it = mscomplex->m_cps[i]->asc.begin();
        it != mscomplex->m_cps[i]->asc.end(); ++it)
    {
      if(this->mscomplex->m_cps[*it]->isCancelled)
        throw std::logic_error("this cancelled cp should not be present here");

      uint conn_cp_ren_idx = crit_ms_idx_ren_idx_map[*it];

      crit_conn_idxs[dim].push_back
          (glutils::line_idx_t(cp_ren_idx,conn_cp_ren_idx));

    }

    if(!mscomplex->m_cps[i]->isBoundryCancelable)
      continue;

    for(connection_t::iterator it = mscomplex->m_cps[i]->des.begin();
        it != mscomplex->m_cps[i]->des.end(); ++it)
    {
      if(mscomplex->m_cps[*it]->isCancelled)
        throw std::logic_error("this cancelled cp should not be present here");

      if(mscomplex->m_cps[*it]->isBoundryCancelable)
        continue;

      uint conn_cp_ren_idx = crit_ms_idx_ren_idx_map[*it];

      crit_conn_idxs[dim-1].push_back
          (glutils::line_idx_t(cp_ren_idx,conn_cp_ren_idx));

    }
  }

  for(uint i = 0 ; i < 2; ++i)
  {
    ren_cp_conns[i] = glutils::create_buffered_lines_ren
                      (crit_loc_bo,
                       glutils::make_buf_obj(crit_conn_idxs[i]),
                       glutils::make_buf_obj());
  }
}

void QuadDataManager::update_disc()
{
  if(m_ren_disc != NULL)
    delete m_ren_disc;

  DataPiece *dp_root = *m_dpTree.begin();

  //  create_critpt_disc_renderer
  //      ( dp_root->mscomplex,
  //        m_ren_disc,m_disc_grad_type ,m_disc_critpt_no,
  //        get3DCoordsFromList_ftor ( m_pData,m_size_x,m_size_y ) );
}
