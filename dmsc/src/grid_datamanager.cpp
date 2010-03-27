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
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <modelcontroller.h>

#include <grid_datamanager.h>
#include <discreteMorseAlgorithm.h>
#include <discreteMorseDSRenderFuncs.h>


using namespace std;

void GridDataManager::createDataPieces ()
{

  rect_t r(cellid_t(0,0),cellid_t(2*(m_size_x-1),2*(m_size_y-1)));
  rect_t e(cellid_t(0,0),cellid_t(2*(m_size_x-1),2*(m_size_y-1)));

  createPieces_quadtree(r,e,m_num_levels);
  return;
}

const unsigned char split_axes = 1;
const unsigned char scan_axes  = (split_axes+1)%2;

void GridDataManager::createPieces_quadtree(rect_t r,rect_t e,u_int level )
{
  if(level == 0)
  {
    GridDataPiece *dp = new GridDataPiece(m_pieces.size());
    dp->dataset = new GridDataset(r,e);
    dp->msgraph = new GridMSComplex(r,e);
    m_pieces.push_back(dp);

    return;
  }

  u_int dim = split_axes;

  rect_size_t  s = r.size();

  rect_point_t tr1 = r.top_right();
  rect_point_t bl1 = r.bottom_left();

  tr1[dim] -= 2*(s[dim]/4);
  bl1[dim]  = tr1[dim];

  rect_t r1 = rect_t(r.bottom_left(),tr1);
  rect_t r2 = rect_t(bl1,r.top_right());

  rect_point_t tr2 = e.top_right();
  rect_point_t bl2 = e.bottom_left();

  tr2[dim] = tr1[dim] + 2;
  bl2[dim] = bl1[dim] - 2;

  rect_t e1 = rect_t(e.bottom_left(),tr2);
  rect_t e2 = rect_t(bl2,e.top_right());

  createPieces_quadtree(r1,e1,level-1);
  createPieces_quadtree(r2,e2,level-1);
}

void read_msgraph_from_archive(GridDataPiece * dp)
{
  std::string filename("dp_msgraph_");
  filename += dp->label();

  std::ifstream ifs(filename.c_str());

  boost::archive::text_iarchive ia(ifs);

  dp->msgraph = new GridDataPiece::mscomplex_t;

  ia >> (*dp->msgraph);
}

void write_msgraph_to_archive(GridDataPiece * dp)
{
  std::string filename("dp_msgraph_");
  filename += dp->label();

  std::ofstream ofs(filename.c_str());

  boost::archive::text_oarchive oa(ofs);

  oa << (*dp->msgraph);

  delete dp->msgraph;
  dp->msgraph = NULL;

}

void GridDataManager::computeMsGraph( GridDataPiece *dp )
{
  if(m_use_ocl != true)
  {
    dp->dataset->assignGradients();

    dp->dataset->computeConnectivity(dp->msgraph);
  }
  else
  {
    dp->dataset->writeout_connectivity_ocl(dp->msgraph);
  }

  if(m_compute_out_of_core)
  {
    write_msgraph_to_archive(dp);

    dp->dataset->clear();
  }
}

void mergePiecesUp_worker
    ( GridDataPiece  *dp,
      GridDataPiece  *dp1,
      GridDataPiece  *dp2,
      bool is_src_archived,
      bool archive_dest)
{

  if(dp1->level != dp2->level)
      throw std::logic_error("dps must have same level");

  if(is_src_archived == true)
  {
    read_msgraph_from_archive(dp1);
    read_msgraph_from_archive(dp2);
  }

  dp->level         = dp1->level+1;
  dp->msgraph       = GridMSComplex::merge_up(*dp1->msgraph,*dp2->msgraph);

  if(is_src_archived == true)
  {
    delete dp1->msgraph;dp1->msgraph = NULL;
    delete dp2->msgraph;dp2->msgraph = NULL;
  }

  if(archive_dest == true)
  {
    write_msgraph_to_archive(dp);
  }
}


void mergePiecesDown_worker
    ( GridDataPiece  * dp,
      GridDataPiece  * dp1,
      GridDataPiece  * dp2,
      bool is_src_archived,
      bool is_dest_archived,
      bool archive_src,
      bool archive_dest)
{

  if(is_src_archived)
    read_msgraph_from_archive(dp);

  if(is_dest_archived)
  {
    read_msgraph_from_archive(dp1);
    read_msgraph_from_archive(dp2);
  }

  dp->msgraph->merge_down(*dp1->msgraph,*dp2->msgraph);

  if(archive_src)
  {
    write_msgraph_to_archive(dp);
  }

  if(is_src_archived && !archive_src)
  {
    delete dp->msgraph;dp->msgraph = NULL;
  }
  if(archive_dest)
  {
    write_msgraph_to_archive(dp1);
    write_msgraph_to_archive(dp2);
  }
}


void GridDataManager::computeMsGraphInRange(uint start,uint end )
{
  _LOG ( "Begin Gradient/ conn work for pieces from "<<start<<" to "<<end );

  for ( uint i = start ; i < end;i++ )
  {
    GridDataPiece * dp = m_pieces[i];

    if(m_use_ocl == true)
      dp->dataset->work_ocl();

    if(m_single_threaded_mode == false)
    {
      _LOG ( "Kicking off thread "<<i-start );


      m_threads[i-start] = new boost::thread
                          ( boost::bind ( &GridDataManager::computeMsGraph,this,dp ) );
    }
    else
    {
      computeMsGraph(dp);
    }

  }
}

void GridDataManager::waitForThreadsInRange(uint start,uint end )
{
  if(m_single_threaded_mode == false)
  {
    _LOG ( "waiting on threads "<<0<<" to "<<end-start );

    for ( uint i = start ; i < end;i++ )
    {
      m_threads[i-start]->join();

      delete m_threads[i-start];

      m_threads[i-start] = NULL;
      _LOG ( "thread "<<i-start<<" joint" );
    }
  }

  _LOG ( "End work for pieces from "<<start<<" to "<<end );
}

void GridDataManager::mergePiecesUp( )
{
  uint num_leafs = pow(2,m_num_levels);

  for(uint i = 0;i<num_leafs-1;++i)
  {
    m_pieces.push_back(new GridDataPiece(m_pieces.size()));
  }

  uint i_incr = std::min((size_t)num_parallel*2,(m_pieces.size()+1)/2);

  for ( uint i = 0; i<m_pieces.size()-1; i+=i_incr)
  {
    uint threadno = 0;

    if(i_incr+i > m_pieces.size() && i_incr > 1)
      i_incr = i_incr>>1;

    bool src_archived = m_compute_out_of_core;
    bool archive_dest = m_compute_out_of_core&&(i_incr != 2);

    for(int j = i ; j < i+i_incr; j +=2 )
    {


      GridDataPiece *dp1 = m_pieces[j];
      GridDataPiece *dp2 = m_pieces[j+1];
      GridDataPiece *dp  = m_pieces[num_leafs+j/2];

      if(m_single_threaded_mode == false)
      {
        _LOG("Kicking off Merge Up "<<j<<" "<<j+1<<"->"<<num_leafs+j/2);

        m_threads[threadno++] = new boost::thread
                              ( boost::bind ( &mergePiecesUp_worker,dp,dp1,dp2,src_archived,archive_dest ));

      }
      else
      {
        _LOG("Kicking off Merge Up "<<j<<" "<<j+1<<"->"<<num_leafs+j/2);

        mergePiecesUp_worker(dp,dp1,dp2,src_archived,archive_dest);
      }
    }

    waitForThreadsInRange(i,i+i_incr/2);
  }

}

void GridDataManager::mergePiecesDown()
{
  uint num_graphs = (0x01<<(m_num_levels+1))-1;

  uint i_incr = 1;

  for ( int i = 1; i < num_graphs/4; )
  {
    uint threadno = 0 ;

    bool src_archived  = m_compute_out_of_core && (i_incr != 1);
    bool dest_archived = m_compute_out_of_core;

    bool archive_src   = m_compute_out_of_core ;
    bool archive_dest  = m_compute_out_of_core ;

    for ( int j = i; j < i+i_incr; j += 1)
    {

      GridDataPiece *dp  = m_pieces[num_graphs- j];
      GridDataPiece *dp1 = m_pieces[num_graphs- 2*j];
      GridDataPiece *dp2 = m_pieces[num_graphs- 2*j-1];

      if(m_single_threaded_mode == false)
      {
        _LOG("Kicking off Merge Down "<< num_graphs - j<<"->"<<
             num_graphs - 2*j<<" "<<num_graphs - 2*j-1);

        m_threads[threadno++] = new boost::thread
                              ( boost::bind ( &mergePiecesDown_worker,dp,dp1,dp2,src_archived,dest_archived,archive_src,archive_dest ));
      }
      else
      {
        _LOG("Merge Down "<< num_graphs - j<<"->"<<
             num_graphs - 2*j<<" "<<num_graphs - 2*j-1);

        mergePiecesDown_worker(dp,dp1,dp2,src_archived,dest_archived,archive_src,archive_dest);
      }
    }

    waitForThreadsInRange(i,i+i_incr);

    i+=i_incr;

    if(i_incr < num_parallel)
      i_incr <<= 1;
  }

  return;
}

void GridDataManager::finalMergeDownPiecesInRange(uint start,uint end)
{

  uint num_subdomains =  ((0x01)<<m_num_levels);

  bool src_archived  = m_compute_out_of_core && (m_num_levels > 1);
  bool dest_archived = m_compute_out_of_core;

  bool archive_src   = false ;
  bool archive_dest  = false ;

  uint threadno = 0;

  for ( int j = start; j < end; j += 2)
  {

    GridDataPiece *dp  = m_pieces[num_subdomains + j/2];
    GridDataPiece *dp1 = m_pieces[j];
    GridDataPiece *dp2 = m_pieces[j+1];

    if(m_single_threaded_mode == false)
    {
      _LOG("Kicking off Merge Down "<< num_subdomains + j/2<<"->"<<
           j+1<<" "<<j);

      m_threads[threadno++] = new boost::thread
                            ( boost::bind ( &mergePiecesDown_worker,dp,dp1,dp2,src_archived,dest_archived,archive_src,archive_dest ));
    }
    else
    {
      _LOG("Merge Down "<< num_subdomains + j/2<<"->"<<
           j+1<<" "<<j);

      mergePiecesDown_worker(dp,dp1,dp2,src_archived,dest_archived,archive_src,archive_dest);
    }
  }
}

void GridDataManager::collectManifold( GridDataPiece  * dp)
{

  if(m_use_ocl == false)
  {
    dp->dataset->assignGradients();
  }

  dp->dataset->postMergeFillDiscs(dp->msgraph);

  if(m_compute_out_of_core)
  {
    dp->dataset->clear();
  }

  std::stringstream ss;

  ss<<"dp_"<<dp->m_pieceno%num_parallel<<"_disc_";

  dp->msgraph->write_discs(ss.str());

  if(m_compute_out_of_core)
  {
    delete dp->msgraph;
  }
}

void GridDataManager::collectManifoldsInRange(uint start,uint end)
{

  uint threadno = 0;

  for ( int j = start; j < end; j += 1)
  {
    GridDataPiece *dp  = m_pieces[j];

    if(m_use_ocl)
      dp->dataset->work_ocl(false);

    if(m_single_threaded_mode == false)
    {
      _LOG("Kicking off collect manifolds "<<j );

      m_threads[threadno++] = new boost::thread
                            ( boost::bind ( &GridDataManager::collectManifold,this,dp ));
    }
    else
    {
      _LOG("Collect manifolds "<<j );

      collectManifold(dp);
    }
  }

}


void GridDataManager::collectSubdomainManifolds( )
{

  std::ifstream data_stream;

  GridDataset::cell_fn_t * data_buffer
      = new GridDataset::cell_fn_t[getMaxDataBufItems()];

  uint num_subdomains =  ((0x01)<<m_num_levels);

  int num_pc_per_buf = std::min(num_parallel,num_subdomains);

  for(uint i = 0 ;i < num_subdomains ; i += num_pc_per_buf)
  {
    // kick off threads to do last merge down
    finalMergeDownPiecesInRange(i,i+num_pc_per_buf);

    // prepare datasets
    readDataAndInit(data_stream,data_buffer,i);

    // wait for merge threads to finish
    waitForThreadsInRange(i,i+num_pc_per_buf/2);

    for(uint j = i; j< i+num_pc_per_buf;++j)
    {
      // collect the manifold info into msgraphs
      collectManifoldsInRange(j,j+1);

      // wait for collection to complete
      waitForThreadsInRange(j,j+1);
    }
  }

  data_stream.close();

  delete data_buffer;
}


void GridDataManager::computeSubdomainMsgraphs ( )
{

  std::ifstream data_stream;

  GridDataset::cell_fn_t * data_buffers[2];

  data_buffers[0] = new GridDataset::cell_fn_t[getMaxDataBufItems()];
  data_buffers[1] = new GridDataset::cell_fn_t[getMaxDataBufItems()];

  uint num_subdomains =  ((0x01)<<m_num_levels);

  int num_pc_per_buf = std::min(num_parallel,num_subdomains);

  for(uint i = 0 ;i < num_subdomains ; i += num_pc_per_buf)
  {

    uint active_buffer = (i/num_pc_per_buf)%2;

    readDataAndInit(data_stream,data_buffers[active_buffer],i);

    if(i != 0 )
    {
      waitForThreadsInRange(i-num_pc_per_buf,i);
    }

    computeMsGraphInRange(i,i+num_pc_per_buf);
  }

  waitForThreadsInRange(num_subdomains-num_pc_per_buf,num_subdomains);

  delete data_buffers[0];
  delete data_buffers[1];

  data_stream.close();
}

void GridDataManager::readDataAndInit
    (std::ifstream &data_stream,
     GridDataset::cell_fn_t *buffer,
     uint start_offset)
{

  uint num_subdomains =  ((0x01)<<m_num_levels);

  int num_pc_per_buf = std::min(num_parallel,num_subdomains);

  GridDataset::rect_size_t domain_sz(m_size_x,m_size_y);

  if(start_offset == 0)
    data_stream.open(m_filename.c_str(),fstream::in|fstream::binary );
  else
    data_stream.seekg(-3*domain_sz[scan_axes]*sizeof ( GridDataset::cell_fn_t),std::ios::cur);

  if(data_stream.is_open() == false)
    throw std::runtime_error("unable to read stream");

  GridDataPiece * dp1 = m_pieces[start_offset];

  GridDataPiece * dp2 = m_pieces[start_offset+num_pc_per_buf-1];

  GridDataset::rect_point_t bl = dp1->dataset->get_ext_rect().bottom_left()/2;
  GridDataset::rect_point_t tr = dp2->dataset->get_ext_rect().top_right()/2;

  size_t num_data_items = (domain_sz[scan_axes]*(tr[split_axes] - bl[split_axes]+1));

  if(num_data_items > getMaxDataBufItems())
    throw std::range_error("insufficient buffer space calculted");

  data_stream.read ( reinterpret_cast<char *> ( buffer),
                sizeof ( GridDataset::cell_fn_t)*num_data_items );


  for(uint j = start_offset ; j < start_offset+num_pc_per_buf;++j)
  {
    GridDataPiece * dp = m_pieces[j];

    GridDataset::rect_point_t dp_bl = dp->dataset->get_ext_rect().bottom_left()/2;

    uint data_offset = domain_sz[scan_axes]*(dp_bl[split_axes] - bl[split_axes]);

    dp->dataset->init(buffer + data_offset);
  }

}

uint GridDataManager::getMaxDataBufItems()
{
  GridDataset::rect_size_t domain_sz(m_size_x,m_size_y);

  uint num_subdomains =  ((0x01)<<m_num_levels);

  int num_pc_per_buf = std::min(num_parallel,num_subdomains);

  uint max_size_split_axis = (domain_sz[split_axes] + num_subdomains-1)/num_subdomains;

  uint max_sz_per_domain = (max_size_split_axis+3)*domain_sz[scan_axes];

  uint buf_ct = max_sz_per_domain*num_pc_per_buf - (num_pc_per_buf-1)*(domain_sz[scan_axes]+1);

  return buf_ct;
}

GridDataManager::GridDataManager
    ( std::string filename,
      u_int        size_x,
      u_int        size_y,
      u_int        num_levels,
      bool         single_threaded_mode,
      bool         use_ocl,
      double       simp_tresh,
      bool         compute_out_of_core):
    m_filename(filename),
    m_size_x(size_x),
    m_size_y(size_y),
    m_num_levels(num_levels),
    m_single_threaded_mode(single_threaded_mode),
    m_bShowCriticalPointLabels(false),
    m_use_ocl(use_ocl),
    m_simp_tresh(simp_tresh),
    m_compute_out_of_core(compute_out_of_core)
{

  m_controller = IModelController::Create();

  createDataPieces();

  if(m_use_ocl)
    GridDataset::init_opencl();

  _LOG ( "==========================" );
  _LOG ( "Starting Processing Peices" );
  _LOG ( "--------------------------" );

  if(m_num_levels == 0 && m_compute_out_of_core)
    m_compute_out_of_core = false;

  computeSubdomainMsgraphs ();

  mergePiecesUp();

  m_pieces[m_pieces.size()-1]->msgraph->simplify_un_simplify(m_simp_tresh);

  mergePiecesDown();

  collectSubdomainManifolds();

  _LOG ( "--------------------------" );
  _LOG ( "Finished Processing peices" );
  _LOG ( "==========================" );

  if ( m_single_threaded_mode == false )
  {
    if(m_use_ocl)
      GridDataset::stop_opencl();

    exit(0);
  }



  for(uint i = 0 ; i <m_pieces.size();++i)
  {
    GridDataPiece *dp = m_pieces[i];

    if(m_compute_out_of_core == true)
      read_msgraph_from_archive(dp);

    dp->create_cp_rens();

    delete dp->msgraph;

    dp->msgraph = NULL;
  }


  std::ifstream data_stream;

  GridDataset::cell_fn_t * data_buffer
      = new GridDataset::cell_fn_t[getMaxDataBufItems()];

  uint num_subdomains =  ((0x01)<<m_num_levels);

  int num_pc_per_buf = std::min(num_parallel,num_subdomains);

  for(uint i = 0 ;i < num_subdomains ; i += num_pc_per_buf)
  {
    // prepare datasets
    readDataAndInit(data_stream,data_buffer,i);

    for(uint j = i; j< i+num_pc_per_buf;++j)
    {
      GridDataPiece *dp = m_pieces[j];

      if(m_use_ocl == true)
        dp->dataset->work_ocl();
      else
        dp->dataset->assignGradients();

      dp->create_grad_rens();

      dp->create_surf_ren();

      dp->dataset->clear();
    }

  }

  data_stream.close();

  delete data_buffer;

  if(m_use_ocl)
    GridDataset::stop_opencl();

  create_ui();
}

void GridDataManager ::logAllConnections(const std::string &prefix)
{
  for(uint i = 0 ; i <m_pieces.size();++i)
  {
    GridDataPiece *dp = m_pieces[i];

    fstream outfile((prefix+dp->label()+string(".txt")).c_str(),ios::out);

    if(outfile.is_open() == false)
      _LOG("failed to open log file");

    print_connections(*((std::ostream *) &outfile),*dp->msgraph);
  }

}

void GridDataManager::logAllCancelPairs(const std::string &prefix)
{
}

GridDataManager::~GridDataManager()
{
  IModelController::Delete(m_controller);

  destroy_ui();
}

glutils::color_t g_grid_cp_colors[] =
{
  glutils::color_t(1.0,0.0,0.0),
  glutils::color_t(0.0,1.0,0.0),
  glutils::color_t(0.0,0.0,1.0),
};


void GridDataManager::renderDataPiece ( GridDataPiece *dp ) const
{
  glPushMatrix();
  glPushAttrib ( GL_ENABLE_BIT );

  glScalef ( 2.0,2.0,2.0 );
  glTranslatef ( -0.5,0.0,-0.5 );

  if ( dp->m_bShowSurface && dp->ren_surf)
  {
    glColor3f ( 0.75,0.75,0.75 );
    dp->ren_surf->render();
  }

  glDisable ( GL_LIGHTING );

  glTranslatef ( 0.0,0.02,0.0 );

  if ( dp->m_bShowGrad && dp->ren_grad)
  {
    glColor3f ( 0.5,0.0,0.5 );
    dp->ren_grad->render();

  }

  glPointSize ( 4.0 );

  if ( dp->m_bShowCps)
  {
    for(uint i = 0 ; i < 3;++i)
    {
      if(dp->ren_cp[i])
      {
        glColor3dv(g_grid_cp_colors[i].data());

        dp->ren_cp[i]->render();

        if(dp->ren_cp_labels[i] &&
           m_bShowCriticalPointLabels)
          dp->ren_cp_labels[i]->render();
      }
    }
  }

  if ( dp->m_bShowCancCps)
  {
    for(uint i = 0 ; i < 3;++i)
    {
      if(dp->ren_canc_cp[i])
      {
        glColor3dv(g_grid_cp_colors[i].data());

        dp->ren_canc_cp[i]->render();

        if(dp->ren_canc_cp_labels[i] &&
           m_bShowCriticalPointLabels)
          dp->ren_canc_cp_labels[i]->render();
      }
    }
  }

  if ( dp->m_bShowMsGraph && dp->ren_cp_conns[0] && dp->ren_cp_conns[1])
  {
    glColor3f ( 0.0,0.5,1.0 );
    dp->ren_cp_conns[0]->render();

    glColor3f ( 1.0,0.5,0.0 );
    dp->ren_cp_conns[1]->render();

  }

  if ( dp->m_bShowCancMsGraph&& dp->ren_canc_cp_conns[0] && dp->ren_canc_cp_conns[1])
  {
    glColor3f ( 0.0,0.5,1.0 );
    dp->ren_canc_cp_conns[0]->render();

    glColor3f ( 1.0,0.5,0.0 );
    dp->ren_canc_cp_conns[1]->render();

  }

  glPopAttrib();
  glPopMatrix();
}

int GridDataManager::Render() const
{
  m_controller->Render();

  glPushAttrib(GL_ENABLE_BIT);

  glEnable(GL_NORMALIZE);

  glTranslatef(-1,0,-1);

  glScalef(0.5/(double) m_size_x,0.1,0.5/(double) m_size_y);

  for ( uint i = 0 ; i < m_pieces.size();i++ )
  {
    renderDataPiece ( m_pieces[i] );
  }

  glPopAttrib();

  return 0;
}

bool GridDataManager::MousePressedEvent
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

bool GridDataManager::MouseReleasedEvent
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

bool GridDataManager::MouseMovedEvent
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

bool GridDataManager::WheelEvent
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

GridDataPiece::GridDataPiece (uint pno):
    dataset(NULL),
    msgraph(NULL),
    level(0),
    m_bShowSurface ( false ),
    m_bShowCps ( false ),
    m_bShowMsGraph ( false ),
    m_bShowGrad ( false ),
    m_bShowCancCps(false),
    m_bShowCancMsGraph(false),
    ren_surf(NULL),
    ren_grad(NULL),
    m_pieceno(pno)
{
  memset(ren_cp,0,sizeof(ren_cp));
  memset(ren_cp_labels,0,sizeof(ren_cp_labels));
  memset(ren_cp_conns,0,sizeof(ren_cp_conns));
  memset(ren_canc_cp,0,sizeof(ren_canc_cp));
  memset(ren_canc_cp_labels,0,sizeof(ren_canc_cp_labels));
  memset(ren_canc_cp_conns,0,sizeof(ren_canc_cp_conns));
}

void  GridDataPiece::create_cp_rens()
{
  if(msgraph == NULL)
    return;

  std::vector<std::string>            crit_labels[3];
  std::vector<glutils::vertex_t>      crit_label_locations[3];
  std::vector<glutils::point_idx_t>   crit_pt_idxs[3];
  std::vector<glutils::line_idx_t>    crit_conn_idxs[2];


  std::vector<std::string>            crit_canc_labels[3];
  std::vector<glutils::vertex_t>      crit_canc_label_locations[3];
  std::vector<glutils::point_idx_t>   crit_canc_pt_idxs[3];
  std::vector<glutils::line_idx_t>    crit_canc_conn_idxs[2];

  std::vector<glutils::vertex_t>      crit_locations;
  std::map<uint,uint>                 crit_ms_idx_ren_idx_map;

  for(uint i = 0; i < msgraph->m_cps.size(); ++i)
  {
    if(msgraph->m_cps[i]->isCancelled)
      continue;

    cellid_t c = (msgraph->m_cps[i]->cellid);

    uint dim = GridDataset::s_getCellDim(c);

    std::stringstream ss;

    ((ostream&)ss)<<c;

    double x = c[0],y = msgraph->m_cp_fns[i],z=c[1];

    if(msgraph->m_cps[i]->isBoundryCancelable)
    {
      crit_canc_labels[dim].push_back(ss.str());
      crit_canc_label_locations[dim].push_back(glutils::vertex_t(x,y,z) );
      crit_canc_pt_idxs[dim].push_back(glutils::point_idx_t(crit_locations.size()));
    }
    else
    {
      crit_labels[dim].push_back(ss.str());
      crit_label_locations[dim].push_back(glutils::vertex_t(x,y,z) );
      crit_pt_idxs[dim].push_back(glutils::point_idx_t(crit_locations.size()));
    }

    crit_ms_idx_ren_idx_map[i] = crit_locations.size();
    crit_locations.push_back(glutils::vertex_t(x,y,z));
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

    ren_canc_cp_labels[i] =
        glutils::create_buffered_text_ren
        (crit_canc_labels[i],crit_canc_label_locations[i]);

    ren_canc_cp[i] =glutils::create_buffered_points_ren
               (crit_loc_bo,
                glutils::make_buf_obj(crit_canc_pt_idxs[i]),
                glutils::make_buf_obj());
  }

  for(uint i = 0 ; i < msgraph->m_cps.size(); ++i)
  {
    if(msgraph->m_cps[i]->isCancelled)
      continue;

    conn_t *cp_acdc[] = {&msgraph->m_cps[i]->des,&msgraph->m_cps[i]->asc};

    uint acdc_ct = 1;

    if(msgraph->m_cps[i]->isBoundryCancelable)
      acdc_ct = 2;

    uint cp_ren_idx = crit_ms_idx_ren_idx_map[i];

    uint dim = GridDataset::s_getCellDim
               (msgraph->m_cps[i]->cellid);

    for (uint j = 0 ; j < acdc_ct; ++j)
    {
      for(conn_t::iterator it = cp_acdc[j]->begin();
      it != cp_acdc[j]->end(); ++it)
      {
        if(msgraph->m_cps[*it]->isCancelled)
          throw std::logic_error("this cancelled cp should not be present here");

        if(msgraph->m_cps[*it]->isBoundryCancelable)
          throw std::logic_error("a true cp should not be connected to a bc cp");

        uint conn_cp_ren_idx = crit_ms_idx_ren_idx_map[*it];

        if(msgraph->m_cps[i]->isBoundryCancelable)
        {
          crit_canc_conn_idxs[dim-1+j].push_back
              (glutils::line_idx_t(cp_ren_idx,conn_cp_ren_idx));
        }
        else
        {
          crit_conn_idxs[dim-1+j].push_back
              (glutils::line_idx_t(cp_ren_idx,conn_cp_ren_idx));
        }
      }
    }
  }

  for(uint i = 0 ; i < 2; ++i)
  {
    ren_cp_conns[i] = glutils::create_buffered_lines_ren
                      (crit_loc_bo,
                       glutils::make_buf_obj(crit_conn_idxs[i]),
                       glutils::make_buf_obj());

    ren_canc_cp_conns[i] = glutils::create_buffered_lines_ren
                      (crit_loc_bo,
                       glutils::make_buf_obj(crit_canc_conn_idxs[i]),
                       glutils::make_buf_obj());
  }

}

void GridDataPiece::create_grad_rens()
{
  if(dataset == NULL)
    return;

  rect_t r = dataset->get_ext_rect();

  std::vector<glutils::vertex_t>      cell_locations;
  std::vector<glutils::line_idx_t>    pair_idxs;


  for(cell_coord_t y = r.bottom(); y<=r.top(); ++y)
    for(cell_coord_t x = r.left(); x<=r.right(); ++x)
    {
    cellid_t c = cellid_t(x,y);
    if(dataset->isCellPaired(c))
    {
      cellid_t p = dataset->getCellPairId(c);

      if(dataset->isPairOrientationCorrect(c,p))
      {
        double x,y,z;

        dataset->getCellCoord(c,x,y,z);
        cell_locations.push_back(glutils::vertex_t(x,y,z) );

        dataset->getCellCoord(p,x,y,z);
        cell_locations.push_back(glutils::vertex_t(x,y,z) );

        pair_idxs.push_back
            (glutils::line_idx_t(cell_locations.size()-2,
                                 cell_locations.size()-1));

      }
    }
  }

  ren_grad = glutils::create_buffered_lines_ren
             (glutils::make_buf_obj(cell_locations),
              glutils::make_buf_obj(pair_idxs),
              glutils::make_buf_obj());


}

void GridDataPiece::create_surf_ren()
{
  if(dataset == NULL)
    return;

  rect_t r = dataset->get_ext_rect();

  std::vector<glutils::vertex_t>      point_locs;
  std::vector<glutils::tri_idx_t>     tri_idxs;


  for(cell_coord_t y = r.bottom(); y<=r.top(); y+=2)
    for(cell_coord_t x = r.left(); x<=r.right(); x+=2)
    {
    cellid_t c = cellid_t(x,y);

    double x,y,z;

    dataset->getCellCoord(c,x,y,z);

    point_locs.push_back(glutils::vertex_t(x,y,z));
  }

  rect_size_t s = r.size();

  glutils::idx_t p1=0,p2=1,p3=s[0]/2+2,p4 = s[0]/2+1;

  for(cell_coord_t y = r.bottom()+1; y<r.top(); y+=2)
  {
    for(cell_coord_t x = r.left()+1; x<r.right(); x+=2)
    {
      tri_idxs.push_back(glutils::tri_idx_t(p1,p4,p3));
      tri_idxs.push_back(glutils::tri_idx_t(p1,p3,p2));

      ++p1;++p2;++p3;++p4;
    }
    ++p1;++p2;++p3;++p4;
  }

  ren_surf = glutils::create_buffered_triangles_ren
             (glutils::make_buf_obj(point_locs),
              glutils::make_buf_obj(tri_idxs),
              glutils::make_buf_obj());
}

std::string GridDataPiece::label()
{
  std::stringstream ss;
  ss<<m_pieceno;

  return ss.str();
}
