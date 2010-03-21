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


void GridDataManager::createPieces_quadtree(rect_t r,rect_t e,u_int level )
{
  if(level == 0)
  {
    stringstream ss;
    ss<<r;

    GridDataPiece *dp = new GridDataPiece();
    dp->dataset = new GridDataset(r,e);
    dp->msgraph = new GridMSComplex(r,e);
    m_pieces.push_back(dp);

    return;
  }

  u_int dim = 1;

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

const uint num_parallel = 8;

void GridDataManager::readFile ( )
{

  u_int dp_idx = 0;

  fstream infile ( m_filename.c_str(),fstream::in|fstream::binary );

  m_pieces[dp_idx]->dataset->init();

  if(dp_idx+1 < m_pieces.size())
    m_pieces[dp_idx+1]->dataset->init();

  for ( uint y = 0 ; y <m_size_y;y++ )
    for ( uint x = 0 ; x < m_size_x;x++ )
    {
    double data;

    if ( infile.eof() )
      throw std::length_error(string(" Premature end of file "));

    infile.read ( reinterpret_cast<char *> ( &data),sizeof ( double ) );

    cellid_t p(2*x,2*y);

    if(!m_pieces[dp_idx]->dataset->get_ext_rect().contains(p))
    {
      ++dp_idx;
      if(dp_idx < m_pieces.size()-1)
        m_pieces[dp_idx+1]->dataset->init();

      if(m_single_threaded_mode == false )
      {
        if(dp_idx%num_parallel == 0)
        {
          uint start = dp_idx - num_parallel;
          uint end   = dp_idx;

          workPiecesInRange_mt(start,end);

          clearInteriorGrad(start,end);
        }
      }
    }

    m_pieces[dp_idx]->dataset->set_cell_fn(p,data);

    if(dp_idx+1 < m_pieces.size() &&
       m_pieces[dp_idx+1]->dataset->get_ext_rect().contains(p))
      m_pieces[dp_idx+1]->dataset->set_cell_fn(p,data);
  }

  if(m_single_threaded_mode == false)
  {
    uint start = m_pieces.size() - m_pieces.size()%num_parallel ;
    uint end   = m_pieces.size();

    if(start == end)
    {
      start -= num_parallel;
    }

    workPiecesInRange_mt(start,end);
    clearInteriorGrad(start,end);

  }
}

void GridDataManager::clearInteriorGrad(uint start,uint end )
{
  for(uint i = start ; i < end; ++i)
  {
    GridDataPiece *dp = m_pieces[i];
    dp->dataset->clear_graddata();
    delete dp->dataset;
  }
}

void GridDataManager::workPiece ( GridDataPiece *dp )
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
}

void mergePiecesUp
    ( GridDataPiece  *dp,
      GridDataPiece  *dp1,
      GridDataPiece  *dp2)
{

  if(dp1->level != dp2->level)
      throw std::logic_error("dps must have same level");

  dp->level         = dp1->level+1;
  dp->msgraph       = GridMSComplex::merge_up(*dp1->msgraph,*dp2->msgraph);
}

void mergePiecesDown
    ( GridDataPiece  * dp,
      GridDataPiece  * dp1,
      GridDataPiece  * dp2)
{
  dp->msgraph->merge_down(*dp1->msgraph,*dp2->msgraph);
}

void GridDataManager::workPiecesInRange_ocl(uint start,uint end )
{
  _LOG ( "Begin ocl work  for pieces from "<<start<<" to "<<end );
  for ( uint i = start ; i < end;i++ )
  {
    GridDataPiece * dp = m_pieces[i];

    dp->dataset->work_ocl();
  }
  _LOG ( "End ocl work  for pieces from "<<start<<" to "<<end );
}

void GridDataManager::workPiecesInRange_mt(uint start,uint end )
{
  if(m_use_ocl == true)
    workPiecesInRange_ocl(start,end);

  _LOG ( "Begin mt work for pieces from "<<start<<" to "<<end );

  boost::thread ** threads = new boost::thread*[ end-start ];

  uint threadno = 0;

  for ( uint i = start ; i < end;i++ )
  {
    _LOG ( "Kicking off thread "<<threadno );

    GridDataPiece * dp = m_pieces[i];
    threads[threadno] = new boost::thread
                        ( boost::bind ( &GridDataManager::workPiece,this,dp ) );
    threadno++;
  }

  threadno = 0;

  for ( uint i = start ; i < end;i++ )
  {
    threads[threadno]->join();
    _LOG ( "thread "<<threadno<<" joint" );
    threadno++;
  }

  delete []threads;

  _LOG ( "End mt work for pieces from "<<start<<" to "<<end );
}

void GridDataManager::workAllPieces_mt( )
{
  workPiecesInRange_mt(0,m_pieces.size());
}

void GridDataManager::workAllPieces_st( )
{

  _LOG ( "Begin st work on pieces" );

  if(m_use_ocl == true)
    workPiecesInRange_ocl(0,m_pieces.size());

  for ( uint i = 0 ; i < m_pieces.size();i++ )
  {
    GridDataPiece * dp = m_pieces[i];

    workPiece(dp);
  }

  _LOG ( "End st work on pieces" );
}

void GridDataManager::mergePiecesUp_mt( )
{
  uint num_leafs = pow(2,m_num_levels);

  for(uint i = 0;i<num_leafs-1;++i)
  {
    m_pieces.push_back(new GridDataPiece);
  }

  uint p_start = 0,p_end = num_leafs;

  for(uint i = m_num_levels ; i >= 1;--i)
  {
    boost::thread ** threads = new boost::thread*[ (p_end -p_start)/2];

    uint threadno = 0;

    for ( uint j = p_start ; j < p_end; j+=2)
    {
      _LOG("Kicking off Merge Up "<<j<<" "<<j+1<<"->"<<num_leafs+j/2);

      GridDataPiece *dp1 = m_pieces[j];
      GridDataPiece *dp2 = m_pieces[j+1];
      GridDataPiece *dp  = m_pieces[num_leafs+j/2];
      threads[threadno++] = new boost::thread
                            ( boost::bind ( &mergePiecesUp,dp,dp1,dp2 ));
    }

    threadno = 0;

    for ( uint j = p_start ; j < p_end; j+=2)
    {
      threads[threadno]->join();
      _LOG ( "thread "<<threadno++<<" joint" );

    }
    p_start = p_end;
    p_end  += 0x01<<i-1;
  }
}

void GridDataManager::mergePiecesUp_st( )
{
  // double the no of pieces

  uint num_leafs = pow(2,m_num_levels);

  for(uint i = 0;i<num_leafs-1;++i)
  {
    m_pieces.push_back(new GridDataPiece);
  }

  for ( uint i = 0;i<m_pieces.size()-1; i+=2)
  {
    GridDataPiece *dp1 = m_pieces[i];
    GridDataPiece *dp2 = m_pieces[i+1];
    GridDataPiece *dp  = m_pieces[num_leafs+i/2];

    _LOG(" Merge Up "<<i<<" "<<i+1<<"->"<<num_leafs+i/2);

    mergePiecesUp(dp,dp1,dp2);
  }

}

void GridDataManager::mergePiecesDown_st()
{
  uint num_nodes = (0x01<<(m_num_levels+1))-1;

  for(uint i = 0 ; i < m_num_levels;++i)
  {
    uint p_start = 0x01<<i,p_end = p_start<<1;

    for ( uint j = p_start ; j < p_end; j++)
    {
      _LOG("Merge Down "<<
           num_nodes - j<<"->"<<
           num_nodes - 2*j<<" "<<
           num_nodes - 2*j-1);

      GridDataPiece *dp  = m_pieces[num_nodes- j];
      GridDataPiece *dp1 = m_pieces[num_nodes- 2*j];
      GridDataPiece *dp2 = m_pieces[num_nodes- 2*j-1];

      mergePiecesDown(dp,dp1,dp2);
    }
  }
}

void GridDataManager::mergePiecesDown_mt()
{
  uint num_nodes = (0x01<<(m_num_levels+1))-1;

  for(uint i = 0 ; i < m_num_levels;++i)
  {
    uint p_start = 0x01<<i,p_end = p_start<<1;

    boost::thread ** threads = new boost::thread*[p_end -p_start];

    uint threadno = 0;

    for ( uint j = p_start ; j < p_end; j++)
    {
      _LOG("Kicking off Merge Down "<<
           num_nodes - j<<"->"<<
           num_nodes - 2*j<<" "<<
           num_nodes - 2*j-1);

      GridDataPiece *dp  = m_pieces[num_nodes- j];
      GridDataPiece *dp1 = m_pieces[num_nodes- 2*j];
      GridDataPiece *dp2 = m_pieces[num_nodes- 2*j-1];

      threads[threadno++] = new boost::thread
                            ( boost::bind ( &mergePiecesDown,dp,dp1,dp2 ));
    }

    threadno = 0;

    for ( uint j = p_start ; j < p_end; j++)
    {
      threads[threadno]->join();
      _LOG ( "thread "<<threadno++<<" joint" );
    }
  }
}

GridDataManager::GridDataManager
    ( std::string filename,
      u_int        size_x,
      u_int        size_y,
      u_int        num_levels,
      bool         single_threaded_mode,
      bool         use_ocl,
      u_int        num_canc):
    m_filename(filename),
    m_size_x(size_x),
    m_size_y(size_y),
    m_num_levels(num_levels),
    m_single_threaded_mode(single_threaded_mode),
    m_bShowCriticalPointLabels(false),
    m_use_ocl(use_ocl),
    m_num_canc(num_canc)
{

  m_controller = IModelController::Create();

  createDataPieces();

  if(m_use_ocl)
    GridDataset::init_opencl();

  _LOG ( "==========================" );
  _LOG ( "Starting Processing Peices" );
  _LOG ( "--------------------------" );

  readFile ();

  {
    if(m_single_threaded_mode == true )
    {
      workAllPieces_st();
    }

    if ( m_single_threaded_mode == false )
    {
      mergePiecesUp_mt();

      m_pieces[m_pieces.size()-1]->msgraph->simplify_un_simplify(num_canc);

      mergePiecesDown_mt();
    }
    else
    {

      mergePiecesUp_st();

      m_pieces[m_pieces.size()-1]->msgraph->simplify_un_simplify(num_canc);

      mergePiecesDown_st();

    }

  }
  _LOG ( "--------------------------" );
  _LOG ( "Finished Processing peices" );
  _LOG ( "==========================" );

  if(m_use_ocl)
    GridDataset::stop_opencl();

  logAllConnections("log/dp-");

  if ( m_single_threaded_mode == false )
    exit(0);

  for(uint i = 0 ; i <m_pieces.size();++i)
  {
    GridDataPiece *dp = m_pieces[i];

    dp->create_cp_rens();

    dp->create_grad_rens();

    dp->create_surf_ren();
  }

  create_ui();
}

void GridDataManager ::logAllConnections(const std::string &prefix)
{
  for(uint i = 0 ; i <m_pieces.size();++i)
  {
    GridDataPiece *dp = m_pieces[i];

    fstream outfile((prefix+dp->label()+string(".txt")).c_str(),ios::out);
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

  if ( dp->m_bShowMsGraph && dp->ren_cp_conns[0] && dp->ren_cp_conns[1])
  {
    glColor3f ( 0.0,0.5,1.0 );
    dp->ren_cp_conns[0]->render();

    glColor3f ( 1.0,0.5,0.0 );
    dp->ren_cp_conns[1]->render();

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

GridDataPiece::GridDataPiece ():
    dataset(NULL),
    msgraph(NULL),
    level(0),
    m_bShowSurface ( false ),
    m_bShowCps ( false ),
    m_bShowMsGraph ( false ),
    m_bShowGrad ( false ),
    ren_surf(NULL),
    ren_grad(NULL)
{
  memset(ren_cp,0,sizeof(ren_cp));
  memset(ren_cp,0,sizeof(ren_cp_labels));
  memset(ren_cp,0,sizeof(ren_cp_conns));
}

void  GridDataPiece::create_cp_rens()
{
  if(msgraph == NULL)
    return;

  std::vector<std::string>            crit_labels[3];
  std::vector<glutils::vertex_t>      crit_label_locations[3];

  std::vector<glutils::vertex_t>      crit_locations;
  std::vector<glutils::point_idx_t>   crit_pt_idxs[3];
  std::vector<glutils::line_idx_t>    crit_conn_idxs[2];
  std::map<uint,uint>                 crit_ms_idx_ren_idx_map;

  for(uint i = 0; i < msgraph->m_cps.size(); ++i)
  {
    if(msgraph->m_cps[i]->isCancelled)
      continue;

    cellid_t c = (msgraph->m_cps[i]->cellid);

    uint dim = GridDataset::s_getCellDim(c);

    std::stringstream ss;

    ((ostream&)ss)<<c;

    crit_labels[dim].push_back(ss.str());

    double x = c[0],y = msgraph->m_cp_fns[i],z=c[1];

    crit_label_locations[dim].push_back(glutils::vertex_t(x,y,z) );

    crit_pt_idxs[dim].push_back(glutils::point_idx_t(crit_locations.size()));

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
  }

  for(uint i = 0; i < msgraph->m_cps.size(); ++i)
  {
    if(msgraph->m_cps[i]->isCancelled )
      continue;

    uint dim = GridDataset::s_getCellDim
               (msgraph->m_cps[i]->cellid);

    if(dim == 2)
      continue;

    uint cp_ren_idx = crit_ms_idx_ren_idx_map[i];

    for(conn_t::iterator it = msgraph->m_cps[i]->asc.begin();
    it != msgraph->m_cps[i]->asc.end(); ++it)
    {
      if(msgraph->m_cps[*it]->isCancelled)
        throw std::logic_error("this cancelled cp should not be present here");

      uint conn_cp_ren_idx = crit_ms_idx_ren_idx_map[*it];

      crit_conn_idxs[dim].push_back
          (glutils::line_idx_t(cp_ren_idx,conn_cp_ren_idx));

    }
  }

  for(uint i = 0; i < msgraph->m_cps.size(); ++i)
  {
    if(msgraph->m_cps[i]->isCancelled )
      continue;

    if(!msgraph->m_cps[i]->isBoundryCancelable)
      continue;

    uint dim = GridDataset::s_getCellDim
               (msgraph->m_cps[i]->cellid);

    if (dim ==0 )
      continue;

    dim--;

    uint cp_ren_idx = crit_ms_idx_ren_idx_map[i];

    for(conn_t::iterator it = msgraph->m_cps[i]->des.begin();
    it != msgraph->m_cps[i]->des.end(); ++it)
    {
      if(msgraph->m_cps[*it]->isCancelled)
        throw std::logic_error("this cancelled cp should not be present here");

      if(msgraph->m_cps[*it]->isBoundryCancelable)
        continue;

      uint conn_cp_ren_idx = crit_ms_idx_ren_idx_map[*it];

      crit_conn_idxs[dim].push_back
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
  ss<<level<<" "<<msgraph->m_rect;

  return ss.str();
}
