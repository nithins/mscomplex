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
    dp->dataset->init();
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

void GridDataManager::readFile ( )
{

  u_int dp_idx = 0;

  fstream infile ( m_filename.c_str(),fstream::in|fstream::binary );

  for ( uint y = 0 ; y <m_size_y;y++ )
    for ( uint x = 0 ; x < m_size_x;x++ )
    {
    double data;

    if ( infile.eof() )
      throw std::length_error(string(" Premature end of file "));

    infile.read ( reinterpret_cast<char *> ( &data),sizeof ( double ) );

    cellid_t p(2*x,2*y);

    if(!m_pieces[dp_idx]->dataset->get_ext_rect().contains(p))
      ++dp_idx;

    m_pieces[dp_idx]->dataset->set_cell_fn(p,data);

    if(dp_idx+1 < m_pieces.size() &&
       m_pieces[dp_idx+1]->dataset->get_ext_rect().contains(p))
      m_pieces[dp_idx+1]->dataset->set_cell_fn(p,data);
  }
}

void GridDataManager::initDataPieceForWork ( GridDataPiece *dp )
{

}

void GridDataManager::initDataPieceForRender ( GridDataPiece *dp )
{

}

void GridDataManager::workPiece ( GridDataPiece *dp )
{
  initDataPieceForWork ( dp );

  dp->dataset->assignGradients();
  dp->dataset->computeConnectivity(dp->msgraph);
}

GridDataPiece  * GridDataManager::mergePiecesUp
    ( GridDataPiece  * dp1,GridDataPiece  *dp2)
{
  if(dp1->level != dp2->level)
    throw std::logic_error("dps must have same level");

  GridDataPiece *p = new GridDataPiece();

  p->level = dp1->level+1;
  p->msgraph = GridMSComplex::merge_up(*dp1->msgraph,*dp2->msgraph);
  return p;
}

void GridDataManager::mergePiecesDown ( GridDataPiece  * dp1,GridDataPiece  *dp2)
{

}

void GridDataManager::workAllPieces_mt( )
{
  _LOG ( "Begin calculating asc/des manifolds for all pieces " );

  boost::thread ** threads = new boost::thread*[ m_pieces.size() ];

  uint threadno = 0;

  for ( uint i = 0 ; i < m_pieces.size();i++ )
  {
    _LOG ( "Kicking off thread "<<threadno );

    GridDataPiece * dp = m_pieces[i];
    threads[threadno] = new boost::thread
                        ( boost::bind ( &GridDataManager::workPiece,this,dp ) );
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

void GridDataManager::workAllPieces_st( )
{
  _LOG ( "Begin calculating asc/des manifolds for all pieces " );

  for ( uint i = 0 ; i < m_pieces.size();i++ )
  {
    GridDataPiece * dp = m_pieces[i];

    workPiece(dp);
  }

  _LOG ( "End calculating asc/des manifolds for all pieces " );
}

void GridDataManager::mergePiecesUp_mt( )
{

  _LOG("Begin Merge");

  for ( uint i = 1 ; i < m_pieces.size(); i*= 2)
  {
    boost::thread ** threads = new boost::thread*[ m_pieces.size()/(i*2) ];

    uint threadno = 0;

    for ( uint j = 0 ; j < m_pieces.size(); j+= i*2)
    {
      _LOG("Kicking off merging Up "<<j<<" "<<j+i);

      GridDataPiece * dp1 = m_pieces[j];
      GridDataPiece * dp2 = m_pieces[j+i];

      threads[threadno] = new boost::thread
                          ( boost::bind ( &GridDataManager::mergePiecesUp,this,dp1,dp2 ));
      threadno++;

    }

    threadno = 0;

    for ( uint j = 0 ; j < m_pieces.size(); j+= i*2)
    {
      threads[threadno]->join();
      _LOG ( "thread "<<threadno<<" joint" );
      threadno++;
    }
    delete []threads;
  }

  _LOG("End Merge");
}

void GridDataManager::mergePiecesUp_st( )
{

  for ( uint j = 0 ; j < m_pieces.size()/2; ++j)
  {
    GridDataPiece * dp1 = m_pieces[2*j];
    GridDataPiece * dp2 = m_pieces[2*j+1];

    _LOG("Merging Up "<<2*j<<" "<<2*j+1<<"->"<<m_pieces.size());

    m_pieces.push_back(GridDataManager::mergePiecesUp(dp1,dp2));
  }

}

void GridDataManager::mergePiecesDown_st()
{
  for ( uint i = m_pieces.size()/2 ; i >= 1 ; i/= 2)
  {
    for ( int j = m_pieces.size()-i ; j >=0 ; j-= i*2)
    {
      GridDataPiece * dp1 = m_pieces[j];
      GridDataPiece * dp2 = m_pieces[j+i];

      _LOG("merging Down "<<j-i<<" "<<j);

      GridDataManager::mergePiecesDown(dp1,dp2);
    }
  }
}

GridDataManager::GridDataManager
    ( std::string filename,
      u_int        size_x,
      u_int        size_y,
      u_int        num_levels,
      bool         single_threaded_mode):
    m_filename(filename),
    m_size_x(size_x),
    m_size_y(size_y),
    m_num_levels(num_levels),
    m_single_threaded_mode(single_threaded_mode),
    m_bShowCriticalPointLabels(false)
{

  m_controller = IModelController::Create();

  createDataPieces();

  readFile ();

  _LOG ( "==========================" );
  _LOG ( "Starting Processing Peices" );
  _LOG ( "--------------------------" );

  //  try
  {
    if ( m_single_threaded_mode == false )
    {
      workAllPieces_mt();

      mergePiecesUp_mt();

      exit(0);
    }
    else
    {
      workAllPieces_st();

      mergePiecesUp_st();

      mergePiecesDown_st();
    }
  }

  //  catch(std::exception &e)
  //  {
  //    for(uint i = 0 ; i <m_pieces.size();++i)
  //    {
  //      std::stringstream ss;
  //      ss<<"log/dp-"<<i;
  //
  //      logAllConnections(ss.str());
  //
  //      throw e;
  //    }
  //  }

  _LOG ( "--------------------------" );
  _LOG ( "Finished Processing peices" );
  _LOG ( "==========================" );

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

  if ( dp->m_bShowGrad && dp->ren_grad)
  {
    glColor3f ( 0.7,0.7,0.2 );
    dp->ren_grad->render();

  }

  glDisable ( GL_LIGHTING );
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

  glTranslatef ( 0.0,0.0002,0.0 );

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
    if(msgraph->m_cps[i]->isCancelled)
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

    if(!msgraph->m_cps[i]->isBoundryCancelable)
      continue;

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

      if(dataset->isPairOrientationCorrect(p,c))
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

  rect_t r = dataset->get_rect();

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
