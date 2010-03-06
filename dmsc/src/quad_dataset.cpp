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

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <list>

#include <boost/bind.hpp>

#include <cpputils.h>
#include <logutil.h>
#include <quad_dataset.h>

#include <discreteMorseDSRenderFuncs.h>
#include <discreteMorseAlgorithm.h>


typedef QuadGenericCell2D<uint>   generic_cell_t;

generic_cell_t getUniCellID ( const QuadDataset *ds, uint cellid )
{
  uint points[20];

  uint point_ct;

  point_ct = ds->getCellPoints ( cellid,points );

  std::transform ( points,points+point_ct,points,boost::bind ( &QuadDataset::getPointGenericIndex,ds,_1 ) );

  return createQuadGenericCell2D ( points,point_ct,ds->getCellDim ( cellid ) );
}


using namespace std;

// functions to setup the quad edge structure
void QuadDataset::setNumVerts ( const uint & vert_ct )
{
  m_q.setNumVerts ( vert_ct );
}

void QuadDataset::setNumQuads ( const uint &quad_ct )
{
  m_q.setNumQuads ( quad_ct );
}

void QuadDataset::setNumExtVert ( const uint &ct )
{
  m_intbnd_vert_ct  = m_q.m_vert_ct - ct;
  m_vert_cells      = new cell[m_q.m_vert_ct];
  m_orig_vert_index = new uint[m_q.m_vert_ct];

  for ( uint i = 0 ; i < m_q.m_vert_ct;i++ )
  {
    m_vert_cells[i].dim = 0;
  }

  m_vertices = new _vertex[m_q.m_vert_ct];
}

void QuadDataset::setNumExtQuad ( const uint &ct )
{
  m_intbnd_quad_ct  = m_q.m_quad_ct - ct;
  m_quad_cells      = new cell[m_q.m_quad_ct];
  m_orig_quad_index = new uint[m_q.m_quad_ct];

  for ( uint i = 0 ; i < m_q.m_quad_ct;i++ )
  {
    m_quad_cells[i].dim = 2;
  }
}


void QuadDataset::setVert ( const uint &vno,const double &fn,const uint& ind,const double& x,const double& y,const double& z )
{
  m_vertices       [ vno ].x  = x;
  m_vertices       [ vno ].y  = y;
  m_vertices       [ vno ].z  = z;
  m_vertices       [ vno ].fn = fn;
  m_orig_vert_index[ vno ]    = ind;
}

void QuadDataset::startAddingQuads()
{
  m_q.start_adding_quads();
}
void QuadDataset::addQuad ( const uint *v ,const uint &qno,const uint &ind )
{
  m_q.add_quad ( v );

  m_orig_quad_index[qno] = ind;
}

void QuadDataset::endAddingQuads()
{
  m_q.end_adding_quads();

  m_edge_cells = new cell[m_q.m_edge_ct];

  for ( uint i = 0 ; i < m_q.m_edge_ct;i++ )
  {
    m_edge_cells[i].dim = 1;
  }

  uint num_ext_edges = 0;

  num_ext_edges = std::count_if ( uint_iterator ( m_q.m_vert_ct ),uint_iterator ( m_q.m_vert_ct+ m_q.m_edge_ct ),
                                  boost::bind ( &QuadDataset::isCellExterior,this,_1 ) );

  m_intbnd_edge_ct = m_q.m_edge_ct - num_ext_edges;

  num_ext_edges = std::count_if ( uint_iterator ( m_q.m_vert_ct ),uint_iterator ( m_q.m_vert_ct+ m_intbnd_edge_ct ),
                                  boost::bind ( &QuadDataset::isCellExterior,this,_1 ) );

  if ( num_ext_edges != 0 )
  {
    _ERROR ( " The edge set is not partitioned" );
    exit ( -1 );
  }
}

void QuadDataset::assignGradients()
{
  assignGradient ( this,uint_iterator::iterators ( getCellid ( m_vert_cells ),getCellid ( m_vert_cells+m_intbnd_vert_ct-1 ) +1 ),
                   boost::bind ( &QuadDataset::addCancellablePair,this,_1,_2 ) );
  assignGradient ( this,uint_iterator::iterators ( getCellid ( m_edge_cells ),getCellid ( m_edge_cells+m_intbnd_edge_ct-1 ) +1 ),
                   boost::bind ( &QuadDataset::addCancellablePair,this,_1,_2 ) );
  assignGradient ( this,uint_iterator::iterators ( getCellid ( m_quad_cells ),getCellid ( m_quad_cells+m_intbnd_quad_ct-1 ) +1 ),
                   boost::bind ( &QuadDataset::addCancellablePair,this,_1,_2 ) );
}

struct add_to_set_ftor
{
  typedef bool result_type ;

  QuadDataset * _qds;
  std::set<uint> * _disc;

  add_to_set_ftor ( QuadDataset * qds,std::set<uint> * disc ) :_qds ( qds ),_disc ( disc ) {}

  bool operator() ( uint id )
  {
    return _disc->insert ( id ).second;
  }
};

void QuadDataset::addCpConnection ( uint src,uint dest )
{
  m_mscomplex.m_cps[m_mscomplex.m_id_cp_map[src]]->des.insert ( m_mscomplex.m_id_cp_map[dest] );
  m_mscomplex.m_cps[m_mscomplex.m_id_cp_map[dest]]->asc.insert ( m_mscomplex.m_id_cp_map[src ] );
}

void QuadDataset::computeDiscs()
{
  addCriticalPointsToMSComplex ( &m_mscomplex,this,m_criticalpoints.begin(),m_criticalpoints.end() );

  for ( uint i = 0 ; i < m_criticalpoints.size();i++ )
  {
    if(isCellExterior(m_criticalpoints[i]))
      continue;

    switch(getCell(m_criticalpoints[i])->dim)
    {
      case 0:
        do_gradient_bfs(m_criticalpoints[i],GRADIENT_DIR_UPWARD);break;
      case 2:
        do_gradient_bfs(m_criticalpoints[i],GRADIENT_DIR_DOWNWARD);break;
      default:
        break;
    }
  }

  for(std::vector<ui_pair_t>::iterator iter = m_cancellable_boundry_pairs.begin();
      iter != m_cancellable_boundry_pairs.end();++iter)
  {
    if(m_mscomplex.m_id_cp_map.find(iter->first) != m_mscomplex.m_id_cp_map.end())
    {
      uint cp_ind = m_mscomplex.m_id_cp_map[iter->first];
      m_mscomplex.m_cps[cp_ind]->isBoundryCancelable = true;
    }

    if(m_mscomplex.m_id_cp_map.find(iter->second) != m_mscomplex.m_id_cp_map.end())
    {
      uint cp_ind = m_mscomplex.m_id_cp_map[iter->second];
      m_mscomplex.m_cps[cp_ind]->isBoundryCancelable = true;
    }

    pairCells(iter->first,iter->second);

    // this is done because we dont do a bfs on saddle pts and extreior
    // cp's . These cp's will be incident only on their paired saddle pts

    if(isCellExterior(iter->first) && getCell(iter->first)->dim == 2)
    {
      addCpConnection(iter->first,iter->second);
    }

    if(isCellExterior(iter->second) && getCell(iter->second)->dim == 2)
    {
      addCpConnection(iter->second,iter->first);
    }

  }
}

void QuadDataset::do_gradient_bfs(uint start_cellId,eGradDirection dir)
{
  static uint ( QuadDataset::*getcets[2] ) ( uint,uint* ) const =
  {
    &QuadDataset::getCellFacets,
    &QuadDataset::getCellCofacets
  };

  std::queue<uint> cell_queue;

  // mark here that that cellid has no parent.

  if(m_mscomplex.m_id_cp_map.find(start_cellId) ==
     m_mscomplex.m_id_cp_map.end())
  {
    throw std::logic_error("could not find the start cp cell in the mscomplex");
  }
  uint cp_ind  = m_mscomplex.m_id_cp_map[start_cellId];

  cell_queue.push ( start_cellId );

  while ( !cell_queue.empty() )
  {
    uint top_cell = cell_queue.front();

    cell_queue.pop();

    // mark this cell id as visited

    uint cets[20];

    uint cet_ct = ( this->*getcets[dir] ) ( top_cell,cets );

    for ( uint i = 0 ; i < cet_ct ; i++ )
    {
      if ( isCellCritical ( cets[i] ) )
      {
        if(m_mscomplex.m_id_cp_map.find(cets[i]) ==
           m_mscomplex.m_id_cp_map.end())
        {
          throw std::logic_error("could not find the adj cp cell in the mscomplex");
        }

        uint adj_ind = m_mscomplex.m_id_cp_map[cets[i]];

        if(isCellCritical(start_cellId))
        {
          switch(dir)
          {
          case GRADIENT_DIR_DOWNWARD:
            m_mscomplex.m_cps[cp_ind]->des.insert(adj_ind);
            m_mscomplex.m_cps[adj_ind]->asc.insert(cp_ind);
            break;

          case GRADIENT_DIR_UPWARD:
            m_mscomplex.m_cps[cp_ind]->asc.insert(adj_ind);
            m_mscomplex.m_cps[adj_ind]->des.insert(cp_ind);
            break;
          default:break;
          }
        }
      }
      else
      {
        if ( !isCellExterior ( cets[i] ) )
        {
          uint next_cell = getCellPairId ( cets[i] );

          if ( getCellDim ( top_cell ) == getCellDim ( next_cell ) && next_cell != top_cell )
          {
            cell_queue.push ( next_cell );
          }
        }
      }
    }
  }
}

uint QuadDataset::getNumCriticalPoints()
{
  return m_criticalpoints.size();
}

void QuadDataset::getGenericMsComplex ( generic_mscomplex_t *&ms_out )
{
  convertMSComplexToGeneric ( &m_mscomplex,ms_out,boost::bind ( getUniCellID,this,_1 ) );
}



void QuadDataset::getBoundryCancellablePairs ( std::vector<std::pair<generic_cell_t,generic_cell_t> > &cancellable_boundry_pairs )
{
  for ( uint i = 0 ; i < m_cancellable_boundry_pairs.size();i++ )
  {
    generic_cell_t c1 = getUniCellID ( this,m_cancellable_boundry_pairs[i].first );
    generic_cell_t c2 = getUniCellID ( this,m_cancellable_boundry_pairs[i].second );
    cancellable_boundry_pairs.push_back ( std::make_pair ( c1,c2 ) );
  }
}

// misc utility
void QuadDataset::logCell ( cell *cellid )
{
  switch ( cellid->dim )
  {
  case 0: m_q.logQuad ( m_q.m_verts[cellid-m_vert_cells] );break;
  case 1: m_q.logQuad ( m_q.m_edges[cellid-m_edge_cells] );break;
  case 2: m_q.logQuad ( m_q.m_quads[cellid-m_quad_cells] );break;
  default: break;
  }
}

cell * QuadDataset::getCell ( uint cellid ) const
{
  if ( cellid < m_q.m_vert_ct )
    return &m_vert_cells[cellid];
  else if ( cellid < m_q.m_vert_ct + m_q.m_edge_ct )
    return &m_edge_cells[cellid - m_q.m_vert_ct];
  else if ( cellid < m_q.m_vert_ct+ m_q.m_edge_ct + m_q.m_quad_ct )
    return &m_quad_cells[cellid - m_q.m_vert_ct - m_q.m_edge_ct];

  _ERROR ( "invalid cell id requested" );
  exit ( 1 );
  return NULL;
}

uint QuadDataset::getCellid ( cell * c ) const
{
  switch ( c->dim )
  {
  case 0:return ( c-m_vert_cells );
  case 1:return ( c-m_edge_cells ) + m_q.m_vert_ct;
  case 2:return ( c-m_quad_cells ) + m_q.m_vert_ct + m_q.m_edge_ct;
  }
  _ERROR ( "invalid cell ptr" );
  exit ( 1 );
  return 0;
}

void QuadDataset::destroy()
{
  if ( m_q.m_vert_ct !=0 )
  {
    delete []m_vert_cells;
    delete []m_vertices;
    delete []m_orig_vert_index;
  }

  if ( m_q.m_edge_ct !=0 )
  {
    delete []m_edge_cells;
  }

  if ( m_q.m_quad_ct !=0 )
  {
    delete []m_quad_cells;
    delete []m_orig_quad_index;
  }

  m_q.destroy();

  for ( uint i = 0 ; i < m_mscomplex.m_cps.size();i++ )
  {
    m_mscomplex.m_cps[i]->asc_disc.clear();
    m_mscomplex.m_cps[i]->des_disc.clear();
    delete m_mscomplex.m_cps[i];
  }

  m_mscomplex.m_id_cp_map.clear();
  m_mscomplex.m_cps.clear();

  m_criticalpoints.clear();

  m_cancellable_boundry_pairs.clear();

}

uint QuadDataset::getPointGenericIndex ( uint point ) const
{
  if ( point >= m_q.m_vert_ct )
  {
    _ERROR ( "Invalid point requested " );
    _ERROR ( "point    = "<<point );
    _ERROR ( "point_ct = "<<m_q.m_vert_ct );
  }

  return m_orig_vert_index[point];
}

void  QuadDataset::addCancellablePair ( uint c1,uint c2 )
{
  m_cancellable_boundry_pairs.push_back ( std::make_pair ( c1,c2 ) );

  if(isCellExterior(c1))
    markCellCritical(c1);

  if(isCellExterior(c2))
    markCellCritical(c2);

}

// dataset interface
uint QuadDataset::getCellPairId ( uint cellid ) const
{

  if(getCell ( cellid )->marked ==false)
    throw std::logic_error("this cell has not been marked");


  return getCellid ( getCell ( cellid )->pair );
}

bool QuadDataset::ptLt ( uint cellid1,uint cellid2 ) const
{
  // in case of debug ensure that we are infact getting points here
  if ( m_vertices[cellid1].fn != m_vertices[cellid2].fn )
    return m_vertices[cellid1].fn < m_vertices[cellid2].fn;
  else
    return m_orig_vert_index[cellid1]< m_orig_vert_index[cellid2];

}

uint QuadDataset::getCellPoints ( uint cellid,uint  *points ) const
{
  cell *c = getCell ( cellid );
  switch ( c->dim )
  {
  case 0:
    points[0] = getCellid ( c );
    return 1;
  case 1:
    {
      uint index  = c-m_edge_cells;
      uint qindex = m_q.m_edges[index];

      points[0] = getCellid ( &m_vert_cells[m_q.vertIndex ( qindex ) ] );
      qindex = quad_enext ( qindex );
      points[1] = getCellid ( &m_vert_cells[m_q.vertIndex ( qindex ) ] );
      return 2;
    }

  case 2:
    {
      uint index  = c-m_quad_cells;
      uint qindex = m_q.m_quads[index];

      points[0] = getCellid ( &m_vert_cells[m_q.vertIndex ( qindex ) ] );
      qindex = quad_enext ( qindex );
      points[1] = getCellid ( &m_vert_cells[m_q.vertIndex ( qindex ) ] );
      qindex = quad_enext ( qindex );
      points[2] = getCellid ( &m_vert_cells[m_q.vertIndex ( qindex ) ] );
      qindex = quad_enext ( qindex );
      points[3] = getCellid ( &m_vert_cells[m_q.vertIndex ( qindex ) ] );
      return 4;
    }

  default:
    return 0;
  }
}

uint QuadDataset::getCellFacets ( uint cellid,uint *facets ) const
{
  cell *c = getCell ( cellid );
  switch ( c->dim )
  {
  case 0:
    return 0;
  case 1:
    {
      uint index  = c-m_edge_cells;
      uint qindex = m_q.m_edges[index];

      facets[0] = getCellid ( &m_vert_cells[m_q.vertIndex ( qindex ) ] );
      qindex = quad_enext ( qindex );
      facets[1] = getCellid ( &m_vert_cells[m_q.vertIndex ( qindex ) ] );

      return 2;
    }

  case 2:
    {
      uint index  = c-m_quad_cells;
      uint qindex = m_q.m_quads[index];

      facets[0] = getCellid ( &m_edge_cells[m_q.edgeIndex ( qindex ) ] );
      qindex = quad_enext ( qindex );
      facets[1] = getCellid ( &m_edge_cells[m_q.edgeIndex ( qindex ) ] );
      qindex = quad_enext ( qindex );
      facets[2] = getCellid ( &m_edge_cells[m_q.edgeIndex ( qindex ) ] );
      qindex = quad_enext ( qindex );
      facets[3] = getCellid ( &m_edge_cells[m_q.edgeIndex ( qindex ) ] );
      return 4;
    }

  default:
    return 0;
  }
}

uint QuadDataset::getCellCofacets ( uint cellid,uint *cofacets ) const
{
  cell *c = getCell ( cellid );
  switch ( c->dim )
  {
  case 0:
    {
      uint qstart = m_q.m_verts[c-m_vert_cells];

      uint q=qstart;

      // if this point is on the boundary wind to the boundry
      do
      {
        if ( !m_q.hasFnext ( q ) ) break;
        q = quad_enext ( m_q.quadFnext ( q ) );
      }
      while ( q!=qstart );

      uint num_cofacets = 0 ;
      qstart = q;

      do
      {
        cofacets[num_cofacets++] = getCellid ( &m_edge_cells[m_q.edgeIndex ( q ) ] );

        q = quad_eprev ( q );

        if ( !m_q.hasFnext ( q ) )
        {
          cofacets[num_cofacets++] = getCellid ( &m_edge_cells[m_q.edgeIndex ( q ) ] );

          break;
        }
        q = m_q.quadFnext ( q );
      }
      while ( q != qstart );


      return num_cofacets;

    }
  case 1:
    {
      uint num_cofacets = 0 ;

      uint q = m_q.m_edges[c-m_edge_cells];

      cofacets[num_cofacets++] = getCellid ( &m_quad_cells[m_q.quadIndex ( q ) ] );

      if ( m_q.hasFnext ( q ) )
      {
        q = m_q.quadFnext ( q );

        cofacets[num_cofacets++] = getCellid ( &m_quad_cells[m_q.quadIndex ( q ) ] );
      }
      return num_cofacets;
    }
  case 2:
    return 0;

  default:
    return 0;
  }
}

bool QuadDataset::isPairOrientationCorrect ( uint cellid, uint pairid ) const
{
  cell *c = getCell ( cellid );
  cell *p = getCell ( pairid );
  return c->dim < p->dim;
}

bool QuadDataset::isCellMarked ( uint cellid ) const
{
  return getCell ( cellid )->marked;
}

bool QuadDataset::isCellCritical ( uint cellid ) const
{
  return getCell ( cellid )->critical;
}

void QuadDataset::pairCells ( uint cellid1,uint cellid2 )
{
  getCell ( cellid1 )->pair = getCell ( cellid2 );
  getCell ( cellid2 )->pair = getCell ( cellid1 );

  getCell ( cellid1 )->marked = true;
  getCell ( cellid2 )->marked = true;
}

void QuadDataset::markCellCritical ( uint cellid )
{
  getCell ( cellid )->critical = true;
  getCell ( cellid )->marked   = true;

  m_criticalpoints.push_back ( cellid );
}

uint QuadDataset::getCellDim ( uint cellId ) const
{
  return getCell ( cellId )->dim;
}

uint QuadDataset::getMaxCellDim() const
{
  return 2;
}

bool QuadDataset::isTrueBoundryCell ( uint cellid ) const
{
  cell *c = getCell ( cellid );
  bool isBoundry = false;
  switch ( c->dim )
  {
  case 0:
    {
      uint qstart = m_q.m_verts[c-m_vert_cells];

      uint q=qstart;

      // if this point is on the boundary wind to the boundry
      do
      {
        if ( !m_q.hasFnext ( q ) )
        {
          isBoundry = true;
          break;
        }
        q = quad_enext ( m_q.quadFnext ( q ) );
      }
      while ( q!=qstart );
      break;
    }
  case 1:
    {
      uint q = m_q.m_edges[c-m_edge_cells];

      if ( !m_q.hasFnext ( q ) )
        isBoundry = true;
    }
  case 2:break;
  default:
    _ERROR ( "Should not reach here" );
    exit ( 0 );
    break;

  };

  return isBoundry;
}

bool QuadDataset::isFakeBoundryCell ( uint cellid ) const
{

  cell *c = getCell ( cellid );
  bool isFakeBoundry = false;
  switch ( c->dim )
  {
  case 0:
    {
      uint qstart = m_q.m_verts[c-m_vert_cells];

      uint q=qstart;

      do
      {
        if ( q >= 8*m_intbnd_quad_ct )
          isFakeBoundry = true;

        if ( !m_q.hasFnext ( q ) )
          break;

        q = quad_enext ( m_q.quadFnext ( q ) );
      }
      while ( q!=qstart );
      break;
    }
  case 1:
    {
      uint q = m_q.m_edges[c-m_edge_cells];

      if ( q >= 8*m_intbnd_quad_ct )
        isFakeBoundry = true;

      if ( m_q.hasFnext ( q ) )
      {
        q = m_q.quadFnext ( q ) ;

        if ( q >= 8*m_intbnd_quad_ct )
          isFakeBoundry = true;
      }


    }
  case 2:break;
  default:
    _ERROR ( "Should not reach here" );
    exit ( 0 );
    break;

  };

  return isFakeBoundry&&!isCellExterior(cellid);

}

bool QuadDataset::isCellExterior ( uint cellid ) const
{
  // this method is incorrect for a general quad mesh.

  uint  pts[20];
  uint  pts_ct  = getCellPoints ( cellid,pts );

  for ( uint i = 0 ; i < pts_ct;i++ )
  {
    if ( pts[i] >= m_intbnd_vert_ct )
      return true;
  }

  return false;
}

std::string  QuadDataset::getCellFunctionDescription ( uint cellid ) const
{
  uint pts[20];

  std::stringstream ss;

  uint pts_ct = this->getCellPoints ( cellid,pts );

  ss.precision ( 3 );

  ss<<"[";

  ss<<fixed;

  for ( uint i = 0 ; i < pts_ct;i++ )
  {
    ss<<m_vertices[pts[i]].fn<<":"<<m_orig_vert_index[pts[i]]<<" ";
  }

  ss<<"]";

  return ss.str();
}

std::string QuadDataset::getCellDescription ( uint cellid ) const
{

  uint pts[20];

  std::stringstream ss;

  uint pts_ct = this->getCellPoints ( cellid,pts );

  ss.precision ( 3 );

  ss<<"[";

  ss<<fixed;

  for ( uint i = 0 ; i < pts_ct;i++ )
  {
    ss<<m_orig_vert_index[pts[i]]<<" ";
  }

  ss<<"]";

  return ss.str();
}

// dataset renderable interface
void QuadDataset::getCellCoords ( uint cellid,double &x,double &y,double &z ) const
{

  uint pts[10];
  uint pts_ct = getCellPoints ( cellid,pts );

  x=0.0;y=0.0;z=0.0;

  for ( uint i = 0 ; i <pts_ct;i++ )
  {
    x += m_vertices[pts[i]].x;
    y += m_vertices[pts[i]].y;
    z += m_vertices[pts[i]].z;
  }

  x/= ( double ) pts_ct;
  y/= ( double ) pts_ct;
  z/= ( double ) pts_ct;
}


ArrayRenderer<uint,double> * QuadDataset::createSurfaceRenderer()
{

  return

      createDatasetSurfaceRenderer
      (
          this,
          uint_iterator ( 0 ),
          uint_iterator ( m_q.m_vert_ct ),
          uint_iterator ( m_q.m_vert_ct+m_q.m_edge_ct ),
          uint_iterator ( m_q.m_vert_ct+m_q.m_edge_ct+m_q.m_quad_ct )

          );


}

ArrayRenderer<uint,double> * QuadDataset::createGradientRenderer ( uint dim )
{
  switch ( dim )
  {
  case 0:
    return createDatasetGradientRenderer(this,uint_iterator( 0),uint_iterator( m_q.m_vert_ct ));
  case 1:
    return createDatasetGradientRenderer(this,uint_iterator( m_q.m_vert_ct),uint_iterator(m_q.m_vert_ct +m_q.m_edge_ct ));
  }

  _ERROR ( "gradient higher than dim 1 does not exist." );
  exit ( 0 );
  return NULL;
}

struct getCellCoords_ftor
{
  QuadDataset *ds;

  getCellCoords_ftor ( QuadDataset *_ds ) :ds ( _ds ) {}

  void operator() ( uint id,double &x,double &y,double &z )
  {
    ds->getCellCoords ( id,x,y,z );
  }
};

ArrayRenderer<uint,double> * QuadDataset::createMsGraphRenderer()
{
  ArrayRenderer<uint,double> * ren;

  createCombinatorialStructureRenderer ( &m_mscomplex,ren,getCellCoords_ftor ( this ) );

  return ren;
}

ArrayRenderer<uint,double> * QuadDataset::createMsQuadsRenderer()
{
  return createMsQuadRenderer_func ( this,&m_mscomplex );
}

ArrayRenderer<uint,double> * QuadDataset::createCancellablePairsRenderer()
{
  ArrayRenderer<uint,double> * ren = new ArrayRenderer<uint,double>;

  for ( uint i = 0 ; i < m_cancellable_boundry_pairs.size();i++ )
  {
    uint v1 = m_cancellable_boundry_pairs[i].first;
    uint v2 = m_cancellable_boundry_pairs[i].second;

    double x,y,z;
    double _x,_y,_z;

    getCellCoords ( v1,x,y,z );
    getCellCoords ( v2,_x,_y,_z );

    ren->add_vertex ( v1,x,y,z );
    ren->add_vertex ( v2,_x,_y,_z );

    ren->add_arrow ( v1,v2 );

    ren->set_vertex_color ( v1,1,1,1 );
    ren->set_vertex_color ( v2,0.5,0.5,0.5 );
  }

  ren->prepare_for_render();
  ren->sqeeze();

  return ren;
}
