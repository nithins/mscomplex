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
#include <cfloat>

#include <boost/bind.hpp>

#include <cpputils.h>
#include <logutil.h>
#include <tri_dataset.h>

#include <discreteMorseDSRenderFuncs.h>
#include <discreteMorseAlgorithm.h>


typedef TriGenericCell2D<uint>   generic_cell_t;

generic_cell_t getUniCellID ( const TriDataset *ds, uint cellid )
{
  uint points[20];

  uint point_ct;

  point_ct = ds->getCellPoints ( cellid, points );

  std::transform ( points, points + point_ct, points, boost::bind ( &TriDataset::getPointGenericIndex, ds, _1 ) );

  return createTriGenericCell2D ( points, point_ct, ds->getCellDim ( cellid ) );
}


using namespace std;

// functions to setup the tri edge structure
void TriDataset::setNumVerts ( const uint & vert_ct )
{
  m_q.setNumVerts ( vert_ct );
}

void TriDataset::setNumTris ( const uint &tri_ct )
{
  m_q.setNumTris ( tri_ct );
}

void TriDataset::setNumExtVert ( const uint &ct )
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

void TriDataset::setNumExtTri ( const uint &ct )
{
  m_intbnd_tri_ct  = m_q.m_tri_ct - ct;
  m_tri_cells      = new cell[m_q.m_tri_ct];
  m_orig_tri_index = new uint[m_q.m_tri_ct];

  for ( uint i = 0 ; i < m_q.m_tri_ct;i++ )
  {
    m_tri_cells[i].dim = 2;
  }
}


void TriDataset::setVert ( const uint &vno, const double &fn, const uint& ind, const double& x, const double& y, const double& z )
{
  m_vertices       [ vno ].x  = x;
  m_vertices       [ vno ].y  = y;
  m_vertices       [ vno ].z  = z;
  m_vertices       [ vno ].fn = fn;
  m_orig_vert_index[ vno ]    = ind;
}

void TriDataset::startAddingTris()
{
  m_q.start_adding_tris();
}

void TriDataset::addTri ( const uint *v , const uint &qno, const uint &ind )
{
  m_q.add_tri ( v );

  m_orig_tri_index[qno] = ind;
}

void TriDataset::endAddingTris()
{
  m_q.end_adding_tris();

  m_edge_cells = new cell[m_q.m_edge_ct];

  for ( uint i = 0 ; i < m_q.m_edge_ct;i++ )
  {
    m_edge_cells[i].dim = 1;
  }

  m_isedge_ext = new bool[m_q.m_edge_ct];

  for ( uint i = 0 ; i < m_q.m_edge_ct;i++ )
  {
    uint edge_id = i + m_q.m_vert_ct;

    uint tris[20];

    uint tris_ct = getCellCofacets ( edge_id, tris );

    m_isedge_ext[i] = true;

    for ( uint j = 0 ; j < tris_ct;j++ )
    {
      if ( isCellExterior ( tris[j] ) == false )
        m_isedge_ext[i] = false;
    }
  }

  uint num_ext_edges = 0;

  num_ext_edges = std::count_if ( uint_iterator ( m_q.m_vert_ct ), uint_iterator ( m_q.m_vert_ct + m_q.m_edge_ct ),
                                  boost::bind ( &TriDataset::isCellExterior, this, _1 ) );

  m_intbnd_edge_ct = m_q.m_edge_ct - num_ext_edges;

  m_intbnd_edge_ids = new uint[m_intbnd_edge_ct];

  uint intbnd_edge_ind = 0;

  for ( uint i = m_q.m_vert_ct;i < m_q.m_vert_ct + m_q.m_edge_ct;i++ )
    if ( isCellExterior ( i ) == false )
      m_intbnd_edge_ids[intbnd_edge_ind++] = i;

}

template <typename id_t>
void add_pairs_to_list ( std::vector<std::pair<id_t, id_t> > *list  , id_t c1, id_t c2 )
{
  list->push_back ( std::make_pair ( c1, c2 ) );
}


void TriDataset::assignGradients()
{
  assignGradient ( this, uint_iterator::iterators ( getCellid ( m_vert_cells ), getCellid ( m_vert_cells + m_intbnd_vert_ct - 1 ) + 1 ),
                   boost::bind ( &TriDataset::addCancellablePair, this, _1, _2 ) );
  assignGradient ( this, m_intbnd_edge_ids, m_intbnd_edge_ids + m_intbnd_edge_ct,
                   boost::bind ( &TriDataset::addCancellablePair, this, _1, _2 ) );
  assignGradient ( this, uint_iterator::iterators ( getCellid ( m_tri_cells ), getCellid ( m_tri_cells + m_intbnd_tri_ct - 1 ) + 1 ),
                   boost::bind ( &TriDataset::addCancellablePair, this, _1, _2 ) );
}

struct add_to_set_ftor
{
  typedef bool result_type ;

  TriDataset * _qds;
  std::set<uint> * _disc;

  add_to_set_ftor ( TriDataset * qds, std::set<uint> * disc ) : _qds ( qds ), _disc ( disc ) {}

  bool operator() ( uint id )
  {
    return _disc->insert ( id ).second;
  }
};

void TriDataset::addCpConnection ( uint src, uint dest )
{
  m_mscomplex.m_cps[m_mscomplex.m_id_cp_map[src]]->des.insert ( m_mscomplex.m_id_cp_map[dest] );
  m_mscomplex.m_cps[m_mscomplex.m_id_cp_map[dest]]->asc.insert ( m_mscomplex.m_id_cp_map[src ] );
}

void TriDataset::computeDiscs ()
{
  addCriticalPointsToMSComplex ( &m_mscomplex, this, m_criticalpoints.begin(), m_criticalpoints.end() );

  for ( uint i = 0 ; i < m_criticalpoints.size();i++ )
  {
//     uint cellid =  m_criticalpoints[i];
    //
//     cell * cell = getCell ( cellid );
    //
//     switch ( cell->dim )
//     {
    //
//       case 2:
    //
//         track_gradient_tree_bfs
//         (
//           this,
//           m_criticalpoints[i],
//           GRADIENT_DIR_DOWNWARD,
//           add_to_set_ftor ( this,&m_mscomplex.m_cps[i]->des_disc ),
//           boost::bind ( &TriDataset::addCpConnection,this,m_criticalpoints[i],_1 )
//         );
    //
//         break;
    //
//       case 0:
    //
//         track_gradient_tree_bfs
//         (
//           this,
//           m_criticalpoints[i],
//           GRADIENT_DIR_UPWARD,
//           add_to_set_ftor ( this,&m_mscomplex.m_cps[i]->asc_disc ),
//           boost::bind ( &TriDataset::addCpConnection,this,_1,m_criticalpoints[i] )
//         );
    //
//         break;
    //
//       default:
//         break;
//     }


    track_gradient_tree_with_closure
    (
      this,
      m_criticalpoints[i],
      GRADIENT_DIR_DOWNWARD,
      add_to_set_ftor ( this, &m_mscomplex.m_cps[i]->des_disc ),
      boost::bind ( &TriDataset::addCpConnection, this, m_criticalpoints[i], _1 )
    );

    track_gradient_tree_with_closure
    (
      this,
      m_criticalpoints[i],
      GRADIENT_DIR_UPWARD,
      add_to_set_ftor ( this, &m_mscomplex.m_cps[i]->asc_disc ),
      do_nothing_functor<uint>()
    );

//     m_mscomplex.m_cps[i]->asc_disc.insert ( m_criticalpoints[i] );
    //
//     m_mscomplex.m_cps[i]->des_disc.insert ( m_criticalpoints[i] );
    //
//     for ( std::multiset<uint>::iterator adj_iter = m_mscomplex.m_cps[i]->asc.begin();adj_iter != m_mscomplex.m_cps[i]->asc.end();++adj_iter )
//     {
//       m_mscomplex.m_cps[i]->asc_disc.insert ( m_mscomplex.m_cps[*adj_iter]->cellid );
//     }
    //
//     for ( std::multiset<uint>::iterator adj_iter = m_mscomplex.m_cps[i]->des.begin();adj_iter != m_mscomplex.m_cps[i]->des.end();++adj_iter )
//     {
//       m_mscomplex.m_cps[i]->des_disc.insert ( m_mscomplex.m_cps[*adj_iter]->cellid );
//     }
  }

//  simplifyComplex ( &m_mscomplex,this );
}

uint TriDataset::getNumCriticalPoints()
{
  return m_criticalpoints.size();
}

void TriDataset::getGenericMsComplex ( generic_mscomplex_t *&ms_out )
{
  convertMSComplexToGeneric ( &m_mscomplex, ms_out, boost::bind ( getUniCellID, this, _1 ) );
}



void TriDataset::getBoundryCancellablePairs ( std::vector<std::pair<generic_cell_t, generic_cell_t> > &cancellable_boundry_pairs )
{
  for ( uint i = 0 ; i < m_cancellable_boundry_pairs.size();i++ )
  {
    generic_cell_t c1 = getUniCellID ( this, m_cancellable_boundry_pairs[i].first );
    generic_cell_t c2 = getUniCellID ( this, m_cancellable_boundry_pairs[i].second );
    cancellable_boundry_pairs.push_back ( std::make_pair ( c1, c2 ) );
  }
}

// misc utility
void TriDataset::logCell ( cell *cellid )
{
  switch ( cellid->dim )
  {

    case 0:
      m_q.logTri ( m_q.m_verts[cellid-m_vert_cells] );
      break;

    case 1:
      m_q.logTri ( m_q.m_edges[cellid-m_edge_cells] );
      break;

    case 2:
      m_q.logTri ( m_q.m_tris[cellid-m_tri_cells] );
      break;

    default:
      break;
  }
}

cell * TriDataset::getCell ( uint cellid ) const
{
  if ( cellid < m_q.m_vert_ct )
    return &m_vert_cells[cellid];
  else
    if ( cellid < m_q.m_vert_ct + m_q.m_edge_ct )
      return &m_edge_cells[cellid - m_q.m_vert_ct];
    else
      if ( cellid < m_q.m_vert_ct + m_q.m_edge_ct + m_q.m_tri_ct )
        return &m_tri_cells[cellid - m_q.m_vert_ct - m_q.m_edge_ct];

  _ERROR ( "invalid cell id requested" );

  exit ( 1 );

  return NULL;
}

uint TriDataset::getCellid ( cell * c ) const
{
  switch ( c->dim )
  {

    case 0:
      return ( c -m_vert_cells );

    case 1:
      return ( c -m_edge_cells ) + m_q.m_vert_ct;

    case 2:
      return ( c -m_tri_cells ) + m_q.m_vert_ct + m_q.m_edge_ct;
  }

  _ERROR ( "invalid cell ptr" );

  exit ( 1 );
  return 0;
}

void TriDataset::destroy()
{
  if ( m_q.m_vert_ct != 0 )
  {
    delete []m_vert_cells;
    delete []m_vertices;
    delete []m_orig_vert_index;
  }

  if ( m_q.m_edge_ct != 0 )
  {
    delete []m_edge_cells;
    delete []m_intbnd_edge_ids;
    delete []m_isedge_ext;
  }

  if ( m_q.m_tri_ct != 0 )
  {
    delete []m_tri_cells;
    delete []m_orig_tri_index;
  }

  m_q.destroy();

  for ( uint i = 0 ; i < m_mscomplex.m_cps.size();i++ )
  {
    m_mscomplex.m_cps[i]->asc_disc.clear();
    m_mscomplex.m_cps[i]->des_disc.clear();
    delete m_mscomplex.m_cps[i];
  }

//  if ( m_mscomplex.m_cps.size() != 0 )
//    delete []m_mscomplex.m_cps;

  m_mscomplex.m_id_cp_map.clear();

  m_criticalpoints.clear();

  m_cancellable_boundry_pairs.clear();

}

uint TriDataset::getPointGenericIndex ( uint point ) const
{
  if ( point >= m_q.m_vert_ct )
  {
    _ERROR ( "Invalid point requested " );
    _ERROR ( "point    = " << point );
    _ERROR ( "point_ct = " << m_q.m_vert_ct );
  }

  return m_orig_vert_index[point];
}

void  TriDataset::addCancellablePair ( uint c1, uint c2 )
{
  m_cancellable_boundry_pairs.push_back ( std::make_pair ( c1, c2 ) );
}

// dataset interface
uint TriDataset::getCellPairId ( uint cellid ) const
{
  if ( getCell ( cellid )->marked == false )
  {
    _ERROR ( "unmarked cell pair requested" );
    _LOG_VAR ( cellid );
    _LOG_VAR ( isCellExterior ( cellid ) );
    _LOG_VAR ( getCell ( cellid )->dim );
    _LOG_VAR ( getCell ( cellid )->critical );
    _LOG_VAR ( getCell ( cellid )->marked );
    _LOG_VAR ( getCell ( cellid )->pair );

    _LOG_VAR ( getCellid ( &m_tri_cells[0] ) );
    _LOG_VAR ( getCellid ( &m_tri_cells[m_intbnd_tri_ct] ) );
  }

  return getCellid ( getCell ( cellid )->pair );
}

bool TriDataset::ptLt ( uint cellid1, uint cellid2 ) const
{
  // in case of debug ensure that we are infact getting points here
  if ( m_vertices[cellid1].fn != m_vertices[cellid2].fn )
    return m_vertices[cellid1].fn < m_vertices[cellid2].fn;
  else
    return m_orig_vert_index[cellid1] < m_orig_vert_index[cellid2];

}

uint TriDataset::getCellPoints ( uint cellid, uint  *points ) const
{
  cell *c = getCell ( cellid );

  switch ( c->dim )
  {

    case 0:
      points[0] = getCellid ( c );
      return 1;

    case 1:
      {
        uint index  = c - m_edge_cells;
        uint qindex = m_q.m_edges[index];

        points[0] = getCellid ( &m_vert_cells[m_q.vertIndex ( qindex ) ] );
        qindex = tri_enext ( qindex );
        points[1] = getCellid ( &m_vert_cells[m_q.vertIndex ( qindex ) ] );
        return 2;
      }

    case 2:
      {
        uint index  = c - m_tri_cells;
        uint qindex = m_q.m_tris[index];

        points[0] = getCellid ( &m_vert_cells[m_q.vertIndex ( qindex ) ] );
        qindex = tri_enext ( qindex );
        points[1] = getCellid ( &m_vert_cells[m_q.vertIndex ( qindex ) ] );
        qindex = tri_enext ( qindex );
        points[2] = getCellid ( &m_vert_cells[m_q.vertIndex ( qindex ) ] );
        return 3;
      }

    default:
      return 0;
  }
}

uint TriDataset::getCellFacets ( uint cellid, uint *facets ) const
{
  cell *c = getCell ( cellid );

  switch ( c->dim )
  {

    case 0:
      return 0;

    case 1:
      {
        uint index  = c - m_edge_cells;
        uint qindex = m_q.m_edges[index];

        facets[0] = getCellid ( &m_vert_cells[m_q.vertIndex ( qindex ) ] );
        qindex = tri_enext ( qindex );
        facets[1] = getCellid ( &m_vert_cells[m_q.vertIndex ( qindex ) ] );

        return 2;
      }

    case 2:
      {
        uint index  = c - m_tri_cells;
        uint qindex = m_q.m_tris[index];

        facets[0] = getCellid ( &m_edge_cells[m_q.edgeIndex ( qindex ) ] );
        qindex = tri_enext ( qindex );
        facets[1] = getCellid ( &m_edge_cells[m_q.edgeIndex ( qindex ) ] );
        qindex = tri_enext ( qindex );
        facets[2] = getCellid ( &m_edge_cells[m_q.edgeIndex ( qindex ) ] );
        return 3;
      }

    default:
      return 0;
  }
}

uint TriDataset::getCellCofacets ( uint cellid, uint *cofacets ) const
{
  cell *c = getCell ( cellid );

  switch ( c->dim )
  {

    case 0:
      {
        uint qstart = m_q.m_verts[c-m_vert_cells];

        uint q = qstart;

        // if this point is on the boundary wind to the boundry

        do
        {
          if ( !m_q.hasFnext ( q ) )
            break;

          q = tri_enext ( m_q.triFnext ( q ) );
        }
        while ( q != qstart );

        uint num_cofacets = 0 ;

        qstart = q;

        do
        {
          cofacets[num_cofacets++] = getCellid ( &m_edge_cells[m_q.edgeIndex ( q ) ] );

          q = tri_eprev ( q );

          if ( !m_q.hasFnext ( q ) )
          {
            cofacets[num_cofacets++] = getCellid ( &m_edge_cells[m_q.edgeIndex ( q ) ] );

            break;
          }

          q = m_q.triFnext ( q );

          if ( num_cofacets > m_q.m_vert_degree[c-m_vert_cells] )
          {
            _ERROR ( "Vertex No = " << ( c - m_vert_cells ) << " qstart =" << qstart << "::Num cofacets > vertex degree" );
            break;
          }
        }
        while ( q != qstart );


        return num_cofacets;

      }

    case 1:
      {
        uint num_cofacets = 0 ;

        uint q = m_q.m_edges[c-m_edge_cells];

        cofacets[num_cofacets++] = getCellid ( &m_tri_cells[m_q.triIndex ( q ) ] );

        if ( m_q.hasFnext ( q ) )
        {
          q = m_q.triFnext ( q );

          cofacets[num_cofacets++] = getCellid ( &m_tri_cells[m_q.triIndex ( q ) ] );
        }

        return num_cofacets;
      }

    case 2:
      return 0;

    default:
      return 0;
  }
}

bool TriDataset::isPairOrientationCorrect ( uint cellid, uint pairid ) const
{
  cell *c = getCell ( cellid );
  cell *p = getCell ( pairid );
  return c->dim < p->dim;
}

bool TriDataset::isCellMarked ( uint cellid ) const
{
  return getCell ( cellid )->marked;
}

bool TriDataset::isCellCritical ( uint cellid ) const
{
  return getCell ( cellid )->critical;
}

void TriDataset::pairCells ( uint cellid1, uint cellid2 )
{
  getCell ( cellid1 )->pair = getCell ( cellid2 );
  getCell ( cellid2 )->pair = getCell ( cellid1 );

  getCell ( cellid1 )->marked = true;
  getCell ( cellid2 )->marked = true;
}

void TriDataset::markCellCritical ( uint cellid )
{
  getCell ( cellid )->critical = true;
  getCell ( cellid )->marked   = true;

  if ( !isCellExterior ( cellid ) )
    m_criticalpoints.push_back ( cellid );
}

uint TriDataset::getCellDim ( uint cellId ) const
{
  return getCell ( cellId )->dim;
}

uint TriDataset::getMaxCellDim() const
{
  return 2;
}

bool TriDataset::isTrueBoundryCell ( uint cellid ) const
{
  cell *c = getCell ( cellid );
  bool isBoundry = false;

  switch ( c->dim )
  {

    case 0:
      {
        uint qstart = m_q.m_verts[c-m_vert_cells];

        uint q = qstart;

        // if this point is on the boundary wind to the boundry

        do
        {
          if ( !m_q.hasFnext ( q ) )
          {
            isBoundry = true;
            break;
          }

          q = tri_enext ( m_q.triFnext ( q ) );
        }
        while ( q != qstart );

        break;
      }

    case 1:
      {
        uint q = m_q.m_edges[c-m_edge_cells];

        if ( !m_q.hasFnext ( q ) )
          isBoundry = true;
      }

    case 2:
      break;

    default:
      _ERROR ( "Should not reach here" );
      exit ( 0 );
      break;

  };

  return isBoundry;
}

bool TriDataset::isFakeBoundryCell ( uint cellid ) const
{

  cell *c = getCell ( cellid );
  bool isFakeBoundry = false;

  switch ( c->dim )
  {

    case 0:
      {
        uint qstart = m_q.m_verts[c-m_vert_cells];

        uint q = qstart;

        do
        {
          if ( q >= 6*m_intbnd_tri_ct )
            isFakeBoundry = true;

          if ( !m_q.hasFnext ( q ) )
            break;

          q = tri_enext ( m_q.triFnext ( q ) );
        }
        while ( q != qstart );

        break;
      }

    case 1:
      {
        uint q = m_q.m_edges[c-m_edge_cells];

        if ( q >= 6*m_intbnd_tri_ct )
          isFakeBoundry = true;

        if ( m_q.hasFnext ( q ) )
        {
          q = m_q.triFnext ( q ) ;

          if ( q >= 6*m_intbnd_tri_ct )
            isFakeBoundry = true;
        }
      }

    case 2:
      break;

    default:
      _ERROR ( "Should not reach here" );
      exit ( 0 );
      break;

  };

  return isFakeBoundry;

}

bool TriDataset::isCellExterior ( uint cellid ) const
{
  switch ( getCellDim ( cellid ) )
  {

    case 0:
      return ( cellid >= m_intbnd_vert_ct );

    case 1:
      return m_isedge_ext[cellid - m_q.m_vert_ct];

    case 2:
      return ( cellid - m_q.m_vert_ct - m_q.m_edge_ct >= m_intbnd_tri_ct );

    default:
      _ERROR ( "invalid celldim" );
      _LOG_VAR ( cellid );
      _LOG_VAR ( getCellDim ( cellid ) );
      exit ( -1 );
      return false;
  }
}

std::string  TriDataset::getCellFunctionDescription ( uint cellid ) const
{
  uint pts[20];

  std::stringstream ss;

  uint pts_ct = this->getCellPoints ( cellid, pts );

  ss.precision ( 3 );

  ss << "[";

  ss << fixed;

  for ( uint i = 0 ; i < pts_ct;i++ )
  {
    ss << m_vertices[pts[i]].fn << ":" << m_orig_vert_index[pts[i]] << " ";
  }

  ss << "]";

  return ss.str();
}

std::string TriDataset::getCellDescription ( uint cellid ) const
{

  uint pts[20];

  std::stringstream ss;

  uint pts_ct = this->getCellPoints ( cellid, pts );

  ss.precision ( 3 );

  ss << "[";

  ss << fixed;

  for ( uint i = 0 ; i < pts_ct;i++ )
  {
    ss << m_orig_vert_index[pts[i]] << " ";
  }

  ss << "]";

  return ss.str();
}

// dataset renderable interface
void TriDataset::getCellCoords ( uint cellid, double &x, double &y, double &z ) const
{

  uint pts[10];
  uint pts_ct = getCellPoints ( cellid, pts );

  x = 0.0;
  y = 0.0;
  z = 0.0;

  for ( uint i = 0 ; i < pts_ct;i++ )
  {
    x += m_vertices[pts[i]].x;
    y += m_vertices[pts[i]].y;
    z += m_vertices[pts[i]].z;
  }

  x /= ( double ) pts_ct;

  y /= ( double ) pts_ct;
  z /= ( double ) pts_ct;
}


ArrayRenderer<uint, double> * TriDataset::createSurfaceRenderer()
{

  return

    createDatasetSurfaceRenderer
    (
      this,
      uint_iterator ( 0 ), uint_iterator ( m_intbnd_vert_ct ),
      uint_iterator ( m_q.m_vert_ct + m_q.m_edge_ct ), uint_iterator ( m_q.m_vert_ct + m_q.m_edge_ct + m_intbnd_tri_ct )
    );


}

ArrayRenderer<uint, double> * TriDataset::createGradientRenderer ( uint dim )
{
  switch ( dim )
  {

    case 0:
      return createDatasetGradientRenderer ( this, uint_iterator ( 0 ), uint_iterator ( m_q.m_vert_ct ) );

    case 1:
      return createDatasetGradientRenderer ( this, uint_iterator ( m_q.m_vert_ct ), uint_iterator ( m_q.m_vert_ct + m_q.m_edge_ct ) );

  }

  _ERROR ( "gradient higher than dim 1 does not exist." );

  exit ( 0 );
  return NULL;
}

ArrayRenderer<uint, double> * TriDataset::createCriticalPointRenderer ()
{
  ArrayRenderer<uint, double> * ren;

//   createCritPtRen
//   (
//     &m_mscomplex,ren,
//     boost::bind ( &TriDataset::getCellCoords,this,_1,_2,_3,_4 ),
//     boost::bind ( &TriDataset::getCellDim,this,_1 )
//   );

  createCritPtRen ( this, ren, m_criticalpoints.begin(), m_criticalpoints.end() );

  return ren;
}

struct getCellCoords_ftor
{
  TriDataset *ds;

  getCellCoords_ftor ( TriDataset *_ds ) : ds ( _ds ) {}

  void operator() ( uint id,double &x, double &y, double &z )
  {
    ds->getCellCoords ( id, x, y, z );
  }
};

ArrayRenderer<uint, double> * TriDataset::createMsGraphRenderer()
{
  ArrayRenderer<uint, double> * ren;

  createCombinatorialStructureRenderer ( &m_mscomplex, ren, getCellCoords_ftor ( this ) );

  return ren;
}

ArrayRenderer<uint, double> * TriDataset::createMsQuadsRenderer()
{
  return createMsQuadRenderer_func ( this, &m_mscomplex );
}

ArrayRenderer<uint, double> * TriDataset::createCancellablePairsRenderer()
{
  ArrayRenderer<uint, double> * ren = new ArrayRenderer<uint, double>;

  for ( uint i = 0 ; i < m_cancellable_boundry_pairs.size();i++ )
  {
    uint v1 = m_cancellable_boundry_pairs[i].first;
    uint v2 = m_cancellable_boundry_pairs[i].second;

    double x, y, z;
    double _x, _y, _z;

    getCellCoords ( v1, x, y, z );
    getCellCoords ( v2, _x, _y, _z );

    ren->add_vertex ( v1, x, y, z );
    ren->add_vertex ( v2, _x, _y, _z );

    ren->add_arrow ( v1, v2 );

    ren->set_vertex_color ( v1, 1, 1, 1 );
    ren->set_vertex_color ( v2, 0.5, 0.5, 0.5 );
  }

  ren->prepare_for_render();

  ren->sqeeze();

  return ren;
}

ArrayRenderer<uint, double> * TriDataset::createUnpairedCellsRenderer()
{

  ArrayRenderer<uint, double> * ren = new ArrayRenderer<uint, double>;

  for ( uint i = 0 ; i < m_intbnd_tri_ct;i++ )
  {
    if ( m_tri_cells[i].marked == false )
    {
      uint cellid = getCellid ( m_tri_cells + i );
      double x, y, z;

      getCellCoords ( cellid, x, y, z );
      ren->add_vertex ( cellid, x, y, z );
      ren->set_vertex_color ( cellid, 1.0, 0.0, 0.0 );

    }
  }

  ren->prepare_for_render();

  ren->sqeeze();

  return ren;
}

ArrayRenderer<uint, double> * TriDataset::createExteriorSurfaceRenderer()
{
  ArrayRenderer<uint, double> * ren = new ArrayRenderer<uint, double>;

  for ( uint i = m_intbnd_tri_ct ; i < m_q.m_tri_ct;i++ )
  {
    uint pts[20];
    uint cellid = getCellid ( m_tri_cells + i );
    uint pts_ct = getCellPoints ( cellid, pts );

    for ( uint j = 0 ; j < pts_ct ; j++ )
    {
      double x, y, z;

      getCellCoords ( pts[j], x, y, z );
      ren->add_vertex ( pts[j], x, y, z );
    }
  }

  for ( uint i = m_intbnd_tri_ct ; i < m_q.m_tri_ct;i++ )
  {
    uint pts[20];
    uint cellid = getCellid ( m_tri_cells + i );
    uint pts_ct = getCellPoints ( cellid, pts );

    if ( pts_ct == 3 )
      ren->add_triangle ( pts[0], pts[1], pts[2] );
    else
      if ( pts_ct == 4 )
        ren->add_quad ( pts[0], pts[1], pts[2], pts[3] );

  }

  ren->prepare_for_render();

  ren->sqeeze();


  return ren;
}
