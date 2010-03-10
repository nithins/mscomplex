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


#ifndef __GRID_DATASET_H_INCLUDED_
#define __GRID_DATASET_H_INCLUDED_

#include <vector>

#include <discreteMorseDS.h>

#include <cpputils.h>
#include <rectangle_complex.h>

#include <boost/multi_array.hpp>



class GridMSComplex:
    public MSComplex<rectangle_complex<short int>::point_def>
{
  typedef int16_t                          cell_coord_t;
  typedef double                           cell_fn_t;
  typedef rectangle_complex<cell_coord_t>  rect_cmplx_t;
  typedef rect_cmplx_t::rectangle_def      rect_t;
  typedef rect_cmplx_t::point_def          cellid_t;
  typedef rect_cmplx_t::point_def          rect_point_t;
  typedef rect_cmplx_t::size_def           rect_size_t;

  typedef std::vector<cellid_t>            cellid_list_t;
  typedef std::pair<cellid_t,cellid_t>     cellid_pair_t;
  typedef std::vector<cellid_pair_t>       cellid_pair_list_t;

  typedef GridMSComplex                    mscomplex_t;
  typedef mscomplex_t::critical_point      critpt_t;
  typedef mscomplex_t::critical_point::connection_t                 conn_t;
  typedef mscomplex_t::critical_point::connection_t::iterator       conn_iter_t;
  typedef mscomplex_t::critical_point::connection_t::const_iterator const_conn_iter_t;

  typedef std::vector<cell_fn_t>           cp_fn_list_t;
public:

  rect_t        m_rect;
  rect_t        m_ext_rect;
  cp_fn_list_t  m_cp_fns;

  void clear();

  static mscomplex_t * merge_up(const mscomplex_t& msc1,const mscomplex_t& msc2);

  void merge_down(mscomplex_t& msc1,mscomplex_t& msc2);

  GridMSComplex(rect_t r,rect_t e):m_rect(r),m_ext_rect(e)
  {
  }
};

class GridDataset/*:
    public IDiscreteDataset<rectangle_complex<short int>::point_def>*/
{


public:

  enum eCellFlags
  {
    CELLFLAG_UNKNOWN = 0,
    CELLFLAG_PAIRED  = 1,
    CELLFLAG_CRITCAL = 2,
  };

  typedef int16_t                          cell_coord_t;
  typedef float                            cell_fn_t;
  typedef rectangle_complex<cell_coord_t>  rect_cmplx_t;
  typedef rect_cmplx_t::rectangle_def      rect_t;
  typedef rect_cmplx_t::point_def          cellid_t;
  typedef rect_cmplx_t::point_def          rect_point_t;
  typedef rect_cmplx_t::size_def           rect_size_t;

  typedef std::vector<cellid_t>            cellid_list_t;
  typedef std::pair<cellid_t,cellid_t>     cellid_pair_t;
  typedef std::vector<cellid_pair_t>       cellid_pair_list_t;

  typedef GridMSComplex                    mscomplex_t;
  typedef mscomplex_t::critical_point      critpt_t;
  typedef critpt_t::connection_t           critpt_conn_t;


  typedef int8_t                            cell_flag_t;
  typedef boost::multi_array<cell_fn_t,2>   varray_t;
  typedef boost::multi_array<cellid_t,2>    cellpair_array_t;
  typedef boost::multi_array<cell_flag_t,2> cellflag_array_t;

private:

  class pt_comp_t
  {
    GridDataset *pOwn;
  public:
    pt_comp_t(GridDataset *o):pOwn(o){}

    bool operator()(cellid_t c1,cellid_t c2)
    {
      return pOwn->ptLt(c1,c2);
    }
  };


  rect_t           m_rect;
  rect_t           m_ext_rect;

  varray_t         m_vertex_fns; // defined on the vertices of bounding rect
  cellpair_array_t m_cell_pairs;
  cellflag_array_t m_cell_flags;
  cellid_list_t    m_critical_cells;

  pt_comp_t        m_ptcomp;

public:

  // initialization of the dataset

  GridDataset ( const rect_t &r,const rect_t &e );

  void  init();

  void  set_cell_fn ( cellid_t c,cell_fn_t f );

  void  clear_graddata();

  // actual algorithm work
public:

  void  assignGradients();

  void  assignGradients_ocl();

  void  computeConnectivity(mscomplex_t *msgraph);

  // sub tasks of the above routines
public:
  void  collateCriticalPoints();

  // dataset interface
public:

  cellid_t   getCellPairId ( cellid_t ) const;

  inline bool   ptLt ( cellid_t c1,cellid_t c2) const
  {
    double f1 = m_vertex_fns[c1[0]>>1][c1[1]>>1];
    double f2 = m_vertex_fns[c2[0]>>1][c2[1]>>1];

    if (f1 != f2)
      return f1 < f2;

    return c1<c2;
  }

  bool   compareCells( cellid_t ,cellid_t ) const;

  uint   getCellPoints ( cellid_t ,cellid_t  * ) const;

  uint   getCellFacets ( cellid_t ,cellid_t * ) const;

  uint   getCellCofacets ( cellid_t ,cellid_t * ) const;

  bool   isPairOrientationCorrect ( cellid_t c, cellid_t p ) const;

  bool   isCellMarked ( cellid_t c ) const;

  bool   isCellCritical ( cellid_t c ) const;

  bool   isCellPaired ( cellid_t c ) const;

  void   pairCells ( cellid_t c,cellid_t p );

  void   markCellCritical ( cellid_t c );

  inline uint getCellDim ( cellid_t c ) const;

  uint   getMaxCellDim() const;

  bool   isTrueBoundryCell ( cellid_t c ) const;

  bool   isFakeBoundryCell ( cellid_t c ) const;

  bool   isCellExterior ( cellid_t c ) const;

  std::string  getCellFunctionDescription ( cellid_t pt ) const;

  std::string getCellDescription ( cellid_t cellid ) const;

  // misc functions
public:
  inline static uint s_getCellDim ( cellid_t c )
  {
    return ( ( c[0]&0x01 ) + ( c[1]&0x01 ) );
  }

  inline rect_t get_rect()
  {
    return m_rect;
  }

  inline rect_t get_ext_rect()
  {
    return m_ext_rect;
  }

  // return fn at point .. averge of points for higher dims
  cell_fn_t get_cell_fn ( cellid_t c ) const;

  // for rendering support
public:
  void getCellCoord ( cellid_t c,double &x,double &y,double &z );

  // opencl implementations
public:

  static void init_opencl();

  static void stop_opencl();

};

inline uint GridDataset::getCellDim ( cellid_t c ) const
{
  return ( ( c[0]&0x01 ) + ( c[1]&0x01 ) );
}




#endif
