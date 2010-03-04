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



class GridDataset:
    public IDiscreteDataset<rectangle_complex<int>::point_def>
{


public:

  enum eCellFlags
  {
    CELLFLAG_UNKNOWN = 0,
    CELLFLAG_PAIRED  = 1,
    CELLFLAG_CRITCAL = 2,
  };


  typedef int                              cell_coord_t;
  typedef double                           cell_fn_t;
  typedef rectangle_complex<cell_coord_t>  rect_cmplx_t;
  typedef rect_cmplx_t::rectangle_def      rect_t;
  typedef rect_cmplx_t::point_def          cellid_t;
  typedef rect_cmplx_t::size_def           rect_size_t;

  typedef std::vector<cellid_t>            cellid_list_t;
  typedef std::pair<cellid_t,cellid_t>     cellid_pair_t;
  typedef std::vector<cellid_pair_t>       cellid_pair_list_t;

  typedef MSComplex<cellid_t>              mscomplex_t;
  typedef mscomplex_t::critical_point      critpt_t;
  typedef critpt_t::connection_t           critpt_conn_t;


  typedef boost::multi_array<cell_fn_t,2>  varray_t;
  typedef boost::multi_array<cellid_t,2>   cellpair_array_t;
  typedef boost::multi_array<eCellFlags,2> cellflag_array_t;

private:

  rect_t  m_rect;
  rect_t  m_ext_rect;

  varray_t         m_vertex_fns; // defined on the vertices of bounding rect
  cellpair_array_t m_cell_pairs;
  cellflag_array_t m_cell_flags;
  cellid_list_t    m_critical_cells;

  mscomplex_t      m_msgraph;

public:

  // initialization of the dataset

  GridDataset(const rect_t &r,
              const rect_t &e);

  void set_datarow(const double *, uint rownum);

  cell_fn_t get_cell_fn(cellid_t c) const;

  void set_cell_fn(cellid_t c,cell_fn_t f);

  static bool isPoint(cellid_t c);

  // actual algorithm work
public:

  void  assignGradients();

  uint  getNumCriticalPoints();

  void  computeDiscs();

  // dataset interface
public:

  virtual cellid_t   getCellPairId ( cellid_t ) const;

  virtual bool   ptLt ( cellid_t ,cellid_t  ) const;

  virtual uint   getCellPoints ( cellid_t ,cellid_t  * ) const;

  virtual uint   getCellFacets ( cellid_t ,cellid_t * ) const;

  virtual uint   getCellCofacets ( cellid_t ,cellid_t * ) const;

  virtual bool   isPairOrientationCorrect ( cellid_t c, cellid_t p) const;

  virtual bool   isCellMarked ( cellid_t c ) const;

  virtual bool   isCellCritical ( cellid_t c) const;

  virtual bool   isCellPaired( cellid_t c) const;

  virtual void   pairCells ( cellid_t c,cellid_t p);

  virtual void   markCellCritical ( cellid_t c );

  inline virtual uint   getCellDim ( cellid_t c ) const;

  virtual uint   getMaxCellDim() const;

  virtual bool   isTrueBoundryCell ( cellid_t c ) const;

  virtual bool   isFakeBoundryCell ( cellid_t c ) const;

  virtual bool   isCellExterior ( cellid_t c ) const;

  virtual void   connectCps ( cellid_t c1, cellid_t c2) ;

  virtual std::string  getCellFunctionDescription ( cellid_t pt ) const;

  virtual std::string getCellDescription ( cellid_t cellid ) const;

  // misc functions
public:
  inline const mscomplex_t & get_ms_complex()
  {
    return m_msgraph;
  }

  inline static uint s_getCellDim ( cellid_t c )
  {
    return ((c[0]&0x01)+(c[1]&0x01));
  }

  inline rect_t get_rect()
  {
    return m_rect;
  }

  inline rect_t get_ext_rect()
  {
    return m_ext_rect;
  }

  // dataset renderable interface
public:
  void getCellCoord(cellid_t c,double &x,double &y,double &z);

};

inline uint GridDataset::getCellDim ( cellid_t c ) const
{
  return s_getCellDim(c);
}




#endif
