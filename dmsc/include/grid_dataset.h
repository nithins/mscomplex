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

#include <boost/multi_array.hpp>

#include <grid_mscomplex.h>

#include <CL/cl.h>



class GridDataset:public grid_types_t
{

public:

  enum eCellFlags
  {
    CELLFLAG_UNKNOWN = 0,
    CELLFLAG_PAIRED  = 1,
    CELLFLAG_CRITCAL = 2,
  };

  typedef GridMSComplex                     mscomplex_t;
  typedef MSComplex<grid_types_t::cellid_t>::critical_point                 critpt_t;
  typedef MSComplex<grid_types_t::cellid_t>::critical_point::connection_t   critpt_conn_t;
  typedef MSComplex<grid_types_t::cellid_t>::critical_point::disc_t         critpt_disc_t;
  typedef int8_t                            cell_flag_t;
  typedef boost::multi_array<cell_fn_t,2>   varray_t;
  typedef boost::multi_array<cellid_t,2>    cellpair_array_t;
  typedef boost::multi_array<cell_flag_t,2> cellflag_array_t;

  typedef boost::multi_array_ref<cell_fn_t,2>   varray_ref_t;

public:

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


  rect_t             m_rect;
  rect_t             m_ext_rect;

  varray_ref_t      *m_vert_fns_ref;

  cellpair_array_t  *m_cell_pairs;
  cellpair_array_t  *m_cell_own;
  cellflag_array_t  *m_cell_flags;
  cellid_list_t      m_critical_cells;

  critpt_idx_list_t  m_saddle_incidence_idx_offset;
  critpt_idx_list_t  m_saddle_incidence_idx;

  cl_mem             m_cell_pair_img;
  cl_mem             m_cell_flag_img;
  cl_mem             m_critical_cells_buf;
  cl_mem             m_cell_own_img;

  pt_comp_t          m_ptcomp;

public:

  // initialization of the dataset

  GridDataset ( const rect_t &r,const rect_t &e );

  GridDataset ();

  ~GridDataset ();

  void  init(cell_fn_t * pData);

  void  set_cell_fn ( cellid_t c,cell_fn_t f );

  void  create_pair_flag_imgs_ocl();

  void  clear_buffers_ocl();

  void  clear();

  // actual algorithm work
public:

  void  work();

  void  writeout_connectivity(mscomplex_t *msgraph);

  void  assignGradients();

  void  collateCriticalPoints();

  void  assignCellOwnerExtrema();



  void  work_ocl(bool collect_cps = true);

  void  writeout_connectivity_ocl(mscomplex_t *msgraph);

  void  assignGradients_ocl(cl_command_queue &commands);

  void  read_pair_img_ocl(cl_command_queue &commands);

  void  read_flag_img_ocl(cl_command_queue &commands);

  void  read_own_img_ocl(cl_command_queue &commands);

  void  collateCritcalPoints_ocl(cl_command_queue &commands);

  int   assignCellOwnerExtrema_ocl(cl_command_queue &commands);

  void  collect_saddle_conn_ocl(cl_command_queue &commands);



  int   postMergeFillDiscs(mscomplex_t *msgraph);




  // dataset interface
public:

  cellid_t   getCellPairId ( cellid_t ) const;

  inline bool   ptLt ( cellid_t c1,cellid_t c2) const
  {
    double f1 = (*m_vert_fns_ref)[c1[0]>>1][c1[1]>>1];
    double f2 = (*m_vert_fns_ref)[c2[0]>>1][c2[1]>>1];

    if (f1 != f2)
      return f1 < f2;

    return c1<c2;
  }

  bool   compareCells( cellid_t ,cellid_t ) const;

  uint   getCellPoints ( cellid_t ,cellid_t  * ) const;

  uint   getCellFacets ( cellid_t ,cellid_t * ) const;

  inline uint   getCellIncCells( cellid_t ,cellid_t * ) const;

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

  void log_flags();

  void log_pairs();

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

namespace boost
{
  namespace serialization
  {
    template<class Archive>
    void serialize(Archive & ar, GridDataset & d, const unsigned int );

  } // namespace serialization
}
#endif
