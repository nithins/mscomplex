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

#ifndef __GRID_MSCOMPLEX_H_INCLUDED_
#define __GRID_MSCOMPLEX_H_INCLUDED_

#include <cpputils.h>

#include <discreteMorseDS.h>
#include <rectangle_complex.h>

class GridMSComplex;

class grid_types_t
{
public:
  typedef int16_t                          cell_coord_t;
  typedef float                            cell_fn_t;
  typedef rectangle_complex<cell_coord_t>  rect_cmplx_t;
  typedef rect_cmplx_t::rectangle_def      rect_t;
  typedef rect_cmplx_t::point_def          cellid_t;
  typedef rect_cmplx_t::point_def          rect_point_t;
  typedef rect_cmplx_t::size_def           rect_size_t;
  typedef std::vector<cellid_t>            cellid_list_t;

};


class GridMSComplex:
    public MSComplex<grid_types_t::cellid_t>,public grid_types_t
{
public:
  typedef std::pair<uint,uint>                        crit_idx_pair_t;
  typedef std::vector<crit_idx_pair_t>                crit_idx_pair_list_t;
  typedef GridMSComplex                               mscomplex_t;
  typedef mscomplex_t::critical_point                 critpt_t;
  typedef mscomplex_t::critical_point::connection_t   conn_t;
  typedef conn_t::iterator                            conn_iter_t;
  typedef conn_t::const_iterator                      const_conn_iter_t;
  typedef std::vector<cell_fn_t>                      cp_fn_list_t;

  rect_t        m_rect;
  rect_t        m_ext_rect;
  cp_fn_list_t  m_cp_fns;

  // call these functions only at the highest levels
  void simplify_un_simplify(uint max_cancellations );

  void simplify(crit_idx_pair_list_t &,uint max_cancellations);

  void un_simplify(const crit_idx_pair_list_t &);

  void clear();

  static mscomplex_t * merge_up(const mscomplex_t& msc1,const mscomplex_t& msc2);

  void merge_down(mscomplex_t& msc1,mscomplex_t& msc2);

  GridMSComplex(rect_t r,rect_t e):m_rect(r),m_ext_rect(e)
  {
  }
};


#endif
