/***************************************************************************
 *   Copyright (C) 2009 by Nithin Shivashankar,   *
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

#ifndef DISCRTEMORSEDS_INCLUDED__
#define DISCRTEMORSEDS_INCLUDED__

#include <vector>
#include <map>
#include <set>
#include <iostream>

#include <glutils.h>
#include <quad_edge.h>

enum eGradDirection {GRADIENT_DIR_DOWNWARD,GRADIENT_DIR_UPWARD,GRADIENT_DIR_COUNT};

template <typename id_type>
    class IDiscreteDataset
{
public:
  virtual id_type getCellPairId ( id_type cellid ) const =0;

  virtual bool    ptLt ( id_type cell1,id_type cell2 ) const =0;

  virtual uint    getCellPoints ( id_type cellId,id_type  *points ) const =0;

  virtual uint    getCellFacets ( id_type cellId,id_type *facets ) const =0;

  virtual uint    getCellCofacets ( id_type cellId,id_type *cofacets ) const =0;

  virtual bool    isPairOrientationCorrect ( id_type cellId, id_type pairId ) const =0;

  virtual bool    isCellMarked ( id_type cellId ) const =0;

  virtual bool    isCellCritical ( id_type cellId ) const =0;

  virtual void    pairCells ( id_type cellId1,id_type cellId2 ) =0;

  virtual void    markCellCritical ( id_type cellId ) =0;

  virtual uint    getCellDim ( id_type cellId ) const = 0;

  virtual uint    getMaxCellDim() const = 0;

  virtual bool    isTrueBoundryCell ( id_type cellId ) const =0;

  virtual bool    isFakeBoundryCell ( id_type cellId ) const =0;

  virtual bool    isCellExterior ( id_type cellId ) const =0;

  // some functions to support debugging

  virtual std::string  getCellFunctionDescription ( id_type pt ) const = 0;

  virtual std::string getCellDescription ( uint cellid ) const =0;

};

template <typename id_t>
    class MSComplex
{

public:

  typedef std::map<id_t,uint> id_cp_map_t;

  struct critical_point
  {
    typedef std::multiset<uint> connection_t;

    id_t cellid;

    bool isCancelled;
    bool isOnStrangulationPath;
    bool isBoundryCancelable;

    critical_point()
    {
      isCancelled           = false;
      isOnStrangulationPath = false;
      isBoundryCancelable   = false;
    }

    std::set<id_t> asc_disc;
    std::set<id_t> des_disc;

    connection_t asc;
    connection_t des;

    void print_asc_connections(std::ostream & os,const MSComplex<id_t> &msc)
    {
      os<<"{ ";
      for(connection_t::iterator it = asc.begin(); it != asc.end(); ++it)
      {
        if(msc.m_cps[*it]->isBoundryCancelable)
          os<<"*";
        os<<msc.m_cps[*it]->cellid;
        os<<", ";
      }
      os<<"}";
    }

    void print_des_connections(std::ostream & os,const MSComplex<id_t> &msc)
    {
      os<<"{ ";
      for(connection_t::iterator it = des.begin(); it != des.end(); ++it)
      {
        if(msc.m_cps[*it]->isBoundryCancelable)
          os<<"*";
        os<<msc.m_cps[*it]->cellid;
        os<<", ";
      }
      os<<"}";
    }


  };

  std::map<id_t,uint> m_id_cp_map;

  uint m_cp_count;

  MSComplex()
  {
    m_cp_count = 0;
  }

  typedef std::vector<critical_point *> cp_ptr_list_t;

  cp_ptr_list_t m_cps;


  void print_connections(std::ostream & os)
  {
    for(uint i = 0 ; i < m_cp_count;++i)
    {
//      if(m_cps[i]->isCancelled)
//        continue;

      os<<"des(";
      if(m_cps[i]->isBoundryCancelable)
        os<<"*";
      os<<m_cps[i]->cellid<<") = ";
      m_cps[i]->print_des_connections(os,*this);
      os<<std::endl;

      os<<"asc(";
      if(m_cps[i]->isBoundryCancelable)
        os<<"*";
      os<<m_cps[i]->cellid<<") = ";
      m_cps[i]->print_asc_connections(os,*this);
      os<<std::endl;
      os<<std::endl;

    }
  }
};

template <typename id_type,typename fn_type>
    class IDiscreteDataset_renderable:public IDiscreteDataset<id_type>
{
public:
  virtual void getCellCoords ( id_type ,fn_type &,fn_type &,fn_type & ) const =0;

  virtual ArrayRenderer<id_type,fn_type> * createSurfaceRenderer() = 0;

  virtual ArrayRenderer<id_type,fn_type> * createGradientRenderer ( uint dim ) = 0;

  virtual ArrayRenderer<id_type,fn_type> * createMsGraphRenderer( ) = 0;

  virtual ArrayRenderer<id_type,fn_type> * createMsQuadsRenderer( ) = 0;
};

template<typename id_t>
    struct QuadGenericCell2D
{
  typedef unsigned char uchar;
  typedef unsigned int  uint;

  id_t  _v[4]; // remember that _v must be in reverse sorted order for comparisons to work properly
  uchar _dim;


public:
  bool operator < ( const QuadGenericCell2D & rhs ) const
  {
    return std::lexicographical_compare( _v,_v+numv(),rhs._v,rhs._v+numv() );
  }

  bool operator== ( const QuadGenericCell2D & rhs ) const
  {
    return ( ! ( *this<rhs ) ) && ( ! ( rhs<*this ) );
  }

  bool operator != ( const QuadGenericCell2D & rhs ) const
  {
    return ! ( *this == rhs );
  }

  void operator= ( const QuadGenericCell2D & rhs )
                 {
    _v[0] = rhs._v[0];
    _v[1] = rhs._v[1];
    _v[2] = rhs._v[2];
    _v[3] = rhs._v[3];
    _dim  = rhs._dim;
  }

  uchar dim() const
  {
    return _dim;
  }

  uint numv() const
  {
    return pow ( 2, ( uint ) dim() );
  }

  std::string toString() const
  {
    std::stringstream ss;

    ss<<"[";

    for ( uint i = 0 ;i < numv()-1;i++ )
      ss<<_v[i]<<",";

    ss<<_v[numv()-1];

    ss<<"]";

    return ss.str();
  }




  friend  std::ostream& operator << ( std::ostream& os, const QuadGenericCell2D<id_t>& cell )
  {
    os<<cell.toString();
    return os;
  }

};

template<typename id_t>
    struct TriGenericCell2D
{
  typedef unsigned char uchar;
  typedef unsigned int  uint;

  id_t  _v[3]; // remember that _v must be in reverse sorted order for comparisons to work properly
  uchar _dim;


public:
  bool operator < ( const TriGenericCell2D & rhs ) const
  {
    return std::lexicographical_compare ( _v,_v+numv(),rhs._v,rhs._v+numv() );
  }

  bool operator== ( const TriGenericCell2D & rhs ) const
  {
    return ( ! ( *this<rhs ) ) && ( ! ( rhs<*this ) );
  }

  bool operator != ( const TriGenericCell2D & rhs ) const
  {
    return ! ( *this == rhs );
  }

  void operator= ( const TriGenericCell2D & rhs )
                 {
    _v[0] = rhs._v[0];
    _v[1] = rhs._v[1];
    _v[2] = rhs._v[2];
    _dim  = rhs._dim;
  }

  uchar dim() const
  {
    return _dim;
  }

  uint numv() const
  {
    return _dim+1;
  }

  friend  std::ostream& operator << ( std::ostream& os, const TriGenericCell2D<id_t>& cell )
  {
    os<<"[";

    for ( uint i = 0 ;i < cell.numv()-1;i++ )
      os<<cell._v[i]<<",";

    os<<cell._v[cell.numv()-1];

    os<<"]";

    return os;
  }

};

#endif
