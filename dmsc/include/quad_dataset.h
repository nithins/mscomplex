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


#ifndef __QUAD_DATASET_H_INCLUDED_
#define __QUAD_DATASET_H_INCLUDED_

#include <vector>

#include <discreteMorseDS.h>
#include <quad_edge.h>

struct cell
{
  int   dim;
  cell *pair;
  bool  marked;
  bool  critical;

  cell()
  {
    marked   = false;
    critical = false;
    pair     = NULL;
  }
};



class QuadDataset:public IDiscreteDataset_renderable<uint,double>
{
    typedef MSComplex<uint>           mscomplex_t;

    typedef QuadGenericCell2D<uint>   generic_cell_t;

    typedef MSComplex<generic_cell_t> generic_mscomplex_t;

    typedef std::pair<uint,uint>      ui_pair_t;



  private:

    cell * m_vert_cells;
    cell * m_edge_cells;
    cell * m_quad_cells;

    QuadEdge m_q;

    mscomplex_t m_mscomplex;

    std::vector<uint> m_criticalpoints;

    std::vector<ui_pair_t> m_cancellable_boundry_pairs;

    uint  m_intbnd_vert_ct;
    uint  m_intbnd_edge_ct;
    uint  m_intbnd_quad_ct;

    // functions to setup the quad edge structure
  public:

    void setNumVerts ( const uint & vert_ct );

    void setNumQuads ( const uint &quad_ct );

    void setNumExtVert ( const uint &ct );

    void setNumExtQuad ( const uint &ct );

    void setVert ( const uint &v,const double &fn,const uint& ind,const double& x,const double& y,const double& z );

    void startAddingQuads();

    void addQuad ( const uint *v ,const uint &qno,const uint &ind );

    void endAddingQuads();

    // actual algorithm work
  public:

    void  assignGradients();

    uint  getNumCriticalPoints();

    void  computeDiscs();

    void  getGenericMsComplex ( generic_mscomplex_t *&ms_out );

    void  getBoundryCancellablePairs ( std::vector<std::pair<generic_cell_t,generic_cell_t> > &boundry_cancellable_list );

    // misc utility
  public:

    void  logCell ( cell *cellid );

    cell* getCell ( uint cellid ) const;

    uint  getCellid ( cell * c ) const;

    uint  getPointGenericIndex ( uint point ) const;

    void  destroy();

    void  addCancellablePair ( uint c1,uint c2 );

    void  addCpConnection ( uint c1,uint c2 );

    void  do_gradient_bfs(uint id,eGradDirection dir);

    // dataset interface
  public:

    virtual uint   getCellPairId ( uint cellid ) const;

    virtual bool   ptLt ( uint cellid1,uint cellid2 ) const;

    virtual bool   compareCells( uint c1,uint c2) const;

    virtual uint   getCellPoints ( uint cellid,uint  *points ) const;

    virtual uint   getCellFacets ( uint cellid,uint *facets ) const;

    virtual uint   getCellCofacets ( uint cellid,uint *cofacets ) const;

    virtual bool   isPairOrientationCorrect ( uint cellid, uint pairid ) const;

    virtual bool   isCellMarked ( uint cellid ) const;

    virtual bool   isCellCritical ( uint cellid ) const;

    virtual void   pairCells ( uint cellid1,uint cellid2 );

    virtual void   markCellCritical ( uint cellid );

    virtual uint   getCellDim ( uint cellId ) const;

    virtual uint   getMaxCellDim() const;

    virtual bool   isTrueBoundryCell ( uint cellId ) const;

    virtual bool   isFakeBoundryCell ( uint cellId ) const;

    virtual bool   isCellExterior ( uint cellId ) const;

    // debugging support functions

    virtual std::string getCellFunctionDescription ( uint cellid ) const ;

    virtual std::string getCellDescription ( uint cellid ) const ;

  private:

    struct _vertex
    {
      double x;
      double y;
      double z;

      double fn;
    };

    _vertex * m_vertices;

    uint *m_orig_vert_index;
    uint *m_orig_quad_index;

    // dataset renderable interface
  public:

    virtual void getCellCoords ( uint cellid,double &x,double &y,double &z ) const;

    virtual ArrayRenderer<uint,double> * createSurfaceRenderer();

    virtual ArrayRenderer<uint,double> * createGradientRenderer ( uint dim );

    virtual ArrayRenderer<uint,double> * createMsGraphRenderer();

    virtual ArrayRenderer<uint,double> * createMsQuadsRenderer();

    virtual ArrayRenderer<uint,double> * createCancellablePairsRenderer();
};


#endif
