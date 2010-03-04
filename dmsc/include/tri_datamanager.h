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

#ifndef QUAD_DATAMANAGER_H_INCLUDED_
#define QUAD_DATAMANAGER_H_INCLUDED_

#include <fstream>
#include <vector>

#include <tree.h>

#include <model.h>

#include <discreteMorseDS.h>
#include <tri_dataset.h>



class TriDataManager:virtual public IModel
{
  public:

    typedef TriGenericCell2D<uint>      generic_cell_t;
    typedef MSComplex<generic_cell_t>   generic_mscomplex_t;

    struct _vertex
    {
      double x,y,z,fn;
      uint index;
    };

    struct _triangle
    {
      uint v[3];
      uint index;
    };

    struct _data_piece
    {
      _vertex   * vertices;
      _triangle * triangles;
      uint        num_vertices;
      uint        num_triangles;
      uint        num_ext_vertices;
      uint        num_ext_triangles;

      TriDataset  tds;

      uint        level;

      bool        m_bShowSurface;
      bool        m_bShowCps;
      bool        m_bShowMsGraph;
      bool        m_bShowMsQuads;
      bool        m_bShowGrad;
      bool        m_bShowCancelablePairs;
      bool        m_bShowExterior;

      ArrayRenderer<uint,double> *ren_surf;
      ArrayRenderer<uint,double> *ren_grad0;
      ArrayRenderer<uint,double> *ren_grad1;
      ArrayRenderer<uint,double> *ren_crit;
      ArrayRenderer<uint,double> *ren_msgraph;
      ArrayRenderer<uint,double> *ren_msquads;
      ArrayRenderer<uint,double> *ren_cancelpairs;
      ArrayRenderer<uint,double> *ren_unpairedcells;
      ArrayRenderer<uint,double> *ren_exteriorsurf;

      _data_piece();
    };

    enum eFileType {FILE_PLY,FILE_TRI};

    bool                m_flip_triangles;

    eFileType           m_filetype;

    std::string         m_plyfilename;

    std::string         m_trifilename;
    std::string         m_binfilename;
    uint                m_fn_component_no;

    uint                m_max_levels;

    tree<_data_piece *> m_dpTree;

    bool                m_bShowCriticalPointLabels;
    bool                m_bShowCriticalPoints;
    bool                m_bShowMsGraph;

  public:

    TriDataManager ( std::string cmdline );

    void parseCommandLine ( std::string cmdline );

    void readTriFileData ( _vertex *&,_triangle *&,uint &,uint & );

    void readPlyFileData ( _vertex *&,_triangle *&,uint &,uint & );

    void setupTriangulation();

    void splitSet ( _data_piece *dp,uint compno,_data_piece *&set1,_data_piece *&set2 );

    void workPiece ( _data_piece * );

    void renderPiece ( _data_piece * ) const ;

    void work();

  public:

    virtual int Render() const;

    virtual std::string Name() const {return std::string("TriMesh");}
};

#endif

