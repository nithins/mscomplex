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

#include <fstream>
#include <string>
#include <iostream>

#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/regex.hpp>

#include <cpputils.h>
#include <ply.h>

#include <tri_datamanager.h>
#include <tri_dataset.h>

using namespace std;

string fnName ( "x" );

TriDataManager::TriDataManager ( std::string cmdline ) :
    m_flip_triangles ( false ), m_max_levels ( 2 ), m_bShowCriticalPointLabels ( false )
{
  parseCommandLine ( cmdline );

  setupTriangulation();

  work();
}

void
TriDataManager::work()
{

  uint i = 0;

  for ( tree<_data_piece*>::iterator dp_it = m_dpTree.begin(); m_dpTree.is_valid (
          dp_it ); ++dp_it )
  {
    _LOG ( "working piece no::" << i++ );

    _data_piece * dp = *dp_it;

    workPiece ( dp );
  }
}

void
TriDataManager::workPiece ( _data_piece * dp )
{
  /*****************************SETUP THE TRIANGULATION*******************************************/

  dp->tds.setNumVerts ( dp->num_vertices );
  dp->tds.setNumTris ( dp->num_triangles );
  dp->tds.setNumExtVert ( dp->num_ext_vertices );
  dp->tds.setNumExtTri ( dp->num_ext_triangles );

  for ( uint i = 0; i < dp->num_vertices; i++ )
    dp->tds.setVert ( i, dp->vertices[i].fn, dp->vertices[i].index,
                      dp->vertices[i].x, dp->vertices[i].y, dp->vertices[i].z );

  dp->tds.startAddingTris();

  for ( uint i = 0; i < dp->num_triangles; i++ )
    dp->tds.addTri ( dp->triangles[i].v, i, dp->triangles[i].index );

  dp->tds.endAddingTris();

  /*****************************DO WORK*************************************************************/

  dp->tds.assignGradients();

  dp->tds.computeDiscs ();

  /*****************************PUT AWAY RENDER INFO*************************************************/

  dp->ren_surf = dp->tds.createSurfaceRenderer();

  dp->ren_grad0 = dp->tds.createGradientRenderer ( 0 );

  dp->ren_grad1 = dp->tds.createGradientRenderer ( 1 );

  dp->ren_crit = dp->tds.createCriticalPointRenderer();

  dp->ren_unpairedcells = dp->tds.createUnpairedCellsRenderer();

  dp->ren_exteriorsurf = dp->tds.createExteriorSurfaceRenderer();

  dp->ren_msgraph     = dp->tds.createMsGraphRenderer();

  dp->ren_msquads     = dp->tds.createMsQuadsRenderer();

  dp->ren_cancelpairs = dp->tds.createCancellablePairsRenderer();

}

void
TriDataManager::renderPiece ( _data_piece * dp ) const
{

  glPushMatrix();
  glPushAttrib ( GL_ENABLE_BIT );

  if ( dp->m_bShowSurface && true )
  {
    glColor3f ( 1.0, 0.95, 1.0 );
    dp->ren_surf->render_triangles();
  }

  if ( dp->m_bShowExterior && true )
  {
    glColor3f ( 1.0, 0.2, 0.2 );
    dp->ren_exteriorsurf->render_triangles();
  }

  if ( dp->m_bShowGrad && true )
  {
    glColor3f ( 0.7, 0.7, 0.2 );
    dp->ren_grad0->render_triangles();
    glColor3f ( 0.7, 0.3, 0.7 );
    dp->ren_grad1->render_triangles();
  }

  if ( dp->m_bShowCancelablePairs && true )
  {
    dp->ren_cancelpairs->render_triangles();
  }

  glDisable ( GL_LIGHTING );

  glPointSize ( 4.0 );

  if ( dp->m_bShowCps && true )
  {
    dp->ren_crit->render_points();

    if ( m_bShowCriticalPointLabels )
    {
      dp->ren_crit->render_text();
    }
  }

  glTranslatef ( 0.0, 0.0002, 0.0 );

  if ( dp->m_bShowMsGraph && true )
  {
    glColor3f ( 0.0, 1.0, 1.0 );
    dp->ren_msgraph->render_points();
  }

  if ( dp->m_bShowMsQuads && true )
  {
    dp->ren_msquads->render_triangles();
  }

  if ( true )
  {
    dp->ren_unpairedcells->render_points();
  }

  glPopAttrib();

  glPopMatrix();
}

int
TriDataManager::Render() const
{

  for ( tree<_data_piece*>::iterator dp_it = m_dpTree.begin(); m_dpTree.is_valid (
          dp_it ); ++dp_it )
  {
    _data_piece * dp = *dp_it;

    renderPiece ( dp );
  }

  return 0;
}

bool
compare_vertices_by_component ( TriDataManager::_vertex *vertices,
                                uint vert_comp, uint v1_ind, uint v2_ind )
{
  switch ( vert_comp )
  {

    case 0:
      return vertices[v1_ind].x < vertices[v2_ind].x;

    case 1:
      return vertices[v1_ind].y < vertices[v2_ind].y;

    case 2:
      return vertices[v1_ind].z < vertices[v2_ind].z;

    default:
      _ERROR ( "no more comps to compare" );
      return false;
  };
}

void
TriDataManager::setupTriangulation()
{

  using boost::bind;
  using std::max;

  uint num_vertices, num_triangles;
  _vertex *vertices;
  _triangle *triangles;

  switch ( m_filetype )
  {

    case FILE_PLY:
      readPlyFileData ( vertices, triangles, num_vertices, num_triangles );
      break;

    case FILE_TRI:
      readTriFileData ( vertices, triangles, num_vertices, num_triangles );
      break;

    default:
      _ERROR ( "unknown file type.. aborting" );
      exit ( 0 );
      break;
  };

  double x_max =
    vertices[*max_element ( uint_iterator ( 0 ), uint_iterator ( num_vertices ),
                            bind ( compare_vertices_by_component, vertices, 0, _1, _2 ) ) ].x;

  double y_max =
    vertices[*max_element ( uint_iterator ( 0 ), uint_iterator ( num_vertices ),
                            bind ( compare_vertices_by_component, vertices, 1, _1, _2 ) ) ].y;

  double z_max =
    vertices[*max_element ( uint_iterator ( 0 ), uint_iterator ( num_vertices ),
                            bind ( compare_vertices_by_component, vertices, 2, _1, _2 ) ) ].z;

  double x_min =
    vertices[*max_element ( uint_iterator ( 0 ), uint_iterator ( num_vertices ),
                            bind ( compare_vertices_by_component, vertices, 0, _2, _1 ) ) ].x;

  double y_min =
    vertices[*max_element ( uint_iterator ( 0 ), uint_iterator ( num_vertices ),
                            bind ( compare_vertices_by_component, vertices, 1, _2, _1 ) ) ].y;

  double z_min =
    vertices[*max_element ( uint_iterator ( 0 ), uint_iterator ( num_vertices ),
                            bind ( compare_vertices_by_component, vertices, 2, _2, _1 ) ) ].z;

  double scale_factor = max ( ( x_max - x_min ), max ( ( y_max - y_min ), ( z_max
                              - z_min ) ) );

  for ( uint i = 0; i < num_vertices; i++ )
  {
    vertices[i].x = 2.0 * ( vertices[i].x - x_min ) / scale_factor - 1.0;
    vertices[i].y = 2.0 * ( vertices[i].y - y_min ) / scale_factor - 1.0;
    vertices[i].z = 2.0 * ( vertices[i].z - z_min ) / scale_factor - 1.0;
  }

  _data_piece *dpRoot;

  dpRoot = new _data_piece;
  dpRoot->level = 0;
  dpRoot->vertices = vertices;
  dpRoot->triangles = triangles;
  dpRoot->num_vertices = num_vertices;
  dpRoot->num_triangles = num_triangles;
  dpRoot->num_ext_vertices = 0;
  dpRoot->num_ext_triangles = 0;

  m_dpTree.set_head ( dpRoot );

  for ( uint level = 0; level < m_max_levels; level++ )
  {
    for ( tree<_data_piece*>::fixed_depth_iterator dp_it = m_dpTree.begin_fixed (
            m_dpTree.begin(), level ); m_dpTree.is_valid ( dp_it ); ++dp_it )
    {
      _data_piece *dp = *dp_it;

      _data_piece *ch_1, *ch_2;

      splitSet ( dp, level % 3, ch_1, ch_2 );

      m_dpTree.append_child ( dp_it, ch_1 );
      m_dpTree.append_child ( dp_it, ch_2 );
    }
  }

  /******CHECK THAT EACH TRIANGLE BELONGS TO ONE AND ONLY ONE SET ****************/

  _data_piece **tri_dp_no = new _data_piece*[num_triangles];

  for ( uint i = 0 ; i < num_triangles ; i++ )
    tri_dp_no[i] = NULL;


  for ( tree<_data_piece*>::iterator dp_it = m_dpTree.begin(); m_dpTree.is_valid (
          dp_it ); ++dp_it )
  {
    _data_piece * dp = *dp_it;

    for ( uint i = 0 ; i < dp->num_triangles - dp->num_ext_triangles;i++ )
    {
      if ( tri_dp_no[dp->triangles[i].index] == NULL )
      {
        tri_dp_no[dp->triangles[i].index] = dp;
      }
      else
      {
        _LOG ( "Triangle belonging to 2 dps found" );
      }
    }
  }

  for ( uint i = 0 ; i < num_triangles ; i++ )
  {
    if ( tri_dp_no[i] == NULL )
    {
      _LOG ( "Triangle belonging to no set found" );
    }
  }



}

void
TriDataManager::splitSet ( _data_piece *set0, uint compno, _data_piece *&set1,
                           _data_piece *&set2 )
{

  _vertex * vertices = set0->vertices;
  _triangle * triangles = set0->triangles;
  uint num_vertices = set0->num_vertices;
  uint num_triangles = set0->num_triangles;
  uint num_ext_triangles = set0->num_ext_triangles;

  set1 = new _data_piece;
  set1->level = set0->level + 1;

  set2 = new _data_piece;
  set2->level = set0->level + 1;

  uint *vertex_indices = new uint[num_vertices];

  generate ( vertex_indices, vertex_indices + num_vertices,
             seq_num_gen_ftor<uint> () );

  nth_element ( vertex_indices, vertex_indices + num_vertices / 2, vertex_indices
                + num_vertices, boost::bind ( compare_vertices_by_component, vertices,
                                              compno, _1, _2 ) );

  // 1: set1 2: set2 3:set1 ext of set2 4: set2 ext of set 1
  uint *vertex_setno = new uint[num_vertices];

  for ( uint i = 0; i < num_vertices / 2; i++ )
    vertex_setno[vertex_indices[i]] = 1;

  for ( uint i = num_vertices / 2; i < num_vertices; i++ )
    vertex_setno[vertex_indices[i]] = 2;

  // 0: shared 1: set1 2: set2
  uint *triangle_setno = new uint[num_triangles];

  for ( uint i = 0; i < num_triangles; i++ )
  {
    if ( vertex_setno[triangles[i].v[0]] == vertex_setno[triangles[i].v[1]]
         && vertex_setno[triangles[i].v[1]] == vertex_setno[triangles[i].v[2]] )
    {
      triangle_setno[i] = vertex_setno[triangles[i].v[0]];
    }
    else
    {
      triangle_setno[i] = 0;
    }
  }

  for ( uint i = 0; i < num_triangles; i++ )
  {
    if ( triangle_setno[i] == 0 )
    {
      if ( vertex_setno[triangles[i].v[0]] < 3 )
        vertex_setno[triangles[i].v[0]] += 2;

      if ( vertex_setno[triangles[i].v[1]] < 3 )
        vertex_setno[triangles[i].v[1]] += 2;

      if ( vertex_setno[triangles[i].v[2]] < 3 )
        vertex_setno[triangles[i].v[2]] += 2;
    }
  }

  for ( uint i = 0; i < num_triangles; i++ )
  {
    if ( triangle_setno[i] != 0 )
    {
      for ( int j = 0; j < 3; j++ )
      {
        if ( vertex_setno[triangles[i].v[j]] == 3 && triangle_setno[i] == 1 )
        {
          triangle_setno[i] = 3;
        }

        if ( vertex_setno[triangles[i].v[j]] == 4 && triangle_setno[i] == 2 )
        {
          triangle_setno[i] = 4;
        }
      }
    }
  }

  set0->num_triangles = 0;

  set1->num_triangles = 0;
  set2->num_triangles = 0;

  set0->num_ext_triangles = 0;
  set1->num_ext_triangles = 0;
  set2->num_ext_triangles = 0;

  for ( uint i = 0; i < num_triangles - num_ext_triangles; i++ )
  {
    switch ( triangle_setno[i] )
    {

      case 0:
        set1->num_ext_triangles++;
        set2->num_ext_triangles++;
        set0->num_triangles++;
        break;

      case 1:
        set1->num_triangles++;
        break;

      case 2:
        set2->num_triangles++;
        break;

      case 3:
        set1->num_triangles++;
        set0->num_ext_triangles++;
        break;

      case 4:
        set2->num_triangles++;
        set0->num_ext_triangles++;
        break;
    };
  }

  for ( uint i = num_triangles - num_ext_triangles; i < num_triangles; i++ )
  {
    switch ( triangle_setno[i] )
    {

      case 0:
        set1->num_ext_triangles++;
        set2->num_ext_triangles++;
        set0->num_ext_triangles++;
        break;

      case 1:
        set1->num_ext_triangles++;
        break;

      case 2:
        set2->num_ext_triangles++;
        break;

      case 3:
        set1->num_ext_triangles++;
        set0->num_ext_triangles++;
        break;

      case 4:
        set2->num_ext_triangles++;
        set0->num_ext_triangles++;
        break;
    };
  }

  set0->num_triangles += set0->num_ext_triangles;

  set1->num_triangles += set1->num_ext_triangles;
  set2->num_triangles += set2->num_ext_triangles;

  set0->triangles = new _triangle[set0->num_triangles];
  set1->triangles = new _triangle[set1->num_triangles];
  set2->triangles = new _triangle[set2->num_triangles];

  uint set0_triangle_no = 0;
  uint set1_triangle_no = 0;
  uint set2_triangle_no = 0;

  // this loop must first run from [0, (num_triangles-num_ext_triangles) ) and then from
  // ((num_triangles-num_ext_triangles),num_triangles] to capture the external triangles later
  // but thats the same thing as running from [0,num_triangles)

  for ( uint i = 0; i < num_triangles; i++ )
  {
    switch ( triangle_setno[i] )
    {

      case 0:
        set0->triangles[set0_triangle_no++] = triangles[i];
        break;

      case 1:

      case 3:
        set1->triangles[set1_triangle_no++] = triangles[i];
        break;

      case 2:

      case 4:
        set2->triangles[set2_triangle_no++] = triangles[i];
        break;
    };
  }

  for ( uint i = 0; i < num_triangles; i++ )
  {
    switch ( triangle_setno[i] )
    {

      case 0:
        set1->triangles[set1_triangle_no++] = triangles[i];
        set2->triangles[set2_triangle_no++] = triangles[i];
        break;

      case 3:

      case 4:
        set0->triangles[set0_triangle_no++] = triangles[i];
        break;
    }
  }

  _data_piece *dps[] =

    { set0, set1, set2 };

  for ( uint setno = 0; setno < 3; setno++ )
  {
    _data_piece *dp = dps[setno];

    std::map<uint, uint> vertices_remapping;

    uint vertex_no = 0;

    for ( uint i = 0; i < dp->num_triangles - dp->num_ext_triangles; i++ )
      for ( uint j = 0; j < 3; j++ )
        if ( vertices_remapping.find ( dp->triangles[i].v[j] )
             == vertices_remapping.end() )
          vertices_remapping[dp->triangles[i].v[j]] = vertex_no++;

    dp->num_vertices = vertex_no;

    for ( uint i = dp->num_triangles - dp->num_ext_triangles; i
          < dp->num_triangles; i++ )
      for ( uint j = 0; j < 3; j++ )
        if ( vertices_remapping.find ( dp->triangles[i].v[j] )
             == vertices_remapping.end() )
          vertices_remapping[dp->triangles[i].v[j]] = vertex_no++;

    dp->num_ext_vertices = vertex_no - dp->num_vertices;

    dp->num_vertices = vertex_no;

    dp->vertices = new _vertex[dp->num_vertices];

    for ( std::map<uint, uint>::iterator vert_map_it =
            vertices_remapping.begin(); vert_map_it != vertices_remapping.end(); ++vert_map_it )
      dp->vertices[vert_map_it->second] = vertices[vert_map_it->first];

    for ( uint i = 0; i < dp->num_triangles; i++ )
      for ( uint j = 0; j < 3; j++ )
        dp->triangles[i].v[j] = vertices_remapping[dp->triangles[i].v[j]];

    vertices_remapping.clear();
  }

  delete[] vertices;

  delete[] triangles;
  delete[] vertex_setno;
  delete[] triangle_setno;
  delete[] vertex_indices;
}

TriDataManager::_data_piece::_data_piece()
{
  m_bShowCancelablePairs = false;
  m_bShowCps = false;
  m_bShowGrad = false;
  m_bShowMsGraph = false;
  m_bShowMsQuads = false;
  m_bShowSurface = false;
  m_bShowExterior = false;
}

void
TriDataManager::parseCommandLine ( std::string cmdline )
{
  static const boost::regex plyfile_re ( "(-ply ([[:alnum:]\\./_]+))" );
  static const std::string plyfile_replace ( "(())" );

  static const boost::regex trifile_re (
    "(-tri ([[:alnum:]\\./_]+) ([[:alnum:]\\./_]+) ([[:digit:]]+))" );
  static const std::string trifile_replace ( "(()()())" );

  static const boost::regex ft_re ( "(-ft)" );
  static const std::string ft_replace ( "()" );

  static const boost::regex max_levels_re ( "(--max-levels ([[:digit:]]+))" );
  static const std::string max_levels_replace ( "(())" );


  _LOG ( "---------------------------------" );
  _LOG ( "Command Line       = " << cmdline );
  _LOG ( "---------------------------------" );

  boost::smatch matches;

  if ( regex_search ( cmdline, matches, plyfile_re ) )
  {
    m_plyfilename.assign ( matches[2].first, matches[2].second );

    m_filetype = FILE_PLY;

    cmdline = regex_replace ( cmdline, plyfile_re, plyfile_replace,
                              boost::match_default | boost::format_all );
  }
  else
    if ( regex_search ( cmdline, matches, trifile_re ) )
    {
      m_filetype = FILE_TRI;

      m_trifilename.assign ( matches[2].first, matches[2].second );
      m_binfilename.assign ( matches[3].first, matches[3].second );

      m_fn_component_no = atoi (
                            string ( matches[4].first, matches[4].second ).c_str() );

      cmdline = regex_replace ( cmdline, trifile_re, trifile_replace,
                                boost::match_default | boost::format_all );
    }
    else
    {
      _ERROR ( "Fatal No File name specified" );
      exit ( -1 );
    }

  if ( regex_search ( cmdline, matches, ft_re ) )
  {
    m_flip_triangles = true;

    cmdline = regex_replace ( cmdline, ft_re, ft_replace, boost::match_default
                              | boost::format_all );
  }

  if ( regex_search ( cmdline, matches, max_levels_re ) )
  {
    m_max_levels = atoi ( string ( matches[2].first, matches[2].second ).c_str() );

    cmdline = regex_replace ( cmdline, max_levels_re, max_levels_replace, boost::match_default
                              | boost::format_all );
  }

  _LOG ( "=================================" );

  _LOG ( "    Command Line Data            " );
  _LOG ( "---------------------------------" );
  _LOG ( "Flip triangles     = " << m_flip_triangles );
  _LOG ( "Max Levels         = " << m_max_levels );

  switch ( m_filetype )
  {

    case FILE_PLY:
      _LOG ( "Filename           = " << m_plyfilename );
      break;

    case FILE_TRI:
      _LOG ( "Tri Filename       = " << m_trifilename );
      _LOG ( "Bin Filename       = " << m_binfilename );
      break;

    default:
      break;
  };

  _LOG ( "=================================" );
}

void
TriDataManager::readTriFileData ( _vertex *&vertices, _triangle *&triangles,
                                  uint &num_vertices, uint &num_triangles )
{

  /****************READ TRIANGULATION DATA********************************************/
  fstream trifile ( m_trifilename.c_str(), fstream::in );

  trifile >> num_vertices >> num_triangles;

  vertices = new _vertex[num_vertices];

  for ( uint i = 0; i < num_vertices; i++ )
  {
    trifile >> vertices[i].x >> vertices[i].y >> vertices[i].z;
    vertices[i].index = i;
  }

  triangles = new _triangle[num_triangles];

  for ( uint i = 0; i < num_triangles; i++ )
  {
    if ( m_flip_triangles )
      trifile >> triangles[i].v[0] >> triangles[i].v[2] >> triangles[i].v[1];
    else
      trifile >> triangles[i].v[0] >> triangles[i].v[1] >> triangles[i].v[2];

    triangles[i].index = i;
  }

  trifile.close();

  _LOG ( "=================================" );
  _LOG ( "    Triangulation File Info      " );
  _LOG ( "---------------------------------" );
  _LOG ( "Filename           = " << m_trifilename );
  _LOG ( "Num Verts          = " << num_vertices );
  _LOG ( "Num Tris           = " << num_triangles );
  _LOG ( "=================================" );

  /**************READ THE FUNCTION FILE ***********************************************/

  const uint fnname_max_size = 32;// arbit???

  fstream fnfile ( m_binfilename.c_str(), fstream::in | fstream::binary );

  int num_vertices_fn, num_components_fn;

  fnfile.read ( reinterpret_cast<char *> ( &num_vertices_fn ), sizeof ( int ) );
  fnfile.read ( reinterpret_cast<char *> ( &num_components_fn ), sizeof ( int ) );

  char *fnnames = new char[num_components_fn * fnname_max_size];

  for ( uint i = 0; i < ( uint ) num_components_fn; i++ )
    fnfile.read ( fnnames + i * fnname_max_size, fnname_max_size );

  fnfile.seekg ( sizeof ( float ) * m_fn_component_no, ios::cur );

  for ( uint i = 0; i < ( uint ) num_vertices_fn; i++ )
  {
    fnfile.read ( reinterpret_cast<char *> ( &vertices[i].fn ), sizeof ( float ) );
    fnfile.seekg ( sizeof ( float ) * ( num_components_fn - 1 ), ios::cur );
  }

  fnfile.close();

  _LOG ( "=================================" );
  _LOG ( "    Function File Info           " );
  _LOG ( "---------------------------------" );
  _LOG ( "Filename           = " << m_binfilename );
  _LOG ( "Num Verts          = " << num_vertices_fn );
  _LOG ( "Num Components     = " << num_components_fn );
  _LOG ( "---------------------------------" );
  _LOG ( "    Component Names              " );
  _LOG ( "---------------------------------" );

  for ( uint i = 0; i < ( uint ) num_components_fn; i++ )
    _LOG ( fnnames + i*fnname_max_size );

  _LOG ( "---------------------------------" );

  _LOG ( "Comp no used       = " << m_fn_component_no );

  _LOG ( "=================================" );

}

struct _face
{
  unsigned char nverts; /* number of vertex indices in list */
  unsigned int *verts; /* vertex index list */
};

void
TriDataManager::readPlyFileData ( _vertex *&vertices, _triangle *&triangles,
                                  uint &num_vertices, uint &num_triangles )
{
  static PlyProperty vert_props[] =
  {
    /* list of property information for a vertex */
    { "x", PLY_FLOAT, PLY_DOUBLE, offsetof ( _vertex, x ), 0, 0, 0, 0 },
    { "y", PLY_FLOAT, PLY_DOUBLE, offsetof ( _vertex, y ), 0, 0, 0, 0 },
    { "z", PLY_FLOAT, PLY_DOUBLE, offsetof ( _vertex, z ), 0, 0, 0, 0 },
    { fnName.c_str(), PLY_FLOAT, PLY_DOUBLE, offsetof ( _vertex, fn ), 0, 0,
      0, 0 },
  };

  static PlyProperty face_props[] = /* list of property information for a vertex */
  {
    { "vertex_indices", PLY_INT, PLY_UINT, offsetof ( _face, verts ), 1,
      PLY_UCHAR, PLY_UCHAR, offsetof ( _face, nverts ) },
  };

  int elemCount;
  char **elemNames;
  int fileType;
  float fileVersion;

  PlyFile *plyfile = ply_open_for_reading ( m_plyfilename.c_str(), &elemCount,
                     &elemNames, &fileType, &fileVersion );

  if ( plyfile == NULL )
  {
    _ERROR ( "Could not open file ... quitting" );
    return;
  }

  int num_comps_per_vert, num_comps_per_face;

  int _num_vertices;
  int _num_triangles;

  ply_get_element_description ( plyfile, "vertex", &_num_vertices,
                                &num_comps_per_vert );
  ply_get_element_description ( plyfile, "face", &_num_triangles,
                                &num_comps_per_face );

  num_vertices = _num_vertices;
  num_triangles = _num_triangles;

  vertices = new _vertex[num_vertices];
  triangles = new _triangle[num_triangles];

  ply_get_property ( plyfile, "vertex", &vert_props[0] );
  ply_get_property ( plyfile, "vertex", &vert_props[1] );
  ply_get_property ( plyfile, "vertex", &vert_props[2] );

  if ( fnName != "x" && fnName != "y" && fnName != "z" )
  {
    ply_get_property ( plyfile, "vertex", &vert_props[3] );
  }

  for ( uint i = 0; i < num_vertices; i++ )
  {

    ply_get_element ( plyfile, ( void * ) ( vertices + i ) );

    if ( fnName == "x" )
      vertices[i].fn = vertices[i].x;

    if ( fnName == "y" )
      vertices[i].fn = vertices[i].y;

    if ( fnName == "z" )
      vertices[i].fn = vertices[i].z;

    vertices[i].index = i;

  }

  ply_get_property ( plyfile, "face", &face_props[0] );

  uint num_unuseable_faces = 0;

  for ( uint i = 0; i < num_triangles; i++ )
  {
    _face f;

    bool useface = true;

    ply_get_element ( plyfile, ( void * ) &f );

    if ( f.nverts < 3 || f.nverts > 3 )
    {
      _ERROR ( "improper triangulation ..got a simplex with nverts = " << ( int ) f.nverts );

      useface = false;
    }

    for ( uint j = 0; j < 3; j++ )
    {
      if ( f.verts[j] >= num_vertices )
      {
        _ERROR ( "read incorrect face data" );
        useface = false;
      }
    }

    if ( useface )
    {
      triangles[i].v[0] = f.verts[0];
      triangles[i].v[1] = f.verts[1];
      triangles[i].v[2] = f.verts[2];
      triangles[i].index = i;
    }
    else
    {
      num_unuseable_faces++;
    }
  }

  _LOG ( "=================================" );

  _LOG ( "    Ply File Details             " );
  _LOG ( "---------------------------------" );
  _LOG ( "Filename           = " << m_plyfilename );
  _LOG ( "Num Verts          = " << num_vertices );
  _LOG ( "Num Tris           = " << num_triangles );
  _LOG ( "=================================" );
}

