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

#ifndef DISCRETEMORSEDSRENDERFUNCS_H_INCLUDED_
#define DISCRETEMORSEDSRENDERFUNCS_H_INCLUDED_

#include <discreteMorseDS.h>

#include <algorithm>

template <typename id_t>
    struct createQuadPatches_ftor
{
  typedef typename MSComplex<id_t>::critical_point    cp_t;
  typedef typename std::map<id_t,uint>                id_cp_map_t;
  typedef typename std::vector<uint >                 cp_vec_t;
  typedef typename std::multiset<uint>                cp_adj_t;
  typedef struct   {uint v[4];}                       uint4;
  typedef typename std::vector<id_t>                  idvec_t;

  void operator() ( MSComplex<id_t> *msc,IDiscreteDataset<id_t> *ds,std::vector<std::vector<id_t> *> & quadpatches )
  {

    std::vector<uint> *cp_dim_vec  = new std::vector<uint>[ds->getMaxCellDim() +1];

    for ( uint i = 0 ; i< msc->m_cps.size();i++ )
    {
      uint dim = ds->getCellDim ( msc->m_cps[i]->cellid );
      cp_dim_vec[dim].push_back ( i );
    }


    std::vector<std::pair<uint,uint> > reached_cps
        (
            msc->m_cps.size(),
            std::make_pair ( msc->m_cps.size(),msc->m_cps.size() )
            );

    std::vector<uint4> q;

    // determine reachability from a d+2 dim cell to d cell
    for ( uint d = 0 ; d <= ds->getMaxCellDim()-2;d++ )
    {
      // for each d+2 cell
      for ( cp_vec_t::iterator cp2_it = cp_dim_vec[d+2].begin();cp2_it != cp_dim_vec[d+2].end(); ++cp2_it )
      {
        if(*cp2_it >= msc->m_cps.size())
        {
          _ERROR("invalid cp index ");
          _LOG_VAR(*cp2_it);
          _LOG_VAR(msc->m_cps.size());
        }

        cp_t *cp2    = msc->m_cps[*cp2_it];

        for ( typename cp_adj_t::iterator cp1_it = cp2->des.begin();cp1_it != cp2->des.end(); ++cp1_it )
        {
          cp_t *cp1 = msc->m_cps[ *cp1_it ];

          if(*cp1_it >= msc->m_cps.size())
          {
            _ERROR("invalid cp index ");
            _LOG_VAR(*cp1_it);
            _LOG_VAR(msc->m_cps.size());
          }

          for ( typename cp_adj_t::iterator cp0_it = cp1->des.begin();cp0_it != cp1->des.end(); ++cp0_it )
          {

            if(*cp0_it >= msc->m_cps.size())
            {
              _ERROR("invalid cp index ");
              _LOG_VAR(*cp0_it);
              _LOG_VAR(msc->m_cps.size());
            }

            if ( reached_cps[ *cp0_it].first == *cp2_it )
            {
              uint4 u4 = {{*cp2_it,  *cp1_it , *cp0_it ,reached_cps[  *cp0_it ].second}};
              q.push_back ( u4 );
            }
            else
            {
              reached_cps[ *cp0_it ] = std::make_pair ( *cp2_it, *cp1_it ) ;
            }
          }
        }
      }
    }

    for ( uint i = 0 ; i < q.size() ; i++ )
    {
      cp_t *cp2  = msc->m_cps[q[i].v[0]];
      //cp_t *cp11  &msc->m_cps[q[i].v[1]];
      cp_t *cp0  = msc->m_cps[q[i].v[2]];
      //cp_t *cp12 = &msc->m_cps[q[i].v[3]];

      std::vector<id_t> *patch = new std::vector<id_t>();

      patch->resize ( cp2->des_disc.size() );

      typename idvec_t::iterator it =  std::set_intersection
                                       (
                                           cp2->des_disc.begin(),
                                           cp2->des_disc.end(),
                                           cp0->asc_disc.begin(),
                                           cp0->asc_disc.end(),
                                           patch->begin()
                                           );



      //       DEBUG_LOG_VV ( "==========================================" );
      //       DEBUG_LOG_VV ( "Quad Patch No."<<i<<" Size = "<<patch->size() );
      //       DEBUG_LOG_VV ( "------------------------------------------" );
      //       DEBUG_STMT_VV ( log_range ( patch->begin(),msc->patch->end() ) );
      //       DEBUG_LOG_VV ( "==========================================" );

      quadpatches.push_back ( patch );

    }
  }
};

template <typename id_t>
    void createQuadPatches ( MSComplex<id_t> *msc,IDiscreteDataset<id_t> *ds,std::vector<std::vector<id_t> *> & quadpatches )
{
  createQuadPatches_ftor<id_t>() ( msc,ds,quadpatches );
}

template <typename id_type,typename fn_type,typename point_iter_type,typename face_iter_type>
    ArrayRenderer<id_type,fn_type> * createDatasetSurfaceRenderer
    (
        IDiscreteDataset_renderable<id_type,fn_type> *ds,
        point_iter_type          pbegin,
        point_iter_type          pend,
        face_iter_type           fbegin,
        face_iter_type           fend
        )
{
  ArrayRenderer<id_type,fn_type> * ren = new ArrayRenderer<id_type,fn_type>();

  for ( point_iter_type it = pbegin; it !=pend ;++it )
  {
    fn_type x,y,z;
    ds->getCellCoords ( *it,x,y,z );
    ren->add_vertex ( *it,x,y,z );
  }

  for ( face_iter_type it = fbegin; it !=fend;++it )
  {
    id_type pts[20];

    uint num_pts = ds->getCellPoints ( *it,pts );

    if ( num_pts == 3 )
    {
      ren->add_triangle ( pts[0],pts[1],pts[2] );
    }

    if ( num_pts == 4 )
    {
      ren->add_quad ( pts[0],pts[1],pts[2],pts[3] );
    }
  }
  ren->prepare_for_render();
  ren->sqeeze();

  return ren;
}

template <typename id_type,typename fn_type,typename cell_iter_type>
    ArrayRenderer<id_type,fn_type> * createDatasetGradientRenderer
    (
        IDiscreteDataset_renderable<id_type,fn_type> *ds,
        cell_iter_type cbegin,
        cell_iter_type cend
        )
{
  ArrayRenderer<id_type,fn_type> * ren = new ArrayRenderer<id_type,fn_type>();

  fn_type x,y,z;
  fn_type x_,y_,z_;

  for ( cell_iter_type iter = cbegin;
        iter != cend;++iter )
  {
    if ( ! ds->isCellCritical ( *iter ) && ! ds->isCellExterior ( *iter ) )
    {
      id_type pairid = ds->getCellPairId ( *iter );

      if ( ds->isPairOrientationCorrect ( *iter,pairid ) )
      {
        ds->getCellCoords ( *iter,x,y,z );
        ds->getCellCoords ( pairid,x_,y_,z_ );

        ren->add_vertex ( *iter,x,y,z );
        ren->add_vertex ( pairid,x_,y_,z_ );
        ren->add_arrow ( *iter,pairid );
      }
    }
  }

  ren->prepare_for_render();
  ren->sqeeze();
  return ren;
}

template <typename id_t,typename fn_t>
    struct addArrowToRenderer_ftor
{
  typedef void result_type;

  void operator()
      (
          id_t cellid,
          IDiscreteDataset_renderable<id_t,fn_t> *ds,
          ArrayRenderer<id_t,fn_t> *ren
          )
  {
    id_t pairid = ds->getCellPairId ( cellid );
    if ( ds->isPairOrientationCorrect ( cellid,pairid ) )
    {
      fn_t x,y,z;
      fn_t x_,y_,z_;

      ds->getCellCoords ( cellid,x,y,z );
      ds->getCellCoords ( pairid,x_,y_,z_ );

      ren->add_vertex ( cellid,x,y,z );
      ren->add_vertex ( pairid,x_,y_,z_ );
      ren->add_arrow ( cellid,pairid );
    }
  }
};


template <typename id_t,typename fn_t>
    ArrayRenderer<id_t,fn_t> * createCombinatorialStructureRenderer
    (
        IDiscreteDataset_renderable<id_t,fn_t> *ds,
        MSComplex<id_t> *msc
        )
{
  typedef typename MSComplex<id_t>::critical_point  cp_t;
  typedef typename std::map<id_t,uint>              id_cp_map_t;
  typedef typename std::vector<id_t>                idvec_t;
  typedef typename std::multiset<uint>              adj_t;

  ArrayRenderer<id_t,fn_t> * ren = new ArrayRenderer<id_t,fn_t>();


  for ( uint i = 0 ; i < msc->m_cps.size();i++ )
  {
    cp_t * src_cp = msc->m_cps[i];

    for ( typename adj_t::iterator pvit = src_cp->des.begin();
    pvit != src_cp->des.end();pvit++ )
    {
      cp_t * dest_cp = msc->m_cps[*pvit];

      //       set_intersection_ftor
      //       (
      //         src_cp->des_manifold->begin(),
      //         src_cp->des_manifold->end(),
      //         dest_cp->asc_manifold->begin(),
      //         dest_cp->asc_manifold->end(),
      //         boost::bind ( addArrowToRenderer_ftor<id_t,fn_t>(),_1,ds,ren )
      //       );
    }
  }

  ren->prepare_for_render();
  ren->sqeeze();

  return ren;
}

template <typename id_t,typename fn_t,typename get_cell_coords_ftor_t,typename get_cell_dim_ftor_t>
    void createCritPtRen
    (
        MSComplex<id_t> * msc,
        ArrayRenderer<id_t,fn_t> * &ren,
        get_cell_coords_ftor_t get_cell_coords_ftor,
        get_cell_dim_ftor_t get_cell_dim_ftor
        )
{
  ren = new ArrayRenderer<id_t,fn_t>();

  static double cpcolorset[][3]=
  {
    {1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0},
    {1.0,0.5,0.0},{0.0,1.0,0.5},{0.5,0.0,1.0}
  };

  fn_t x,y,z;

  for ( uint i = 0;i< msc->m_cps.size();i++ )
  {
    id_t cellid = msc->m_cps[i]->cellid;

    uint dim = get_cell_dim_ftor ( cellid );

    if ( msc->m_cps[i]->isCancelled == false )
    {
      std::stringstream label;
      label<<cellid;

      get_cell_coords_ftor ( cellid,x,y,z );
      ren->add_vertex ( cellid,x,y,z );
      ren->add_text ( label.str(),x,y,z );
      ren->set_vertex_color ( cellid,cpcolorset[dim] );
    }

  }

  ren->prepare_for_render();
  ren->sqeeze();
}

template <typename id_t,typename fn_t,typename critpt_iter_t>
    void createCritPtRen
    (
        IDiscreteDataset_renderable<id_t,fn_t> *ds,
        ArrayRenderer<id_t,fn_t> * &ren,
        critpt_iter_t begin,
        critpt_iter_t end
        )
{
  ren = new ArrayRenderer<id_t,fn_t>();

  static double cpcolorset[][3]=
  {
    {1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0},
    {1.0,0.5,0.0},{0.0,1.0,0.5},{0.5,0.0,1.0}
  };

  fn_t x,y,z;

  for ( ; begin != end ; ++begin )
  {
    id_t cellid = *begin;

    uint dim = ds->getCellDim ( cellid );

    std::stringstream label;
    label<<cellid;

    ds->getCellCoords( cellid,x,y,z );
    ren->add_vertex ( cellid,x,y,z );
    ren->add_text ( label.str(),x,y,z );
    ren->set_vertex_color ( cellid,cpcolorset[dim] );

  }

  ren->prepare_for_render();
  ren->sqeeze();
}

template <typename id_t,typename fn_t,typename get_cell_coords_ftor_t>
    void add_point_to_renderer
    (
        id_t cellid,
        ArrayRenderer<id_t,fn_t> *ren,
        get_cell_coords_ftor_t get_cell_coords_ftor

        )
{
  fn_t x,y,z;
  get_cell_coords_ftor ( cellid,x,y,z );
  ren->add_vertex ( cellid,x,y,z );
}


template <typename id_t,typename fn_t,typename get_cell_coords_ftor_t>
    void createCombinatorialStructureRenderer
    (
        MSComplex<id_t> *msc,
        ArrayRenderer<id_t,fn_t> * &ren,
        get_cell_coords_ftor_t get_cell_coords_ftor
        )
{
  typedef typename MSComplex<id_t>::critical_point  cp_t;
  typedef typename std::map<id_t,uint>              id_cp_map_t;
  typedef typename std::vector<id_t>                idvec_t;
  typedef typename std::multiset<uint>              adj_t;

  ren = new ArrayRenderer<id_t,fn_t>();


  //  for ( uint i = 0 ; i < msc->m_cps.size();i++ )
  //  {
  //    cp_t * src_cp = msc->m_cps[i];
  //
  //        for ( typename adj_t::iterator pvit = src_cp->des.begin();
  //              pvit != src_cp->des.end();pvit++ )
  //    {
  //            cp_t * dest_cp = msc->m_cps[*pvit];
  //            set_intersection_ftor
  //            (
  //              src_cp->des_disc.begin(),
  //              src_cp->des_disc.end(),
  //              dest_cp->asc_disc.begin(),
  //              dest_cp->asc_disc.end(),
  //              boost::bind ( add_point_to_renderer<id_t,fn_t,get_cell_coords_ftor_t>,_1,ren,get_cell_coords_ftor )
  //            );
  //      std::for_each
  //          (src_cp->des_disc.begin(),
  //           src_cp->des_disc.end(),
  //           boost::bind ( add_point_to_renderer<id_t,fn_t,get_cell_coords_ftor_t>,_1,ren,get_cell_coords_ftor ));

  //    }
  //  }

  ren->prepare_for_render();
  ren->sqeeze();
}

template <typename id_t,typename fn_t,typename get_cell_coords_ftor_t>
    void create_critpt_disc_renderer
    (
        MSComplex<id_t> *msc,
        ArrayRenderer<id_t,fn_t> * &ren,
        eGradDirection dir,
        uint critptno,
        get_cell_coords_ftor_t get_cell_coords_ftor)
{

  typedef typename MSComplex<id_t>::critical_point  cp_t;
  typedef typename std::map<id_t,uint>              id_cp_map_t;
  typedef typename std::vector<id_t>                idvec_t;
  typedef typename std::multiset<uint>              adj_t;

  ren = new ArrayRenderer<id_t,fn_t>();


  cp_t * src_cp = msc->m_cps[critptno];


  if(dir == GRADIENT_DIR_DOWNWARD)
  {
    std::for_each
        (src_cp->des_disc.begin(),
         src_cp->des_disc.end(),
         boost::bind ( add_point_to_renderer<id_t,fn_t,get_cell_coords_ftor_t>,_1,ren,get_cell_coords_ftor ));
  }
  else
  {
    std::for_each
        (src_cp->asc_disc.begin(),
         src_cp->asc_disc.end(),
         boost::bind ( add_point_to_renderer<id_t,fn_t,get_cell_coords_ftor_t>,_1,ren,get_cell_coords_ftor ));
  }



  ren->prepare_for_render();
  ren->sqeeze();

}




static double colorset[][3] =
{
  {0.000000, 0.000000, 0.000000},
  {0.184300, 0.309800, 0.309800},
  {0.329400, 0.329400, 0.329400},
  {0.658800, 0.658800, 0.658800},
  {0.752900, 0.752900, 0.752900},
  {0.184300, 0.184300, 0.309800},
  {0.137300, 0.137300, 0.556900},
  {0.196100, 0.196100, 0.800000},
  {0.000000, 0.000000, 1.000000},
  {0.000000, 0.498000, 1.000000},
  {0.196100, 0.600000, 0.800000},
  {0.258800, 0.258800, 0.435300},
  {0.419600, 0.137300, 0.556900},
  {0.498000, 0.000000, 1.000000},
  {0.560800, 0.560800, 0.737200},
  {0.749000, 0.847100, 0.847100},
  {0.137300, 0.419600, 0.556900},
  {0.678400, 0.917600, 0.917600},
  {0.000000, 1.000000, 1.000000},
  {0.000000, 1.000000, 0.498000},
  {0.439200, 0.858800, 0.576500},
  {0.196100, 0.800000, 0.600000},
  {0.439200, 0.858800, 0.858800},
  {0.439200, 0.576500, 0.858800},
  {0.137300, 0.556900, 0.419600},
  {0.258800, 0.435300, 0.258800},
  {0.184300, 0.309800, 0.184300},
  {0.137300, 0.556900, 0.137300},
  {0.498000, 1.000000, 0.000000},
  {0.000000, 1.000000, 0.000000},
  {0.196100, 0.800000, 0.196100},
  {0.560800, 0.737200, 0.560800},
  {0.576500, 0.858800, 0.439200},
  {0.600000, 0.800000, 0.196100},
  {0.419600, 0.556900, 0.137300},
  {0.309800, 0.309800, 0.184300},
  {0.309800, 0.184300, 0.184300},
  {0.556900, 0.137300, 0.137300},
  {0.647100, 0.164700, 0.164700},
  {0.800000, 0.196100, 0.196100},
  {1.000000, 0.000000, 0.000000},
{1.000000, 0.000000, 0.498000},
{1.000000, 0.000000, 0.498000},
{0.800000, 0.196100, 0.600000},
{0.858800, 0.439200, 0.576500},
{0.623500, 0.372600, 0.623500},
{0.600000, 0.196100, 0.800000},
{0.576500, 0.439200, 0.858800},
{0.858800, 0.439200, 0.858800},
{0.917600, 0.678400, 0.917600},
{0.847100, 0.749000, 0.847100},
{1.000000, 0.000000, 1.000000},
{0.556900, 0.137300, 0.419600},
{0.309800, 0.184300, 0.309800},
{0.435300, 0.258800, 0.258800},
{0.737200, 0.560800, 0.560800},
{0.858800, 0.576500, 0.439200},
{0.800000, 0.498000, 0.196100},
{1.000000, 0.498000, 0.000000},
{1.000000, 1.000000, 0.000000},
{0.858800, 0.858800, 0.439200},
{0.917600, 0.917600, 0.678400},
{0.556900, 0.419600, 0.137300},
{0.623500, 0.623500, 0.372600},
{0.847100, 0.847100, 0.749000},
{0.988200, 0.988200, 0.988200}
};



template <typename id_t,typename fn_t>
    ArrayRenderer<id_t,fn_t> * createMsQuadRenderer_func
    (
        IDiscreteDataset_renderable<id_t,fn_t> *ds,
        MSComplex<id_t> *msc
        )
{
  std::vector<std::vector<id_t> *> quadpatches;

  createQuadPatches ( msc,ds,quadpatches );

  ArrayRenderer<id_t,fn_t> * ren = new ArrayRenderer<id_t,fn_t>();

  fn_t x,y,z;

  for ( uint i = 0 ; i < quadpatches.size();i++ )
  {
    double *color = colorset[ ( i+2 ) % ( sizeof ( colorset ) / ( 3*sizeof ( double ) ) ) ];

    std::vector<id_t> *patch = quadpatches[i];

    for ( uint j = 0 ; j < patch->size(); j++ )
    {
      id_t cellid = ( *patch ) [j];

      if ( ds->getCellDim ( cellid ) == 2 )
      {

        id_t pts[20];

        uint pts_ct = ds->getCellPoints ( cellid,pts );

        for ( uint k = 0 ; k < pts_ct;k++ )
        {

          ds->getCellCoords ( pts[k],x,y,z );
          ren->add_vertex ( pts[k],x,y,z );
          ren->set_vertex_color ( pts[k],color );
        }

        if ( pts_ct == 3 )
          ren->add_triangle ( pts[0],pts[1],pts[2] );
        else if ( pts_ct == 4 )
          ren->add_quad ( pts[0],pts[1],pts[2],pts[3] );

      }
    }

  }

  for_each ( quadpatches.begin(),quadpatches.end(),delete_ftor<std::vector<id_t> > );

  ren->prepare_for_render();
  ren->sqeeze();

  return ren;
}

#endif
